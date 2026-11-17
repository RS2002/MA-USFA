"""Bounded process-tree resource checks for development traffic runs."""
from __future__ import annotations

import json
import atexit
import os
from pathlib import Path
import shutil
import time
import threading

from .config import load_manifest


class ResourceLimitError(RuntimeError):
    pass


def configure_torch_threads(threads: int = 2, device: str = "cpu",
                            gpu_memory_fraction: float = 0.75) -> None:
    import torch
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # PyTorch permits this setting only before its first parallel job.
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not 0 < float(gpu_memory_fraction) <= 1:
            raise ValueError("gpu_memory_fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), device)


class ResourceGuard:
    """Check wall time, process-tree RSS, free RAM and free disk.

    This guard never signals an unrelated process. Callers close their own
    SUMO environment on exceptions. Checks are throttled to once a second;
    only compact snapshots are written, at most once every ten seconds.
    """
    def __init__(self, manifest, output_dir, check_interval: float = 1.0,
                 device: str = "cpu"):
        import psutil
        import torch
        self.psutil = psutil
        self.device = torch.device(device)
        self.manifest = (load_manifest(manifest)
                         if isinstance(manifest, (str, Path)) else manifest)
        self.limits = self.manifest.resource_limits
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "resources.jsonl"
        self.started = time.monotonic()
        self.last_check = self.last_log = -float("inf")
        self.interval = float(check_interval)
        self.peak_rss_gib = 0.0
        self.last = {}
        self._last_cpu = None
        self._failure = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.check(force=True)
        self._watcher = threading.Thread(target=self._watch, name="traffic-resource-guard", daemon=True)
        self._watcher.start()
        atexit.register(self.close)

    def check(self, force: bool = False) -> dict:
        with self._lock:
            if self._failure:
                raise ResourceLimitError(self._failure)
            return self._check(force)

    def _check(self, force: bool = False) -> dict:
        now = time.monotonic()
        wall = now - self.started
        if wall > float(self.limits["max_wall_seconds"]):
            raise ResourceLimitError("run wall-time limit reached; existing files preserved")
        if not force and now - self.last_check < self.interval:
            return self.last
        self.last_check = now
        process = self.psutil.Process(os.getpid())
        rss = process.memory_info().rss
        children = process.children(recursive=True)
        cpu_time = process.cpu_times()
        cpu_seconds = cpu_time.user + cpu_time.system
        for child in children:
            try:
                rss += child.memory_info().rss
                cpu_time = child.cpu_times()
                cpu_seconds += cpu_time.user + cpu_time.system
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        rss_gib = rss / 1024**3
        self.peak_rss_gib = max(self.peak_rss_gib, rss_gib)
        free_disk = shutil.disk_usage(self.root).free / 1024**3
        free_ram = self.psutil.virtual_memory().available / 1024**3
        cpu_percent = 0.0
        if self._last_cpu is not None:
            previous_time, previous_cpu = self._last_cpu
            cpu_percent = max(0., cpu_seconds - previous_cpu) / max(1e-6, now - previous_time) * 100.
        self._last_cpu = (now, cpu_seconds)
        self.last = {"elapsed_seconds": wall, "rss_gib": rss_gib,
                     "peak_rss_gib": self.peak_rss_gib,
                     "free_disk_gib": free_disk, "system_available_ram_gib": free_ram,
                     "process_tree_cpu_percent": cpu_percent,
                     "owned_child_processes": len(children)}
        failure = None
        if rss_gib > float(self.limits["max_rss_gib"]):
            failure = "process-tree memory limit reached"
        elif free_disk < float(self.limits["min_free_disk_gib"]):
            failure = "free disk is below the run limit"
        elif free_ram < float(self.limits.get("min_system_available_ram_gib", 2.0)):
            failure = "system available RAM is below the run limit"
        # Device availability is checked on the same monitor as CPU/RAM/disk;
        # this does not inspect or signal any other user's GPU processes.
        import torch
        if (self.device.type == "cuda" and torch.cuda.is_available()
                and torch.cuda.is_initialized()):
            free_gpu, total_gpu = torch.cuda.mem_get_info(self.device)
            self.last.update({"gpu_free_gib": free_gpu / 1024**3,
                              "gpu_total_gib": total_gpu / 1024**3,
                              "gpu_allocated_gib": torch.cuda.memory_allocated(self.device) / 1024**3,
                              "gpu_reserved_gib": torch.cuda.memory_reserved(self.device) / 1024**3,
                              "gpu_device": str(self.device)})
            if free_gpu / 1024**3 < float(self.limits.get("min_free_gpu_gib", .5)):
                failure = failure or "GPU free memory is below the run limit"
        if force or failure or now - self.last_log >= 10:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**self.last, "failure": failure}) + "\n")
            self.last_log = now
        if failure:
            raise ResourceLimitError(failure + "; existing files preserved")
        return self.last

    def _watch(self):
        while not self._stop.wait(max(1., self.interval)):
            try:
                self.check()
            except Exception as exc:
                self._failure = str(exc)
                try:
                    (self.root / "resource_failure.json").write_text(json.dumps({
                        "status": "STOPPED", "error": str(exc),
                        "snapshot": self.last, "old_files_preserved": True,
                        "stops_only_this_run_process_tree": True,
                    }, indent=2) + "\n", encoding="utf-8")
                finally:
                    self._stop_owned_run()
                return

    def _stop_owned_run(self):
        # A blocked TraCI read cannot reach the next cooperative check. Stop
        # only descendants of this runner so its read unblocks and finally
        # closes the environment. If the runner itself remains blocked, exit
        # after a grace period. Stage artifacts are saved atomically.
        children = self.psutil.Process(os.getpid()).children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        if self._stop.wait(5.):
            return
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        os._exit(3)

    def close(self):
        self._stop.set()
