"""Resource protection checks; no real processes are signalled or trained."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Load Torch before injecting the optional process-monitor dependency. Torch
# may inspect installed modules during its own lazy imports.
import torch
import torch._dynamo

from .resources import ResourceGuard, ResourceLimitError


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.limits = {"max_wall_seconds": 60, "max_rss_gib": 1.,
                       "min_free_disk_gib": 1., "min_system_available_ram_gib": 1.}
        self.process = SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=1024**2),
            cpu_times=lambda: SimpleNamespace(user=.1, system=.1),
            children=lambda recursive=True: [])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        fake_psutil = ModuleType('psutil')
        fake_psutil.Process = Mock(return_value=self.process)
        fake_psutil.virtual_memory = Mock(
            return_value=SimpleNamespace(available=8 * 1024**3))
        fake_psutil.NoSuchProcess = type('NoSuchProcess', (Exception,), {})
        fake_psutil.AccessDenied = type('AccessDenied', (Exception,), {})
        self.stack.enter_context(patch.dict(sys.modules, {'psutil': fake_psutil}))
        self.disk = self.stack.enter_context(patch('tsc_repaired.resources.shutil.disk_usage',
                              return_value=SimpleNamespace(free=10 * 1024**3)))

    def test_low_disk_stops_before_starting_a_watcher(self):
        self.disk.return_value.free = 0
        with self.assertRaisesRegex(ResourceLimitError, 'free disk'):
            ResourceGuard(SimpleNamespace(resource_limits=self.limits), self.root)

    def test_process_tree_rss_is_counted_and_snapshot_written(self):
        child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=2 * 1024**2),
                                cpu_times=lambda: SimpleNamespace(user=.1, system=.1))
        self.process.children = lambda recursive=True: [child]
        guard = ResourceGuard(SimpleNamespace(resource_limits=self.limits), self.root)
        self.addCleanup(guard.close)
        self.assertAlmostEqual(guard.last['rss_gib'], 3 / 1024.)
        self.assertEqual(guard.last['owned_child_processes'], 1)
        self.assertTrue(guard.log_path.is_file())
        guard.close()

    def test_cuda_queries_use_the_selected_device(self):
        available = Mock(return_value=True)
        initialized = Mock(return_value=True)
        free = Mock(return_value=(4 * 1024**3, 8 * 1024**3))
        allocated = Mock(return_value=1024**3)
        reserved = Mock(return_value=2 * 1024**3)
        with patch.multiple(torch.cuda, is_available=available,
                            is_initialized=initialized, mem_get_info=free,
                            memory_allocated=allocated, memory_reserved=reserved):
            guard = ResourceGuard(SimpleNamespace(resource_limits=self.limits),
                                  self.root, device="cuda:1")
            self.addCleanup(guard.close)
            guard.close()
            free.assert_called_once_with(torch.device("cuda:1"))
            allocated.assert_called_once_with(torch.device("cuda:1"))
            reserved.assert_called_once_with(torch.device("cuda:1"))
            self.assertEqual(guard.last["gpu_device"], "cuda:1")

    def test_cpu_run_does_not_query_cuda(self):
        available, free, allocated, reserved = (Mock() for _ in range(4))
        with patch.multiple(torch.cuda, is_available=available,
                            mem_get_info=free, memory_allocated=allocated,
                            memory_reserved=reserved):
            guard = ResourceGuard(SimpleNamespace(resource_limits=self.limits),
                                  self.root, device="cpu")
            self.addCleanup(guard.close)
            guard.close()
            for query in (available, free, allocated, reserved):
                query.assert_not_called()
            self.assertNotIn("gpu_device", guard.last)

    def test_watchdog_detects_timeout_without_another_main_thread_check(self):
        called = threading.Event()
        with patch.object(ResourceGuard, '_stop_owned_run', side_effect=called.set) as stop:
            guard = ResourceGuard(SimpleNamespace(resource_limits=self.limits), self.root)
            self.addCleanup(guard.close)
            self.limits['max_wall_seconds'] = .01
            # The main thread makes no check call; a blocked SUMO call would
            # therefore still be noticed by the independent watchdog.
            self.assertTrue(called.wait(3.))
            guard.close()
            self.assertEqual(stop.call_count, 1)
            self.assertEqual(json.loads((self.root / 'resource_failure.json').read_text())['status'], 'STOPPED')
            with self.assertRaisesRegex(ResourceLimitError, 'wall-time'):
                guard.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
