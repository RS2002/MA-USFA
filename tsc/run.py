"""Train/evaluate the current strict-entry traffic attention-DDQN pipeline.

Relative CLI paths are resolved from the working directory.  This release
creates fresh scenario-bound sources; it never uses the historical sf_psi.pth.
All expensive stages have hash-checked completion receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
VERSION = "ma-usfa-traffic-release-v1"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")
    temp.replace(path)


def _rel(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def _code_hashes():
    return {_rel(p, HERE): _sha(p) for p in sorted(HERE.rglob("*.py"))
            if "runs" not in p.relative_to(HERE).parts
            and "__pycache__" not in p.parts}


def prepare(scene, output, settings):
    """Freeze paths, bytes, budgets and task split without launching SUMO."""
    from paper_aligned_followup.legacy_tasks import build_legacy_protocol
    from tsc_repaired.task_protocol_v2 import default_anchors, load_protocol
    from tsc_repaired.scenario import inspect_network, inspect_routes
    from tsc_repaired.config import V2_CONTRACT, load_manifest

    scene, output = Path(scene).resolve(), Path(output).resolve()
    net, routes = scene / "manhattan_28x7.net.xml", scene / "manhattan_28x7.rou.xml"
    ids = inspect_network(net).controlled_signal_ids
    inspect_routes(routes)
    if not ids:
        raise ValueError("network has no controlled signals")
    inputs = {"version": VERSION, "settings": settings,
              "network_sha256": _sha(net), "routes_sha256": _sha(routes),
              "code_sha256": _code_hashes()}
    fingerprint = _hash(inputs)
    manifest = output / "manifest.json"
    receipt = output / "recipe.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError("scene, code or settings changed; choose a new --output")
        for name, digest in saved.get("prepared_files", {}).items():
            p = output / name
            if not p.is_file() or _sha(p) != digest:
                raise RuntimeError("prepared artifacts changed; choose a new --output")
        load_manifest(manifest)
        load_protocol(manifest, ids)
        return manifest, fingerprint
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output contains files without a valid recipe; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    protocol = build_legacy_protocol(ids, default_anchors().tolist())
    tasks = output / "protocol" / "tasks.json"
    code = output / "protocol" / "code.json"
    _write(tasks, protocol)
    _write(code, inputs["code_sha256"])
    raw = {
        "version": "ma-usfa-tsc-server-manifest-v2", "status": "READY_FOR_PILOT",
        "scenario_id": "manhattan_28x7",
        "scenario": {"delta_time": 10, "horizon_steps": 60, "yellow_time": 2,
                     "min_green": 5, "max_green": 50, "scale": 2.0},
        "files": [{"role": role, "path": _rel(p, output), "sha256": _sha(p)}
                  for role, p in (("network", net), ("routes", routes), ("code_manifest", code))],
        "source_anchors": protocol["source_anchors"],
        "traffic_task_protocol": {"version": protocol["version"], "path": _rel(tasks, output),
                                  "sha256": _sha(tasks)},
        "method_contract": {key: True for key in V2_CONTRACT},
        "protocol": {"protocol_id": VERSION, "code_sha256": _sha(code),
                     "data_manifest_sha256": _hash({"network": _sha(net), "routes": _sha(routes)}),
                     "train_split_sha256": _hash(protocol["tasks"]["train"]),
                     "validation_split_sha256": _hash(protocol["tasks"]["validation"]),
                     "training_seeds": [settings["source_seed"], settings["upper_seed"]],
                     "traffic_seeds": protocol["traffic_seeds"]},
        "resource_limits": {"threads": settings["threads"], "max_wall_seconds": settings["max_wall_seconds"],
                            "max_rss_gib": settings["max_rss_gib"],
                            "min_free_disk_gib": settings["min_free_disk_gib"],
                            "min_system_available_ram_gib": 2., "gpu_memory_fraction": .75,
                            "min_free_gpu_gib": .5},
        "paper_alignment": {"task_seed": 11, "validation_traffic_seed": 10000000},
        "test_accessed_before_formal_run": False, "formal_evaluation_enabled": False,
        "scope": "Legacy-aligned Homo/Hetero validation; fresh source training, frozen lower and attention-DDQN.",
    }
    _write(manifest, raw)
    load_manifest(manifest)
    load_protocol(manifest, ids)
    _write(receipt, {"fingerprint": fingerprint, "inputs": inputs,
                     "prepared_files": {_rel(p, output): _sha(p) for p in (manifest, tasks, code)}})
    return manifest, fingerprint


def _stage(name, output, fingerprint, inputs, artifacts, action):
    """Only reuse a verified complete stage; partial stages are never retried."""
    receipt = output / "stages" / (name + ".json")
    identity = _hash({"run": fingerprint, "inputs": inputs})
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("status") != "PASS" or saved.get("fingerprint") != identity:
            raise RuntimeError(name + " receipt/input mismatch; preserve run and choose a new output")
        hashes = saved.get("artifact_hashes", {})
        if any(_rel(p, output) not in hashes for p in artifacts):
            raise RuntimeError(name + " receipt lacks required artifacts")
        if any(not (output / n).is_file() or _sha(output / n) != h for n, h in hashes.items()):
            raise RuntimeError(name + " saved artifact changed")
        return saved["report"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent or interrupted work from being
    # mistaken for an unstarted stage, even before its first artifact exists.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"status": "STARTED", "fingerprint": identity}, stream)
    try:
        report = action()
        if report.get("status") != "PASS":
            raise RuntimeError(name + " did not pass")
        if any(not p.is_file() for p in artifacts):
            raise RuntimeError(name + " did not create required artifacts")
        _write(receipt, {"status": "PASS", "fingerprint": identity, "report": report,
                         "artifact_hashes": {_rel(p, output): _sha(p) for p in artifacts}})
        return report
    except BaseException as exc:
        _write(receipt, {"status": "FAILED", "fingerprint": identity,
                         "error": str(exc), "automatic_retry": False})
        raise


def execute(args):
    import torch
    from tsc_repaired.resources import ResourceGuard, configure_torch_threads
    from tsc_repaired.config import load_manifest
    from tsc_repaired.train_sources_v2 import train_sources_v2
    from tsc_repaired.collect_task_reference import collect_reference_v2
    from tsc_repaired.train_task_sf import train_task_sf
    from tsc_repaired.traffic_runtime import FrozenTrafficRuntime
    from development_followup.train_budget import train_budget_upper
    from paper_aligned_followup.evaluate import evaluate_paper_aligned

    if args.upper_episodes > 30:
        raise ValueError("current evaluated upper recipe supports at most 30 episodes")
    if args.replays > 10:
        raise ValueError("replays must not exceed 10")
    output = Path(args.output).resolve()
    settings = {name: getattr(args, name) for name in (
        "source_seed", "upper_seed", "source_episodes", "reference_episodes", "sf_steps",
        "upper_episodes", "replays", "threads", "max_wall_seconds", "max_rss_gib",
        "min_free_disk_gib", "device")}
    manifest, fingerprint = prepare(args.scene_dir, output, settings)
    if args.stage == "prepare":
        return {"status": "PASS", "stage": "prepare", "manifest": _rel(manifest, Path.cwd()),
                "launches_sumo": False, "launches_training": False, "test_accessed": False}
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    declared = load_manifest(manifest)
    configure_torch_threads(args.threads, device)
    guard = ResourceGuard(declared, output / "logs", device=device)
    sources = output / "sources"
    source_paths = [sources / ("source_%03d.pt" % i) for i in range(5)]
    reference = output / "reference_train.npz"
    sf = output / "sf" / "task_sf.pt"
    upper = output / "upper" / ("upper_ep%03d.pt" % args.upper_episodes)
    evaluation = output / "evaluation_validation.json"
    order = ("sources", "reference", "sf", "upper", "evaluate")
    chosen = order if args.stage == "all" else (args.stage,)
    try:
        for stage in chosen:
            guard.check()
            if stage == "sources":
                _stage(stage, output, fingerprint, {}, source_paths + [sources / "train_sources_v2.json"],
                       lambda: train_sources_v2(manifest, sources, seed=args.source_seed,
                                                episodes=args.source_episodes, device=device, guard=guard))
                continue
            if stage in {"reference", "upper", "evaluate"}:
                if not all(p.is_file() for p in source_paths):
                    raise FileNotFoundError("frozen sources missing; run --stage sources first")
                with FrozenTrafficRuntime(manifest, sources, device=device) as runtime:
                    if not runtime.source_scenario_binding_verified:
                        raise ValueError("release pipeline requires scenario-bound sources; train fresh sources")
            source_hashes = {p.name: _sha(p) for p in source_paths} if all(p.is_file() for p in source_paths) else {}
            if stage == "reference":
                _stage(stage, output, fingerprint, {"sources": source_hashes}, [reference, reference.with_suffix(".json")],
                       lambda: collect_reference_v2(manifest, sources, reference, seed=args.source_seed,
                                                    episodes=args.reference_episodes, device=device, guard=guard))
            elif stage == "sf":
                _stage(stage, output, fingerprint, {"reference": _sha(reference)}, [sf, sf.parent / "train_task_sf.json"],
                       lambda: train_task_sf(manifest, reference, sf.parent, seed=args.source_seed,
                                             steps=args.sf_steps, device=device, guard=guard))
            elif stage == "upper":
                _stage(stage, output, fingerprint, {"sources": source_hashes, "sf": _sha(sf)},
                       [upper, upper.parent / "train_budget.json"],
                       lambda: train_budget_upper(manifest, sources, sf, upper.parent, seed=args.upper_seed,
                                                  checkpoints=(args.upper_episodes,), epsilon_decay_transitions=480,
                                                  device=device, guard=guard))
            elif stage == "evaluate":
                _stage(stage, output, fingerprint, {"sources": source_hashes, "sf": _sha(sf), "upper": _sha(upper)},
                       [evaluation], lambda: evaluate_paper_aligned(
                           manifest, sources, sf, {"ma_usfa_attention_ddqn": upper}, evaluation,
                           replays=args.replays, device=device, guard=guard))
        report = {"status": "PASS", "version": VERSION, "stage": args.stage,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "test_accessed": False, "evaluation_updates": 0,
                  "source_scenario_binding_verified": True if args.stage in ("all", "reference", "upper", "evaluate") else None,
                  "resources": guard.check(force=True),
                  "evaluation": _rel(evaluation, Path.cwd()) if evaluation.exists() else None,
                  "scope": "Legacy-aligned validation; same-seed replays are not independent traffic draws."}
        _write(output / "pipeline_status.json", report)
        return report
    finally:
        guard.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", default="../data/raw_data/manhattan_28x7")
    parser.add_argument("--output", default="runs/traffic_ddqn_seed100")
    parser.add_argument("--stage", choices=("prepare", "sources", "reference", "sf", "upper", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--upper-seed", type=int, default=100)
    for name, value in (("source-episodes", 5), ("reference-episodes", 20), ("sf-steps", 200),
                        ("upper-episodes", 30), ("replays", 5), ("threads", 2),
                        ("max-wall-seconds", 43200)):
        parser.add_argument("--" + name, type=_positive, default=value)
    parser.add_argument("--max-rss-gib", type=_positive_float, default=8.)
    parser.add_argument("--min-free-disk-gib", type=_positive_float, default=20.)
    args = parser.parse_args(argv)
    try:
        if args.source_seed < 0 or args.upper_seed < 0:
            raise ValueError("training seeds must be nonnegative")
        result = execute(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc),
                          "original_artifacts_preserved": True, "automatic_retry": False},
                         indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
