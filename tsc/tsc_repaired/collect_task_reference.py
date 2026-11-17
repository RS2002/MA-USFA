"""Collect V2 task-conditioned SF reference transitions on training tasks.

All node sources are sampled independently each step from the declared
D(.|w_i).  Conditional on executed own z, this supplies off-policy one-step
samples for fixed-own-source evaluation with task-distributed teammates.
The sampled teammate labels are never used as SF observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_manifest, sha256_file
from .features import FEATURE_NAMES
from .task_sf import REFERENCE_SEMANTICS, REFERENCE_VERSION


def _check_guard(guard):
    if guard is not None:
        guard.check()


def collect_reference_v2(manifest_path, source_dir, output, seed=0,
                         episodes=5, device="cpu", guard=None):
    """Collect train-only data; no validation/test objective is sampled."""
    from .task_protocol_v2 import (load_protocol, source_sampling_probabilities,
                                   validate_task_weights)
    from .traffic_runtime import FrozenTrafficRuntime

    if int(episodes) <= 0:
        raise ValueError("reference episodes must be positive")
    manifest = load_manifest(manifest_path)
    manifest.require_stage("pilot")
    out = Path(output).expanduser().resolve()
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(f"reference output exists; preserve it or choose a new path: {out}")
    protocol = load_protocol(manifest_path)
    tasks = protocol.tasks("train")
    traffic_seeds = list(protocol.traffic_seeds("train"))
    if not tasks or not traffic_seeds:
        raise ValueError("reference collection needs training tasks and traffic seeds")
    if protocol.sampling_resample != "each_step":
        raise ValueError("V2 state SF requires independent teammate source draws each_step")
    rng = np.random.default_rng(int(seed))
    runtime = FrozenTrafficRuntime(manifest_path, source_dir, device=device)
    rows = {name: [] for name in (
        "state", "next_state", "weights", "entries", "features", "done",
        "time_fraction", "next_time_fraction", "task_id", "traffic_seed")}
    task_ids, used_traffic_seeds = [], []
    probability_min, probability_max = 1.0, 0.0
    try:
        for episode in range(int(episodes)):
            _check_guard(guard)
            task = tasks[int(rng.integers(len(tasks)))]
            weights = validate_task_weights(task["weights"], runtime.N)
            traffic_seed = int(traffic_seeds[episode % len(traffic_seeds)])
            task_id = str(task["id"])
            state = runtime.reset(weights, traffic_seed)
            probabilities = source_sampling_probabilities(
                weights, runtime.anchors, protocol.sampling_temperature)
            if probabilities.shape != (runtime.N, runtime.K):
                raise ValueError("source sampling distribution differs from runtime [N,K]")
            probability_min = min(probability_min, float(probabilities.min()))
            probability_max = max(probability_max, float(probabilities.max()))
            task_ids.append(task_id)
            used_traffic_seeds.append(traffic_seed)
            for step in range(runtime.horizon):
                _check_guard(guard)
                entries = np.asarray([rng.choice(runtime.K, p=p) for p in probabilities], dtype=np.int64)
                result = runtime.step(entries)
                # The reference experiment is finite horizon, even if an
                # environment wrapper does not announce truncation itself.
                done = bool(result.done or step + 1 >= runtime.horizon)
                rows["state"].append(np.asarray(state, dtype=np.float32).copy())
                rows["next_state"].append(np.asarray(result.state, dtype=np.float32).copy())
                rows["weights"].append(weights.copy())
                rows["entries"].append(entries)
                rows["features"].append(np.asarray(result.features, dtype=np.float32).copy())
                rows["done"].append(done)
                rows["time_fraction"].append(step / runtime.horizon)
                rows["next_time_fraction"].append((step + 1) / runtime.horizon)
                rows["task_id"].append(task_id)
                rows["traffic_seed"].append(traffic_seed)
                state = result.state
                if done:
                    break
        if not rows["state"]:
            raise ValueError("reference collection produced no transitions")
        _check_guard(guard)
        frozen_audit = runtime.audit_frozen()
        arrays = {name: np.asarray(value) for name, value in rows.items()}
        for name in ("state", "next_state", "weights", "features",
                     "time_fraction", "next_time_fraction"):
            arrays[name] = arrays[name].astype(np.float32, copy=False)
            if not np.isfinite(arrays[name]).all():
                raise FloatingPointError(f"non-finite reference array: {name}")
        arrays["entries"] = arrays["entries"].astype(np.int64, copy=False)
        source_visits = np.bincount(arrays["entries"].reshape(-1), minlength=runtime.K)
        arrays["source_visit_counts"] = source_visits
        arrays["done"] = arrays["done"].astype(np.bool_, copy=False)
        arrays["version"] = np.asarray(REFERENCE_VERSION)
        arrays["source_count"] = np.asarray(runtime.K, dtype=np.int64)
        arrays["source_hashes_json"] = np.asarray(json.dumps(runtime.source_hashes, sort_keys=True))
        arrays["source_sampling_temperature"] = np.asarray(protocol.sampling_temperature)
        arrays["reference_semantics"] = np.asarray(REFERENCE_SEMANTICS)
        arrays["manifest_sha256"] = np.asarray(sha256_file(Path(manifest_path)))
        arrays["task_protocol_sha256"] = np.asarray(
            str(manifest.raw.get("traffic_task_protocol", {}).get("sha256", "")))
        arrays["protocol_sha256"] = np.asarray(protocol.fingerprint)
        arrays["scenario_hashes_json"] = np.asarray(json.dumps(runtime.spec.files.hashes(), sort_keys=True))
        arrays["feature_names"] = np.asarray(FEATURE_NAMES)
        # Bind optional graph context to the same node order used in data.
        if getattr(runtime, "adjacency", None) is not None:
            arrays["adjacency"] = np.asarray(runtime.adjacency, dtype=np.bool_)
        arrays["node_ids"] = np.asarray(runtime.ids)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(out.name + ".partial")
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temp.replace(out)
        finally:
            if temp.exists():
                temp.unlink()
        report = {
            "status": "PASS", "version": REFERENCE_VERSION,
            "split": "train", "seed": int(seed), "episodes": int(episodes),
            "transitions": len(rows["state"]), "source_count": int(runtime.K),
            "node_count": int(runtime.N), "source_hashes": runtime.source_hashes,
            "protocol_sha256": protocol.fingerprint,
            "scenario_hashes": runtime.spec.files.hashes(),
            "feature_names": list(FEATURE_NAMES),
            "task_ids": task_ids, "traffic_seeds": used_traffic_seeds,
            "reference_semantics": REFERENCE_SEMANTICS,
            "source_sampling_temperature": float(protocol.sampling_temperature),
            "source_visit_counts": source_visits.tolist(),
            "all_sources_observed": bool(np.all(source_visits > 0)),
            "source_probability_min": probability_min,
            "source_probability_max": probability_max,
            "actual_teammate_entries_are_inputs": False,
            "output": str(out), "sha256": sha256_file(out),
            "test_accessed": False, "test_updates": 0,
            "frozen_audit": frozen_audit,
        }
        out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        import torch
        from .resources import ResourceGuard, configure_torch_threads
        manifest = load_manifest(args.manifest)
        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        configure_torch_threads(manifest.resource_limits["threads"], device)
        guard = ResourceGuard(manifest, Path(args.output).parent, device=device)
        result = collect_reference_v2(args.manifest, args.source_dir, args.output,
                                      args.seed, args.episodes, device, guard=guard)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, FloatingPointError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc),
                          "test_accessed": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
