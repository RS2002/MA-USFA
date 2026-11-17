"""Reconstruct the old TSC Homo/Hetero target-task protocol.

This module reproduces the old task generation and evaluation seed only. It
does not claim that the historical primitive-action composer is the repaired
strict-entry selector. A fresh manifest is required because repaired SF
checkpoints bind their task protocol and manifest hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LEGACY_TASK_VERSION = "ma-usfa-traffic-task-v2"
LEGACY_PROTOCOL_VERSION = "ma-usfa-paper-aligned-traffic-protocol-v1"
TASK_SEED = 11
TRAIN_TASK_SEED = 12
LEGACY_SCALE = (0.5, 1.5)
LEGACY_EVAL_SEED = 10_000_000
LEGACY_TEST_SEED = 20_000_000
DEFAULT_W = np.asarray([-1.0, -0.3, -1.5, 3.0, 0.0], dtype=np.float32)


def _sample_task(rng, base=DEFAULT_W, scale=LEGACY_SCALE):
    base = np.asarray(base, dtype=np.float32)
    lo, hi = map(float, scale)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("invalid task scale")
    return (np.sign(base) * np.abs(base) * rng.uniform(lo, hi, size=base.shape)).astype(np.float32)


def old_homogeneous_weight(seed: int = TASK_SEED) -> np.ndarray:
    """Return the old shared Homo vector using ``RandomState(seed)``."""
    return _sample_task(np.random.RandomState(int(seed)))


def old_heterogeneous_weights(node_count: int, seed: int = TASK_SEED) -> np.ndarray:
    """Return the old per-intersection Hetero matrix using one shared RNG."""
    if isinstance(node_count, bool) or int(node_count) <= 0:
        raise ValueError("node_count must be positive")
    rng = np.random.RandomState(int(seed))
    return np.stack([_sample_task(rng) for _ in range(int(node_count))], axis=0).astype(np.float32)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_legacy_protocol(node_ids: Sequence[str], anchors: Sequence[Sequence[float]],
                          task_seed: int = TASK_SEED,
                          train_task_seed: int = TRAIN_TASK_SEED,
                          train_traffic_seeds: Sequence[int] = tuple(range(10)),
                          validation_traffic_seed: int = LEGACY_EVAL_SEED,
                          test_traffic_seed: int = LEGACY_TEST_SEED) -> dict[str, Any]:
    """Build a V2 task file with old Homo/Hetero validation targets.

    Training uses a separate seed so the old validation targets are not fitted
    during SF or upper training. The source/SF reference sampling remains the
    repaired V2 contract and is declared explicitly.
    """
    ids = [str(value) for value in node_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("node_ids must be nonempty and unique")
    anchor_array = np.asarray(anchors, dtype=np.float32)
    if anchor_array.ndim != 2 or anchor_array.shape[1] != 5 or not np.isfinite(anchor_array).all():
        raise ValueError("anchors must have finite shape [K,5]")
    train_homo = old_homogeneous_weight(train_task_seed)
    train_hetero = old_heterogeneous_weights(len(ids), train_task_seed)
    validation_homo = old_homogeneous_weight(task_seed)
    validation_hetero = old_heterogeneous_weights(len(ids), task_seed)
    raw = {
        "version": LEGACY_TASK_VERSION,
        "source_anchors": anchor_array.tolist(),
        "source_sampling": {"temperature": 20.0, "score": "anchor_dot_task", "resample": "each_step"},
        "tasks": {
            "train": [
                {"id": "train_legacy_homo_seed12", "weights": train_homo.tolist()},
                {"id": "train_legacy_heterogeneous_seed12", "weights": train_hetero.tolist()},
            ],
            "validation": [
                {"id": "legacy_homo_seed11", "weights": validation_homo.tolist()},
                {"id": "legacy_heterogeneous_seed11", "weights": validation_hetero.tolist()},
            ],
        },
        "traffic_seeds": {
            "train": [int(value) for value in train_traffic_seeds],
            "validation": [int(validation_traffic_seed)],
            "test": [int(test_traffic_seed)],
        },
        "node_order": ids,
        "scope": "Old Homo/Hetero tasks with repaired frozen-source strict-entry selector; not historical residual composer.",
        "reference_semantics": "Repaired per-step teammate-task reference distribution; no selected-source or teammate-policy input.",
        "legacy_task_generation": {
            "task_seed": int(task_seed), "train_task_seed": int(train_task_seed),
            "rng": "numpy.random.RandomState", "scale": list(LEGACY_SCALE),
            "homo": "one shared task vector across all controlled nodes",
            "heterogeneous": "one independently sampled vector per controlled node",
            "effective_feature_weights": ["queue", "wait", "pressure", "speed"],
            "fairness_weight": 0.0, "legacy_eval_seed": int(validation_traffic_seed),
        },
    }
    json.dumps(raw, allow_nan=False)
    return raw


def write_protocol(path: str | Path, raw: dict[str, Any]) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {path}")
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return _file_hash(path)


def prepare_from_parent(parent_manifest: str | Path, output: str | Path,
                        task_seed: int = TASK_SEED,
                        train_task_seed: int = TRAIN_TASK_SEED) -> dict[str, Any]:
    """Create an isolated legacy-aligned protocol and manifest."""
    parent_path = Path(parent_manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output contains files; choose a fresh sibling directory")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("version") != "ma-usfa-tsc-server-manifest-v2":
        raise ValueError("parent must be a repaired V2 manifest")
    parent_files = {}
    for item in parent.get("files", []):
        if isinstance(item, dict) and item.get("role"):
            path = Path(str(item["path"]))
            if not path.is_absolute(): path = parent_path.parent / path
            path = path.resolve()
            if not path.is_file() or _file_hash(path) != item.get("sha256"):
                raise ValueError(f"parent declared file changed or missing: {path}")
            parent_files[str(item["role"])] = path
    network, routes = parent_files.get("network"), parent_files.get("routes")
    if network is None or routes is None: raise ValueError("parent needs network and routes")
    protocol_decl = parent.get("traffic_task_protocol", {})
    protocol_path = Path(str(protocol_decl.get("path", "")))
    if not protocol_path.is_absolute(): protocol_path = parent_path.parent / protocol_path
    protocol_path = protocol_path.resolve()
    if not protocol_path.is_file() or _file_hash(protocol_path) != protocol_decl.get("sha256"):
        raise ValueError("parent task protocol changed or missing")
    parent_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    node_ids = parent_protocol.get("node_order")
    anchors = parent_protocol.get("source_anchors", parent.get("source_anchors"))
    if not isinstance(node_ids, list) or not isinstance(anchors, list): raise ValueError("missing parent task identity")
    tasks = build_legacy_protocol(node_ids, anchors, task_seed=task_seed, train_task_seed=train_task_seed)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "protocol" / "legacy_homo_heterogeneous_tasks.json"
    task_hash = write_protocol(task_path, tasks)
    code_manifest = {
        "version": LEGACY_PROTOCOL_VERSION, "parent_manifest_sha256": _file_hash(parent_path),
        "parent_task_protocol_sha256": _file_hash(protocol_path),
        "task_generation": tasks["legacy_task_generation"],
        "current_repaired_method": "frozen source/SF + attention DDQN strict-entry selector",
        "historical_method_not_reproduced": "target-specific primitive-action residual composer",
    }
    code_path = output / "protocol" / "paper_aligned_code_manifest.json"
    code_path.write_text(json.dumps(code_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scenario = dict(parent.get("scenario", {}))
    rel = lambda p: Path(os.path.relpath(p, output)).as_posix()
    rel_network, rel_routes, rel_code, rel_task = rel(network), rel(routes), rel(code_path), rel(task_path)
    manifest = {
        "version": "ma-usfa-tsc-server-manifest-v2", "status": "READY_FOR_PILOT",
        "scenario_id": parent.get("scenario_id", "manhattan_28x7"), "scenario": scenario,
        "files": [{"role":"network","path":rel_network,"sha256":_file_hash(network)},
                  {"role":"routes","path":rel_routes,"sha256":_file_hash(routes)},
                  {"role":"code_manifest","path":rel_code,"sha256":_file_hash(code_path)}],
        "source_anchors": anchors,
        "traffic_task_protocol": {"version":LEGACY_TASK_VERSION,"path":rel_task,"sha256":task_hash},
        "method_contract": dict(parent.get("method_contract", {})), "protocol": {
            "protocol_id": LEGACY_PROTOCOL_VERSION, "code_sha256": _json_hash(code_manifest),
            "data_manifest_sha256": _json_hash({"network":_file_hash(network),"routes":_file_hash(routes)}),
            "train_split_sha256": _json_hash(tasks["tasks"]["train"]),
            "validation_split_sha256": _json_hash(tasks["tasks"]["validation"]),
            "training_seeds": [0], "traffic_seeds": tasks["traffic_seeds"],
        }, "resource_limits": dict(parent.get("resource_limits", {})),
        "test_accessed_before_formal_run": False, "formal_evaluation_enabled": False,
        "scope": "paper-aligned task protocol for repaired method; requires fresh V2 source/SF/upper artifacts",
        "legacy_alignment": {"task_seed":int(task_seed),"train_task_seed":int(train_task_seed),
                             "validation_traffic_seed":LEGACY_EVAL_SEED,
                             "horizon_steps":int(scenario.get("horizon_steps",60)),
                             "delta_time":float(scenario.get("delta_time",10)),
                             "requires_new_sf_and_upper":True},
        # Keep the short marker used by the evaluator as a separate explicit
        # field; legacy_alignment carries the full provenance record.
        "paper_alignment": {"task_seed": int(task_seed),
                             "validation_traffic_seed": int(LEGACY_EVAL_SEED),
                             "task_protocol": "legacy_homo_heterogeneous"},
    }
    manifest_path = output / "manifest_paper_aligned.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    recipe = {"version":LEGACY_PROTOCOL_VERSION,"parent_manifest_sha256":_file_hash(parent_path),
              "parent_task_protocol_sha256":_file_hash(protocol_path),"manifest_sha256":_file_hash(manifest_path),
              "task_protocol_sha256":task_hash,"prepared_files":{"manifest_paper_aligned.json":_file_hash(manifest_path),
              rel_task:task_hash,rel_code:_file_hash(code_path)},"requires_new_sf_and_upper":True}
    (output / "paper_aligned_recipe.json").write_text(json.dumps(recipe, indent=2)+"\n", encoding="utf-8")
    return {"manifest":manifest_path,"protocol":task_path,"recipe":output/"paper_aligned_recipe.json",
            "manifest_sha256":recipe["manifest_sha256"],"protocol_sha256":task_hash,
            "validation_task_ids":[row["id"] for row in tasks["tasks"]["validation"]],
            "requires_new_sf_and_upper":True}
