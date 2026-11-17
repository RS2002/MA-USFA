"""Read frozen traffic tasks without using an agent's selected source as input.

The protocol file contains training and validation tasks. Test objectives must
be stored in a separate hash-bound file; loading this module or the protocol
does not open that file. Source indices may be sampled from the declared
anchor/task softmax during reference collection. Those indices are execution
labels, not a substitute for teammate task weights in an SF query.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Manifest, load_manifest, resolve_manifest_path, sha256_file
from .features import DEFAULT_W, FEATURE_DIM


VERSION = "ma-usfa-traffic-task-v2"
SPLITS = ("train", "validation", "test")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid task protocol JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("task protocol must be a JSON object")
    return value


def _checked_hash(path: Path, expected: Any) -> str:
    value = str(expected).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("task protocol declaration requires a SHA-256 digest")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != value:
        raise ValueError(f"task protocol SHA-256 mismatch: {path}")
    return actual


def _weight_array(weights: Any) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.asarray(weights, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("task weights must be a numeric vector or matrix") from exc
    if value.ndim == 1 and value.shape == (FEATURE_DIM,):
        pass
    elif value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == FEATURE_DIM:
        pass
    else:
        raise ValueError(f"task weights must have shape [{FEATURE_DIM}] or [N,{FEATURE_DIM}]")
    if not np.isfinite(value).all():
        raise ValueError("task weights must be finite float32 values")
    return value.copy()


def validate_task_weights(weights: Any, node_count: int) -> np.ndarray:
    """Return a finite ``[N,5]`` matrix in the caller's declared node order."""
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)) or node_count <= 0:
        raise ValueError("node_count must be a positive integer")
    value = _weight_array(weights)
    if value.ndim == 1:
        return np.repeat(value[None], int(node_count), axis=0)
    if value.shape[0] != int(node_count):
        raise ValueError(f"task has {value.shape[0]} rows, expected {node_count} controlled nodes")
    return value


def _anchor_array(value: Any) -> np.ndarray:
    result = _weight_array(value)
    if result.ndim != 2:
        raise ValueError("source_anchors must have shape [K,5]")
    return result


def default_anchors() -> np.ndarray:
    """Preserve the original traffic source anchors, independent of runners."""
    anchors = [DEFAULT_W.copy()]
    base, signs = np.abs(DEFAULT_W), np.sign(DEFAULT_W)
    for index in range(4):
        value = base.copy()
        value[index] *= 3.0
        anchors.append((signs * value).astype(np.float32))
    return np.stack(anchors)


def source_sampling_probabilities(weights: Any, anchors: Any,
                                  temperature: float = 0.5) -> np.ndarray:
    """Declared ``D_z(.|w)``: softmax of signed anchor/task dot products.

    Traffic reward weights use negative costs and positive speed, so their
    signs are retained. This distribution never uses a selected source index
    or an observed teammate action. Its temperature belongs to the protocol.
    """
    value = _weight_array(weights)
    if value.ndim == 1:
        value = value[None]
    anchor_values = _anchor_array(anchors)
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("source sampling temperature must be finite and positive")
    scores = value.astype(np.float64) @ anchor_values.astype(np.float64).T
    scores -= scores.max(axis=-1, keepdims=True)
    # Large negative logits may underflow to zero; a row maximum remains 1.
    with np.errstate(over="ignore", under="ignore"):
        probabilities = np.exp(scores / float(temperature))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("source sampling produced non-finite probabilities")
    return probabilities


def _weight_key(weights: np.ndarray) -> tuple:
    """Canonicalize a homogeneous vector and its repeated-row matrix alike."""
    value = np.asarray(weights, dtype=np.float32)
    if value.ndim == 2 and np.all(value == value[0]):
        value = value[0]
    value = value.copy()
    value[value == 0] = 0.0  # signed zero does not identify a different task
    return tuple(value.shape), value.tobytes(order="C")


def _task_rows(value: Any, split: str, node_count: int | None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"tasks.{split} must be a nonempty list")
    result = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"every {split} task must have a nonempty string id")
        if "weights" not in row:
            raise ValueError(f"task {row['id']!r} has no weights")
        weights = (_weight_array(row["weights"]) if node_count is None
                   else validate_task_weights(row["weights"], node_count))
        weights.setflags(write=False)
        result.append({"id": row["id"], "weights": weights})
    return result


def _validate_distinct(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    ids, weights = {}, {}
    row_counts = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            ident = row["id"]
            if ident in ids:
                raise ValueError(f"duplicate task id {ident!r} in {ids[ident]} and {split}")
            ids[ident] = split
            key = _weight_key(row["weights"])
            if key in weights:
                other_split, other_id = weights[key]
                raise ValueError(
                    f"duplicate task weights in {other_split}/{other_id} and {split}/{ident}")
            weights[key] = (split, ident)
            if row["weights"].ndim == 2:
                row_counts.add(row["weights"].shape[0])
    if len(row_counts) > 1:
        raise ValueError("heterogeneous task matrices disagree on controlled node count")


def _seeds(raw: Any, split: str, required: bool = True) -> tuple[int, ...]:
    if raw is None and not required:
        return ()
    if not isinstance(raw, list) or (required and not raw):
        raise ValueError(f"traffic_seeds.{split} must be a nonempty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in raw):
        raise ValueError(f"traffic_seeds.{split} must contain nonnegative integers")
    if len(set(raw)) != len(raw):
        raise ValueError(f"traffic_seeds.{split} contains duplicates")
    return tuple(raw)


class TrafficTaskProtocol:
    """Validated training/validation tasks and an unopened test declaration."""

    version = VERSION

    def __init__(self, manifest: Manifest, path: Path, raw: dict[str, Any],
                 fingerprint: str, node_ids: Sequence[str] | None):
        self.manifest = manifest
        self.path = path
        self.fingerprint = fingerprint
        self._raw = copy.deepcopy(raw)
        if raw.get("version") != VERSION:
            raise ValueError(f"task protocol version must be {VERSION!r}")

        declared_order = raw.get("node_order")
        if declared_order is not None:
            if (not isinstance(declared_order, list) or not declared_order
                    or any(not isinstance(x, str) or not x for x in declared_order)
                    or len(set(declared_order)) != len(declared_order)):
                raise ValueError("node_order must be a nonempty list of unique string ids")
        if node_ids is not None:
            ids = tuple(str(x) for x in node_ids)
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("runtime node_ids must be nonempty and unique")
            if declared_order is not None and tuple(declared_order) != ids:
                raise ValueError("protocol node_order differs from the runtime node order")
            self.node_ids = ids
            node_count = len(ids)
        else:
            self.node_ids = tuple(declared_order) if declared_order is not None else None
            # Preserve homogeneous input vectors if the caller supplied no ids.
            node_count = None

        protocol_anchors = raw.get("source_anchors")
        manifest_anchors = manifest.raw.get("source_anchors")
        if protocol_anchors is None and manifest_anchors is None:
            self.anchors = default_anchors()
        elif protocol_anchors is None:
            self.anchors = _anchor_array(manifest_anchors)
        else:
            self.anchors = _anchor_array(protocol_anchors)
            if manifest_anchors is not None:
                other = _anchor_array(manifest_anchors)
                if self.anchors.shape != other.shape or not np.array_equal(self.anchors, other):
                    raise ValueError("source_anchors differ between manifest and task protocol")
        self.anchors.setflags(write=False)

        sampling = raw.get("source_sampling")
        if not isinstance(sampling, dict) or "temperature" not in sampling:
            raise ValueError("source_sampling.temperature must be explicitly declared")
        if sampling.get("score", "anchor_dot_task") != "anchor_dot_task":
            raise ValueError("source_sampling.score must be anchor_dot_task")
        if sampling.get("resample", "each_step") != "each_step":
            raise ValueError("this reference protocol requires source resampling each_step")
        self.sampling_temperature = float(sampling["temperature"])
        # Validate the temperature through the same public sampling function.
        source_sampling_probabilities(DEFAULT_W[None], self.anchors, sampling["temperature"])
        self.sampling_resample = "each_step"

        task_raw = raw.get("tasks")
        if not isinstance(task_raw, dict):
            raise ValueError("task protocol requires tasks.train and tasks.validation")
        if "test" in task_raw:
            raise ValueError("inline test tasks are forbidden; declare sealed_test_tasks instead")
        self._tasks = {
            split: _task_rows(task_raw.get(split), split, node_count)
            for split in ("train", "validation")
        }
        if declared_order is not None:
            for rows in self._tasks.values():
                for row in rows:
                    if row["weights"].ndim == 2 and row["weights"].shape[0] != len(declared_order):
                        raise ValueError("task matrix rows differ from protocol node_order")
        _validate_distinct(self._tasks)

        seed_raw = raw.get("traffic_seeds")
        if not isinstance(seed_raw, dict):
            raise ValueError("task protocol requires traffic_seeds declarations")
        self._traffic_seeds = {
            split: _seeds(seed_raw.get(split), split, required=split != "test")
            for split in SPLITS
        }
        seen_seeds = {}
        for split, seeds in self._traffic_seeds.items():
            for seed in seeds:
                if seed in seen_seeds:
                    raise ValueError(f"traffic seed {seed} overlaps {seen_seeds[seed]} and {split}")
                seen_seeds[seed] = split

        self._sealed_test = raw.get("sealed_test_tasks")
        if self._sealed_test is not None:
            if (not isinstance(self._sealed_test, dict)
                    or not isinstance(self._sealed_test.get("path"), str)
                    or not self._sealed_test["path"].strip()):
                raise ValueError("sealed_test_tasks must declare a path and SHA-256")
            digest = str(self._sealed_test.get("sha256", "")).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sealed_test_tasks must declare a valid SHA-256")
        self.test_objectives_loaded = False

    def tasks(self, split: str) -> list[dict[str, Any]]:
        if split not in SPLITS:
            raise ValueError(f"unknown task split {split!r}")
        if split == "test":
            # This gate runs before checking the test path or reading any bytes.
            self.manifest.require_stage("formal")
            if self._sealed_test is None:
                raise ValueError("no sealed_test_tasks file is declared")
            path = Path(self._sealed_test["path"]).expanduser()
            if not path.is_absolute():
                path = self.path.parent / path
            path = path.resolve()
            _checked_hash(path, self._sealed_test["sha256"])
            raw = _read_json(path)
            if raw.get("version") != VERSION:
                raise ValueError("sealed test task version differs from the traffic protocol")
            node_count = len(self.node_ids) if self.node_ids is not None else None
            test_rows = _task_rows(raw.get("tasks"), "test", node_count)
            _validate_distinct({**self._tasks, "test": test_rows})
            self.test_objectives_loaded = True
            return [{"id": row["id"], "weights": row["weights"].copy()} for row in test_rows]
        return [{"id": row["id"], "weights": row["weights"].copy()} for row in self._tasks[split]]

    def traffic_seeds(self, split: str) -> list[int]:
        if split not in SPLITS:
            raise ValueError(f"unknown traffic seed split {split!r}")
        return list(self._traffic_seeds[split])


def load_protocol(manifest_path: str | Path,
                  node_ids: Sequence[str] | None = None) -> TrafficTaskProtocol:
    """Verify the declared protocol hash and read train/validation tasks only."""
    manifest = load_manifest(manifest_path)
    declaration = manifest.raw.get("traffic_task_protocol")
    if not isinstance(declaration, dict) or declaration.get("version") != VERSION:
        raise ValueError("manifest requires a ma-usfa-traffic-task-v2 declaration")
    if not isinstance(declaration.get("path"), str) or not declaration["path"].strip():
        raise ValueError("traffic_task_protocol.path must be declared")
    path = resolve_manifest_path(manifest, declaration["path"])
    fingerprint = _checked_hash(path, declaration.get("sha256"))
    return TrafficTaskProtocol(manifest, path, _read_json(path), fingerprint, node_ids)
