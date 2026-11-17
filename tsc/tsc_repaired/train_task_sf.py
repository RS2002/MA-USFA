"""Train V2 task-conditioned state SF on executed own-source labels only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_manifest, sha256_file
from .features import FEATURE_NAMES
from .task_sf import (REFERENCE_VERSION, TaskConditionedSF, TaskSFConfig,
                      task_sf_checkpoint, task_sf_td_target)


def _reference_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        if "version" not in data or str(data["version"].item()) != REFERENCE_VERSION:
            raise ValueError("references are not task-conditioned V2; recollect legacy entry-context data")
        names = ("state", "next_state", "weights", "entries", "features",
                 "done", "time_fraction", "next_time_fraction", "source_count",
                 "source_hashes_json", "manifest_sha256", "task_protocol_sha256",
                 "protocol_sha256", "scenario_hashes_json", "feature_names")
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError("reference NPZ missing: " + ", ".join(missing))
        result = {name: data[name].copy() for name in names}
        if "adjacency" in data:
            result["adjacency"] = data["adjacency"].copy()
    state, nxt, weights, entries, features = (
        result[name] for name in ("state", "next_state", "weights", "entries", "features"))
    if state.ndim != 3 or nxt.shape != state.shape or not len(state):
        raise ValueError("reference state/next_state must share nonempty [T,N,D]")
    t, n = state.shape[:2]
    if weights.shape != (t, n, 5) or features.shape != (t, n, 5):
        raise ValueError("traffic reference weights/features must be [T,N,5]")
    if entries.shape != (t, n) or not np.issubdtype(entries.dtype, np.integer):
        raise ValueError("reference entries must be integer [T,N]")
    k = int(result["source_count"].item())
    if k <= 0 or np.any(entries < 0) or np.any(entries >= k):
        raise ValueError("reference source labels outside declared source_count")
    counts = np.bincount(entries.reshape(-1), minlength=k)
    if np.any(counts == 0):
        missing_sources = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"reference data did not observe sources {missing_sources}; recollect with adequate sampling/budget")
    for name in ("done", "time_fraction", "next_time_fraction"):
        if result[name].shape != (t,):
            raise ValueError(f"{name} must have shape [T]")
    for name in ("state", "next_state", "weights", "features", "time_fraction", "next_time_fraction"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"non-finite reference data: {name}")
    if not np.isin(result["done"], [False, True]).all():
        raise ValueError("reference done contains invalid values")
    if "adjacency" in result and result["adjacency"].shape != (n, n):
        raise ValueError("reference adjacency must be [N,N]")
    if result["feature_names"].tolist() != list(FEATURE_NAMES):
        raise ValueError("reference feature names/order differ from the five traffic features")
    return result


def train_task_sf(manifest_path, references, output, seed=0, steps=50,
                  batch_size=32, device="cpu", guard=None, gamma=.99,
                  lr=1e-3, target_every=100, hidden_dim=64, attention_dim=32):
    manifest = load_manifest(manifest_path)
    manifest.require_stage("pilot")
    from .task_protocol_v2 import load_protocol
    protocol = load_protocol(manifest_path)
    if int(steps) <= 0 or int(batch_size) <= 0 or int(target_every) <= 0:
        raise ValueError("SF steps, batch_size, and target_every must be positive")
    if not np.isfinite(lr) or float(lr) <= 0:
        raise ValueError("SF learning rate must be finite and positive")
    out = Path(output).expanduser().resolve()
    checkpoint = out / "task_sf.pt"
    if checkpoint.exists() or (out / "train_task_sf.json").exists():
        raise FileExistsError(f"SF output exists; preserve it or choose another directory: {out}")
    if guard is not None:
        guard.check()
    path = Path(references).expanduser().resolve()
    data = _reference_arrays(path)
    if str(data["manifest_sha256"].item()) != sha256_file(Path(manifest_path)):
        raise ValueError("reference manifest identity differs from SF training manifest")
    protocol_hash = str(manifest.raw.get("traffic_task_protocol", {}).get("sha256", ""))
    if not protocol_hash or str(data["task_protocol_sha256"].item()) != protocol_hash:
        raise ValueError("reference task protocol differs from SF training protocol")
    if str(data["protocol_sha256"].item()) != protocol.fingerprint:
        raise ValueError("reference protocol fingerprint differs from manifest protocol")
    state = data["state"].astype(np.float32, copy=False)
    next_state = data["next_state"].astype(np.float32, copy=False)
    weights = data["weights"].astype(np.float32, copy=False)
    features = data["features"].astype(np.float32, copy=False)
    source_count = int(data["source_count"].item())
    # The task dimensionality is five regardless of how many anchors exist.
    cfg = TaskSFConfig(state_dim=state.shape[-1], source_count=source_count,
                       task_dim=weights.shape[-1], feature_dim=features.shape[-1],
                       hidden_dim=int(hidden_dim), attention_dim=int(attention_dim),
                       gamma=float(gamma))
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    model = TaskConditionedSF(cfg).to(device)
    target = TaskConditionedSF(cfg).to(device)
    target.load_state_dict(model.state_dict())
    target.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    adjacency = data.get("adjacency")
    if adjacency is not None:
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device=device)
    losses = []
    for update in range(int(steps)):
        if guard is not None:
            guard.check()
        idx = rng.integers(0, len(state), size=min(int(batch_size), len(state)))
        s = torch.as_tensor(state[idx], dtype=torch.float32, device=device)
        nxt = torch.as_tensor(next_state[idx], dtype=torch.float32, device=device)
        w = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
        own = torch.as_tensor(data["entries"][idx], dtype=torch.long, device=device)
        time = torch.as_tensor(data["time_fraction"][idx], dtype=torch.float32, device=device)
        next_time = torch.as_tensor(data["next_time_fraction"][idx], dtype=torch.float32, device=device)
        prediction = model(s, own, w, time, adjacency)
        with torch.no_grad():
            bootstrap = target(nxt, own, w, next_time, adjacency)
            y = task_sf_td_target(
                torch.as_tensor(features[idx], dtype=torch.float32, device=device),
                bootstrap, torch.as_tensor(data["done"][idx], device=device), gamma)
        loss = torch.nn.functional.smooth_l1_loss(prediction, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite SF loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError("non-finite SF gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (update + 1) % int(target_every) == 0:
            target.load_state_dict(model.state_dict())
    if guard is not None:
        guard.check()
    out.mkdir(parents=True, exist_ok=True)
    # Save CPU tensors without changing the requested training device until
    # all training operations have finished.
    source_hashes = json.loads(str(data["source_hashes_json"].item()))
    payload = task_sf_checkpoint(
        model.cpu().eval(), seed=int(seed), steps=int(steps),
        references_sha256=sha256_file(path), source_hashes=source_hashes,
        manifest_sha256=sha256_file(Path(manifest_path)),
        task_protocol_sha256=protocol_hash, protocol_sha256=protocol.fingerprint,
        scenario_hashes=json.loads(str(data["scenario_hashes_json"].item())),
        feature_names=data["feature_names"].tolist(), optimizer_updates=int(steps),
        test_updates=0, adjacency=data.get("adjacency"))
    temporary = checkpoint.with_suffix(".pt.partial")
    try:
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "status": "PASS", "version": payload["version"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "reference_sha256": sha256_file(path), "source_hashes": source_hashes,
        "protocol_sha256": protocol.fingerprint,
        "scenario_hashes": payload["scenario_hashes"],
        "feature_names": payload["feature_names"],
        "seed": int(seed), "steps": int(steps), "optimizer_updates": int(steps),
        "source_count": source_count, "task_dim": cfg.task_dim,
        "source_visit_counts": np.bincount(data["entries"].reshape(-1), minlength=source_count).tolist(),
        "mean_recent_loss": float(np.mean(losses[-min(100, len(losses)):])),
        "reference_semantics": payload["reference_semantics"],
        "actual_teammate_entries_are_inputs": False,
        "test_accessed": False, "test_updates": 0,
    }
    (out / "train_task_sf.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        from .resources import ResourceGuard, configure_torch_threads
        manifest = load_manifest(args.manifest)
        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        configure_torch_threads(manifest.resource_limits["threads"], device)
        guard = ResourceGuard(manifest, args.output, device=device)
        report = train_task_sf(args.manifest, args.references, args.output,
                               args.seed, args.steps, args.batch_size, device,
                               guard=guard)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, FloatingPointError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc),
                          "test_accessed": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
