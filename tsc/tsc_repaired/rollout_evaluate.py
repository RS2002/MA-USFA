"""Frozen, paired traffic-policy evaluation by executing actions in SUMO.

Unlike ``evaluate.py``, this module does not average stored behavior rewards.
Every reported return is accumulated from ``runtime.step(selected_entries)``.
The task protocol provides all objectives and traffic seeds before evaluation;
no policy, normalizer, or checkpoint is fitted or selected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .config import load_manifest, sha256_file
from .task_protocol_v2 import load_protocol
from .task_selector import load_selector
from .traffic_runtime import FrozenTrafficRuntime


EVALUATION_VERSION = "ma-usfa-traffic-rollout-evaluation-v2"
SELECTOR_VERSION = "ma-usfa-traffic-attention-ddqn-v2"


def _tensor_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    # Include non-persistent graph buffers as well as checkpoint state.
    tensors = {f"state:{name}": value for name, value in model.state_dict().items()}
    tensors.update({f"buffer:{name}": value for name, value in model.named_buffers()})
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _weights_fingerprint(weights: np.ndarray) -> str:
    value = np.ascontiguousarray(weights, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _system_metric(info: Any, name: str) -> float | None:
    """Read a global metric once, even if parallel_env repeats it per agent."""
    if not isinstance(info, dict):
        return None
    if name in info:
        return _optional_float(info[name])
    values = [_optional_float(item[name]) for item in info.values()
              if isinstance(item, dict) and name in item]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if not np.allclose(values, values[0], rtol=1e-7, atol=1e-9):
        # These are not an unambiguous repeated global statistic.
        return None
    return values[0]


def _complete_time_mean(values: list[float | None]) -> float | None:
    # Do not silently average just the steps for which a metric exists.
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(values))


def parse_tripinfo(path: str | Path) -> dict:
    """Completed-trip statistics, parsed only after the SUMO episode closes.

    Missing or invalid files yield unavailable values rather than zeros.
    If unfinished trips are emitted by SUMO, they are counted separately and
    excluded from the completed-trip means. These means alone do not measure
    network-wide performance under censored/unfinished demand.
    """
    path = Path(path)
    empty = {
        "status": "unavailable",
        "completed_trip_count": None,
        "unfinished_trip_count": None,
        "mean_completed_duration_seconds": None,
        "mean_completed_waiting_time_seconds": None,
        "mean_completed_time_loss_seconds": None,
    }
    if not path.is_file():
        return {**empty, "reason": "tripinfo file was not produced"}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {**empty, "status": "unreadable", "reason": str(exc)}
    complete, unfinished = [], 0
    for item in root.findall("tripinfo"):
        arrival = _optional_float(item.get("arrival"))
        if arrival is not None and arrival < 0:
            unfinished += 1
        else:
            complete.append(item)
    result = {
        **empty,
        "status": "available" if complete else "no_completed_trips",
        "completed_trip_count": len(complete),
        "unfinished_trip_count": unfinished,
    }
    for attribute, field in (
        ("duration", "mean_completed_duration_seconds"),
        ("waitingTime", "mean_completed_waiting_time_seconds"),
        ("timeLoss", "mean_completed_time_loss_seconds"),
    ):
        values = [_optional_float(item.get(attribute)) for item in complete]
        result[field] = _complete_time_mean(values)
    return result


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)


def _check_selector_identity(payload: dict, runtime, protocol, sf_checkpoint: Path) -> float:
    if payload.get("version") != SELECTOR_VERSION:
        raise ValueError("closed-loop evaluation requires a V2 attention-DDQN checkpoint")
    metadata = payload.get("metadata")
    config = payload.get("config")
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise ValueError("selector checkpoint is missing V2 identity/config metadata")
    checks = {
        "source_hashes": runtime.source_hashes,
        "sf_sha256": sha256_file(sf_checkpoint),
        "protocol_fingerprint": protocol.fingerprint,
        "node_ids": list(runtime.ids),
        "training_split": "train",
    }
    for name, expected in checks.items():
        if metadata.get(name) != expected:
            raise ValueError(f"selector checkpoint {name} differs from this evaluation")
    dimensions = {
        "observation_dim": runtime.observation_dim,
        "source_count": runtime.K,
        "action_count": runtime.action_count,
        "task_dim": int(runtime.weights.shape[-1]),
        "horizon": runtime.horizon,
    }
    for name, expected in dimensions.items():
        if config.get(name) != expected:
            raise ValueError(f"selector config {name} differs from runtime")
    saved_adjacency = payload.get("adjacency")
    if saved_adjacency is None:
        raise ValueError("selector checkpoint does not bind its attention graph")
    if not np.array_equal(_numpy(saved_adjacency).astype(bool),
                          _numpy(runtime.adjacency).astype(bool)):
        raise ValueError("selector attention adjacency differs from runtime")
    gamma = float(config.get("gamma", float("nan")))
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("selector gamma must lie in [0,1]")
    sf_config = getattr(runtime, "sf_payload", {}).get("config", {})
    if "gamma" in sf_config and float(sf_config["gamma"]) != gamma:
        raise ValueError("SF and upper selector discount factors differ")
    return gamma


def _choose_entries(method: str, model, runtime, state: np.ndarray,
                    candidates: np.ndarray, values: np.ndarray, step: int,
                    device: str) -> np.ndarray:
    if method == "independent_entry":
        return values.argmax(-1).astype(np.int64)
    if method == "sync_entry":
        return np.full(runtime.N, int(values.mean(0).argmax()), dtype=np.int64)
    if method.startswith("fixed_source_"):
        index = int(method.removeprefix("fixed_source_"))
        return np.full(runtime.N, index, dtype=np.int64)
    if method != "ma_usfa_ddqn":
        raise ValueError(f"unknown traffic evaluation method {method!r}")
    obs = torch.as_tensor(state[None], dtype=torch.float32, device=device)
    weights = torch.as_tensor(runtime.weights[None], dtype=torch.float32, device=device)
    actions = torch.as_tensor(candidates[None], dtype=torch.long, device=device)
    scores = torch.as_tensor(values[None], dtype=torch.float32, device=device)
    fraction = torch.tensor([step / max(1, runtime.horizon)], dtype=torch.float32, device=device)
    return model.choose_entries(obs, weights, actions, scores, fraction)[0].cpu().numpy().astype(np.int64)


def _run_episode(runtime, model, method: str, task: dict, traffic_seed: int,
                 tripinfo_path: Path, gamma: float, device: str, guard) -> dict:
    if guard is not None:
        guard.check()
    state = runtime.reset(task, traffic_seed, tripinfo_path=tripinfo_path)
    team_return, discounted_return = 0.0, 0.0
    histogram = np.zeros(runtime.K, dtype=np.int64)
    metrics = {key: [] for key in ("system_total_stopped", "system_mean_waiting_time", "system_mean_speed")}
    steps = 0
    done = False
    try:
        with torch.inference_mode():
            for step in range(runtime.horizon):
                if guard is not None:
                    guard.check()
                candidates = runtime.candidate_actions(state, step)
                values = runtime.sf_values(state, step)
                if candidates.shape != (runtime.N, runtime.K) or values.shape != candidates.shape:
                    raise ValueError("candidate actions/SF values must have shape [N,K]")
                if not np.isfinite(values).all():
                    raise ValueError("non-finite candidate SF values during rollout")
                entries = _choose_entries(method, model, runtime, state, candidates, values, step, device)
                if entries.shape != (runtime.N,) or np.any(entries < 0) or np.any(entries >= runtime.K):
                    raise ValueError("rollout selector produced invalid source entries")
                transition = runtime.step(entries)
                actual_actions = np.asarray(transition.actions, dtype=np.int64)
                expected_actions = candidates[np.arange(runtime.N), entries]
                if not np.array_equal(actual_actions, expected_actions):
                    raise RuntimeError("executed primitive actions differ from the selected frozen source entries")
                rewards = np.asarray(transition.rewards, dtype=np.float64)
                if rewards.shape != (runtime.N,) or not np.isfinite(rewards).all():
                    raise ValueError("runtime rewards must be finite with one value per node")
                reward_sum = float(rewards.sum())
                team_return += reward_sum
                discounted_return += gamma ** step * reward_sum
                histogram += np.bincount(entries, minlength=runtime.K)
                steps += 1
                for key in metrics:
                    metrics[key].append(_system_metric(transition.info, key))
                state, done = transition.state, bool(transition.done)
                if done:
                    break
        if not done:
            raise RuntimeError("runtime did not terminate or truncate at its declared horizon")
    finally:
        # Each episode has its own tripinfo file, flushed before parsing.
        runtime.close()
    return {
        "status": "PASS",
        "method": method,
        "task_id": str(task["id"]),
        "task_sha256": _weights_fingerprint(runtime.weights),
        "traffic_seed": int(traffic_seed),
        "node_count": runtime.N,
        "steps": steps,
        "terminated_or_truncated": done,
        "episode_team_return": team_return,
        "episode_mean_agent_return": team_return / runtime.N,
        "mean_step_reward": team_return / (runtime.N * steps),
        "discounted_team_return": discounted_return,
        "gamma": gamma,
        "mean_queue_per_node": (
            None if (queue := _complete_time_mean(metrics["system_total_stopped"])) is None
            else queue / runtime.N),
        "mean_waiting_time_seconds": _complete_time_mean(metrics["system_mean_waiting_time"]),
        "mean_speed_m_per_second": _complete_time_mean(metrics["system_mean_speed"]),
        "physical_metric_available_steps": {
            key: sum(value is not None for value in values) for key, values in metrics.items()
        },
        "source_selection_histogram": {str(index): int(count) for index, count in enumerate(histogram)},
        "tripinfo_file": tripinfo_path.name,
        "tripinfo": parse_tripinfo(tripinfo_path),
        "evaluation_updates": 0,
        "test_updates": 0,
    }


def _summary(rows: list[dict], methods: list[str]) -> dict:
    result = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        team_returns = [row["episode_team_return"] for row in selected]
        result[method] = {
            "episodes": len(selected),
            "mean_episode_team_return": float(np.mean(team_returns)),
            "sample_std_episode_team_return": (
                float(np.std(team_returns, ddof=1)) if len(team_returns) > 1 else None),
            "mean_episode_mean_agent_return": float(np.mean([row["episode_mean_agent_return"] for row in selected])),
            "mean_step_reward": float(np.mean([row["mean_step_reward"] for row in selected])),
        }
    return result


def evaluate_rollouts(manifest_path, source_dir, sf_checkpoint, checkpoint, output,
                      split="validation", episodes_per_task=1, seed=0,
                      device="cpu", methods=None, guard=None) -> dict:
    """Execute paired frozen policies and preserve every task/seed result.

    ``episodes_per_task`` chooses the first declared traffic seeds, without
    inventing new seeds or adapting objectives. A failure suppresses aggregate
    scores and is recorded with all already completed and failed rows.
    """
    report_path = Path(output).expanduser().resolve()
    if report_path.suffix.lower() != ".json":
        report_path = report_path / f"evaluation_{split}.json"
    artifacts = report_path.parent / f"{report_path.stem}_episodes"
    report = {
        "version": EVALUATION_VERSION,
        "status": "NOT_READY",
        "evaluation_kind": "closed_loop_sumo_rollout",
        "policy_performance_evaluated": False,
        "split": split,
        "evaluation_seed": int(seed),
        "evaluation_updates": 0,
        "test_updates": 0,
        "episodes": [],
        "summary": None,
        "errors": [],
    }
    runtime, model = None, None
    model_before = None
    artifact_hashes = {}
    can_write = not report_path.exists()
    try:
        if not can_write:
            raise FileExistsError("evaluation report already exists; preserve it and choose a new output path")
        if artifacts.exists() and any(artifacts.iterdir()):
            raise FileExistsError("episode evidence directory already exists; choose a new output path")
        if split not in {"validation", "test"}:
            raise ValueError("policy evaluation accepts validation or explicitly gated test only")
        manifest = load_manifest(manifest_path)
        manifest.require_stage("formal" if split == "test" else "pilot")
        if isinstance(episodes_per_task, bool) or int(episodes_per_task) != episodes_per_task or episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be a positive integer")
        sf_path = Path(sf_checkpoint).expanduser().resolve()
        source_root = Path(source_dir).expanduser().resolve()
        model_path = Path(checkpoint).expanduser().resolve()
        files = [model_path, sf_path, *sorted(source_root.glob("source_*.pt"))]
        if len(files) < 3:
            raise ValueError("source directory contains no source checkpoints")
        artifact_hashes = {str(path): sha256_file(path) for path in files}
        runtime = FrozenTrafficRuntime(manifest_path, source_dir, sf_checkpoint, device=device)
        protocol = load_protocol(manifest_path, runtime.ids)
        tasks = protocol.tasks(split)
        traffic_seeds = protocol.traffic_seeds(split)
        if not tasks:
            raise ValueError("no evaluation tasks were declared")
        if len(traffic_seeds) < int(episodes_per_task):
            raise ValueError("episodes_per_task exceeds the declared independent traffic seeds")
        traffic_seeds = traffic_seeds[:int(episodes_per_task)]
        default_methods = ["ma_usfa_ddqn", "sync_entry", "independent_entry"]
        default_methods += [f"fixed_source_{index}" for index in range(runtime.K)]
        chosen_methods = default_methods if methods is None else list(methods)
        if not chosen_methods or len(set(chosen_methods)) != len(chosen_methods):
            raise ValueError("methods must be a nonempty list without duplicates")
        if any(method not in default_methods for method in chosen_methods):
            raise ValueError("unsupported method in paired traffic evaluation")
        model, payload = load_selector(model_path, device=device)
        _freeze(model)
        for frozen_model in runtime.library.models:
            _freeze(frozen_model)
        if getattr(runtime, "sf", None) is not None:
            _freeze(runtime.sf)
        gamma = _check_selector_identity(payload, runtime, protocol, sf_path)
        source_bound = bool(getattr(runtime, "source_scenario_binding_verified", False))
        if split == "test" and not source_bound:
            raise ValueError("formal test requires source checkpoints bound to the declared scenario")
        model_before = _tensor_fingerprint(model)
        report.update({
            "protocol_fingerprint": protocol.fingerprint,
            "node_ids": list(runtime.ids),
            "methods": chosen_methods,
            "traffic_seeds": traffic_seeds,
            "task_ids": [str(task["id"]) for task in tasks],
            "expected_episode_count": len(tasks) * len(traffic_seeds) * len(chosen_methods),
            "checkpoint_sha256": artifact_hashes[str(model_path)],
            "sf_sha256": artifact_hashes[str(sf_path)],
            "source_hashes": runtime.source_hashes,
            "source_scenario_binding_verified": source_bound,
            "development_only": split != "test" or not source_bound,
            "episode_evidence_directory": artifacts.name,
            "gamma": gamma,
            "horizon_steps": runtime.horizon,
            "return_definition": "episode_team_return = sum_t sum_i reward_i(t); un-discounted",
            "baseline_definitions": {
                "sync_entry": "At each state choose one source maximizing the mean conditional-SF score; all nodes use that source.",
                "independent_entry": "Each node chooses its highest conditional-SF source-action score, with first-index ties.",
                "fixed_source_k": "Every node executes source k throughout the episode; every k is reported separately.",
                "ma_usfa_ddqn": "Frozen attention DDQN chooses source entries; only their frozen primitive actions execute.",
            },
            "scientific_scope": (
                "paired validation development rollouts; no claim of multi-training-seed or formal-test evidence"
                if split != "test" else "explicitly gated paired test rollouts; no claim of multi-training-seed evidence"),
        })
        artifacts.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            for traffic_seed in traffic_seeds:
                for method in chosen_methods:
                    ordinal = len(report["episodes"])
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task["id"]))[:60]
                    path = artifacts / f"{ordinal:05d}_{safe_id}_{method}_seed{traffic_seed}.xml"
                    try:
                        row = _run_episode(runtime, model, method, task, int(traffic_seed),
                                           path, gamma, device, guard)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        report["episodes"].append({
                            "status": "NOT_READY", "method": method,
                            "task_id": str(task["id"]), "traffic_seed": int(traffic_seed),
                            "error": str(exc), "evaluation_updates": 0, "test_updates": 0,
                        })
                        raise
                    report["episodes"].append(row)
        runtime.close()
        if hasattr(runtime, "audit_frozen"):
            frozen_audit = runtime.audit_frozen()
            if isinstance(frozen_audit, dict) and frozen_audit.get("status", "PASS") != "PASS":
                raise RuntimeError("runtime frozen-source/SF audit failed")
            report["runtime_frozen_audit"] = frozen_audit
        if _tensor_fingerprint(model) != model_before:
            raise RuntimeError("selector parameters or buffers changed during evaluation")
        after = {path: sha256_file(Path(path)) for path in artifact_hashes}
        if artifact_hashes != after:
            raise RuntimeError("a source/SF/selector checkpoint file changed during evaluation")
        report.update({
            "status": "PASS",
            "policy_performance_evaluated": True,
            "completed_episode_count": len(report["episodes"]),
            "checkpoint_files_unchanged": True,
            "selector_parameters_unchanged": True,
            "summary": _summary(report["episodes"], chosen_methods),
        })
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
        report["completed_episode_count"] = sum(row["status"] == "PASS" for row in report["episodes"])
        report["summary"] = None
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                report["status"] = "NOT_READY"
                report["policy_performance_evaluated"] = False
                report["errors"].append(f"environment close failed: {exc}")
                report["summary"] = None
    if can_write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--sf-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()
    try:
        from .resources import ResourceGuard, configure_torch_threads
        manifest = load_manifest(args.manifest)
        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        configure_torch_threads(
            int(manifest.resource_limits.get("threads", 2)), device,
            float(manifest.resource_limits.get("gpu_memory_fraction", .75)))
        output = Path(args.output)
        guard_root = output.parent if output.suffix.lower() == ".json" else output
        guard = ResourceGuard(manifest, guard_root / "logs", device=device)
        report = evaluate_rollouts(
            args.manifest, args.source_dir, args.sf_checkpoint, args.checkpoint,
            args.output, split=args.split, episodes_per_task=args.episodes_per_task,
            seed=args.seed, device=device, methods=args.methods, guard=guard)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        report = {"status": "NOT_READY", "evaluation_kind": "closed_loop_sumo_rollout",
                  "policy_performance_evaluated": False, "evaluation_updates": 0,
                  "test_updates": 0, "errors": [str(exc)]}
    # Full episode evidence is in the file; the terminal stays compact.
    compact = {name: report.get(name) for name in (
        "status", "evaluation_kind", "policy_performance_evaluated", "split",
        "expected_episode_count", "completed_episode_count", "evaluation_updates",
        "test_updates", "summary", "errors")}
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
