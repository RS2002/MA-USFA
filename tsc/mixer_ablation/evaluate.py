"""Frozen, paired validation for the two explicitly tagged mixer variants.

Only the node-weight normalization differs between the two training arms.
This evaluator never fits/selects a checkpoint or accesses test objectives.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import torch

from tsc_repaired.config import load_manifest, sha256_file
from tsc_repaired.rollout_evaluate import (
    SELECTOR_VERSION, _check_selector_identity, _freeze, _run_episode,
    _summary, _tensor_fingerprint,
)
from tsc_repaired.task_protocol_v2 import load_protocol
from tsc_repaired.traffic_runtime import FrozenTrafficRuntime
from .selector import load_ablation_selector

EVALUATION_VERSION = "ma-usfa-traffic-paired-mixer-evaluation-v1"
PAIR_FIELDS = ("initial_parameter_sha256", "training_configuration", "seed", "episodes",
               "transitions", "optimizer_updates", "checkpoint_selection",
               "training_reward_reduction", "source_hashes", "sf_sha256",
               "protocol_fingerprint", "node_ids", "training_split")
METRICS = ("episode_team_return", "discounted_team_return", "mean_queue_per_node",
           "mean_waiting_time_seconds", "mean_speed_m_per_second", "completed_trip_count")


def _check_pair(control, normalized):
    """Reject comparisons that silently differ in more than the mixer."""
    for variant, payload in (("control", control), ("normalized", normalized)):
        metadata = payload.get("metadata", {})
        if metadata.get("ablation_variant") != variant:
            raise ValueError("checkpoint ablation_variant differs from its assigned arm")
        if metadata.get("test_accessed") is not False or metadata.get("test_updates") != 0:
            raise ValueError("mixer arms must be train-only with zero test updates")
        digest = metadata.get("initial_parameter_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("arm is missing a valid initial_parameter_sha256")
        if not isinstance(metadata.get("training_configuration"), dict) or not metadata["training_configuration"]:
            raise ValueError("arm is missing its declared training_configuration")
        for name in ("episodes", "transitions", "optimizer_updates"):
            if isinstance(metadata.get(name), bool) or not isinstance(metadata.get(name), int) or metadata[name] <= 0:
                raise ValueError("arm has no positive training " + name)
    for name in PAIR_FIELDS:
        if name not in control["metadata"] or control["metadata"][name] != normalized["metadata"].get(name):
            raise ValueError("mixer arm training mismatch: " + name)
    if control.get("config") != normalized.get("config"):
        raise ValueError("mixer arm selector configurations differ")


def _check_runtime_identity(payload, runtime, protocol, sf_path):
    # load_ablation_selector has already checked the actual variant/version and
    # selected the correct class. Reuse the legacy helper only for its shared
    # dimensions, source/SF/protocol, graph and discount checks; its old version
    # gate does not know the new normalized checkpoint identifier.
    identity_view = {**payload, "version": SELECTOR_VERSION}
    return _check_selector_identity(identity_view, runtime, protocol, sf_path)


def _metric(row, name):
    if name == "completed_trip_count":
        return row.get("tripinfo", {}).get(name)
    return row.get(name)


def _paired_differences(rows):
    result = []
    pairs = {}
    for row in rows:
        if row["method"] in {"control_ddqn", "normalized_ddqn"}:
            pairs.setdefault((row["task_id"], row["traffic_seed"]), {})[row["method"]] = row
    for (task, seed), pair in pairs.items():
        if set(pair) != {"control_ddqn", "normalized_ddqn"}:
            raise ValueError("incomplete task/traffic pair")
        control, normalized = pair["control_ddqn"], pair["normalized_ddqn"]
        for name in ("task_sha256", "node_count", "steps", "gamma"):
            if control[name] != normalized[name]:
                raise ValueError("mixer evaluation pair differs in " + name)
        metrics = {}
        for name in METRICS:
            a, b = _metric(control, name), _metric(normalized, name)
            metrics[name] = {
                "control": a, "normalized": b,
                "normalized_minus_control": None if a is None or b is None else float(b - a),
                "relative_percent": None if a is None or b is None or a == 0 else float(100 * (b - a) / abs(a)),
            }
        result.append({"task_id": task, "traffic_seed": seed, "metrics": metrics})
    return result


def _extended_summary(rows, methods):
    summary = _summary(rows, methods)
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for name in METRICS[1:]:
            values = [_metric(row, name) for row in selected]
            summary[method]["mean_" + name] = (
                None if any(value is None for value in values) else float(np.mean(values)))
    return summary


def evaluate_ablation(manifest_path, source_dir, sf_checkpoint, control_checkpoint,
                      normalized_checkpoint, output, episodes_per_task=1,
                      device="cpu", guard=None, all_fixed_sources=False):
    """Execute all methods on every declared validation task/selected seed.

    Any failed rollout or changed artifact invalidates the whole aggregate.
    Existing reports and episode evidence are never overwritten.
    """
    report_path = Path(output).expanduser().resolve()
    if report_path.suffix.lower() != ".json":
        report_path = report_path / "evaluation_validation.json"
    artifacts = report_path.parent / (report_path.stem + "_episodes")
    report = {"version": EVALUATION_VERSION, "status": "NOT_READY",
              "evaluation_kind": "closed_loop_sumo_rollout", "split": "validation",
              "policy_performance_evaluated": False, "evaluation_updates": 0,
              "test_updates": 0, "test_accessed": False, "ready_for_formal": False,
              "episodes": [], "summary": None, "paired_differences": None, "errors": []}
    can_write = not report_path.exists()
    runtime, models = None, {}
    fingerprints, input_hashes = {}, {}
    try:
        if not can_write:
            raise FileExistsError("evaluation report already exists; preserve it and choose a new output")
        if artifacts.exists() and any(artifacts.iterdir()):
            raise FileExistsError("episode evidence already exists; preserve it and choose a new output")
        if isinstance(episodes_per_task, bool) or int(episodes_per_task) != episodes_per_task or episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be a positive integer")
        if not isinstance(all_fixed_sources, bool):
            raise ValueError("all_fixed_sources must be boolean")
        manifest = load_manifest(manifest_path)
        manifest.require_stage("pilot")
        sf_path = Path(sf_checkpoint).expanduser().resolve()
        source_root = Path(source_dir).expanduser().resolve()
        checkpoint_paths = {"control": Path(control_checkpoint).expanduser().resolve(),
                            "normalized": Path(normalized_checkpoint).expanduser().resolve()}
        if checkpoint_paths["control"] == checkpoint_paths["normalized"]:
            raise ValueError("mixer comparison requires two distinct checkpoint files")
        source_files = sorted(source_root.glob("source_*.pt"))
        if not source_files:
            raise ValueError("no frozen source checkpoints were found")
        files = [Path(manifest_path).expanduser().resolve(), sf_path, *checkpoint_paths.values(), *source_files]
        input_hashes = {str(path): sha256_file(path) for path in files}
        runtime = FrozenTrafficRuntime(manifest_path, source_root, sf_path, device=device)
        scenario_files = getattr(getattr(runtime, "spec", None), "files", None)
        if scenario_files is not None:
            for path in (scenario_files.network, scenario_files.routes):
                path = Path(path)
                files.append(path)
                input_hashes[str(path)] = sha256_file(path)
        protocol = load_protocol(manifest_path, runtime.ids)
        protocol_path = getattr(protocol, "path", None)
        if protocol_path is not None:
            files.append(Path(protocol_path))
            input_hashes[str(Path(protocol_path))] = sha256_file(Path(protocol_path))
        tasks, traffic = protocol.tasks("validation"), protocol.traffic_seeds("validation")
        if not tasks or len(traffic) < int(episodes_per_task):
            raise ValueError("not enough declared validation tasks or traffic seeds")
        traffic = traffic[:int(episodes_per_task)]
        payloads = {}
        for variant, path in checkpoint_paths.items():
            models[variant], payloads[variant] = load_ablation_selector(path, device=device)
            _freeze(models[variant])
        _check_pair(payloads["control"], payloads["normalized"])
        gamma = None
        for variant, payload in payloads.items():
            current = _check_runtime_identity(payload, runtime, protocol, sf_path)
            if gamma is not None and current != gamma:
                raise ValueError("mixer arms have different discount factors")
            gamma = current
            fingerprints[variant] = _tensor_fingerprint(models[variant])
        for model in runtime.library.models:
            _freeze(model)
        if getattr(runtime, "sf", None) is not None:
            _freeze(runtime.sf)
        methods = ["control_ddqn", "normalized_ddqn", "independent_entry", "sync_entry"]
        methods += [f"fixed_source_{i}" for i in range(runtime.K)] if all_fixed_sources else ["fixed_source_0"]
        report.update({
            "methods": methods, "task_ids": [str(task["id"]) for task in tasks],
            "traffic_seeds": list(traffic), "expected_episode_count": len(tasks) * len(traffic) * len(methods),
            "checkpoint_sha256": {variant: input_hashes[str(path)] for variant, path in checkpoint_paths.items()},
            "sf_sha256": input_hashes[str(sf_path)], "source_hashes": runtime.source_hashes,
            "protocol_fingerprint": protocol.fingerprint, "node_ids": list(runtime.ids),
            "gamma": gamma, "horizon_steps": runtime.horizon,
            "source_scenario_binding_verified": bool(getattr(runtime, "source_scenario_binding_verified", False)),
            "episode_evidence_directory": artifacts.name,
            "controlled_training_configuration": {name: payloads["control"]["metadata"][name] for name in PAIR_FIELDS},
            "return_definition": "episode_team_return = sum_t sum_i reward_i(t); undiscounted",
            "comparison_definition": "normalized_minus_control on the same validation task and SUMO seed",
            "scientific_scope": "single-training-seed development A/B; no formal test or significance claim",
            "completed_trip_metric_scope": "Completed trips only; short-window censoring can differ by policy; zero unfinished records does not imply all demand finished.",
            "baseline_definitions": {
                "independent_entry": "Per-node greedy under the same frozen task-conditioned SF.",
                "sync_entry": "One common source maximizing mean task-conditioned SF score.",
                "fixed_source_k": "Every node executes frozen source k; default compares k=0 only.",
            },
        })
        artifacts.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            for seed in traffic:
                for method in methods:
                    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task["id"]))[:60]
                    path = artifacts / f"{len(report['episodes']):05d}_{safe_task}_{method}_seed{seed}.xml"
                    variant = "normalized" if method == "normalized_ddqn" else "control"
                    original_method = "ma_usfa_ddqn" if method in {"control_ddqn", "normalized_ddqn"} else method
                    try:
                        row = _run_episode(runtime, models[variant], original_method, task,
                                           int(seed), path, gamma, device, guard)
                        row["method"] = method
                        report["episodes"].append(row)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        report["episodes"].append({"status": "NOT_READY", "method": method,
                            "task_id": str(task["id"]), "traffic_seed": int(seed), "error": str(exc),
                            "evaluation_updates": 0, "test_updates": 0})
                        raise
        runtime.close()
        audit = runtime.audit_frozen()
        if isinstance(audit, dict) and audit.get("status", "PASS") != "PASS":
            raise RuntimeError("runtime frozen-model audit failed")
        if any(_tensor_fingerprint(models[v]) != before for v, before in fingerprints.items()):
            raise RuntimeError("a mixer selector's parameters or buffers changed")
        if {str(path): sha256_file(path) for path in files} != input_hashes:
            raise RuntimeError("an input or checkpoint file changed during evaluation")
        evidence = {}
        unavailable = []
        for row in report["episodes"]:
            path = artifacts / row["tripinfo_file"]
            if path.is_file():
                evidence[row["tripinfo_file"]] = sha256_file(path)
            else:
                unavailable.append(row["tripinfo_file"])
        report.update({"status": "PASS", "policy_performance_evaluated": True,
            "completed_episode_count": len(report["episodes"]),
            "checkpoint_files_unchanged": True, "selector_parameters_unchanged": True,
            "runtime_frozen_audit": audit, "input_sha256": input_hashes,
            "episode_evidence_sha256": evidence, "unavailable_episode_evidence": unavailable,
            "summary": _extended_summary(report["episodes"], methods),
            "paired_differences": _paired_differences(report["episodes"])})
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "NOT_READY"
        report["policy_performance_evaluated"] = False
        report["errors"].append(str(exc))
        report["completed_episode_count"] = sum(row["status"] == "PASS" for row in report["episodes"])
        report["summary"] = report["paired_differences"] = None
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                report["status"] = "NOT_READY"
                report["policy_performance_evaluated"] = False
                report["summary"] = report["paired_differences"] = None
                report["errors"].append("environment close failed: " + str(exc))
    if can_write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report
