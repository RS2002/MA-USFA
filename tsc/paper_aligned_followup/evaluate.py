"""Frozen evaluation on the recovered legacy Homo/Hetero traffic protocol.

This module deliberately evaluates the repaired strict source-entry selector
under a paper-aligned task protocol.  It does not reinterpret the historical
primitive-action composer numbers as results for the repaired method.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re

from mixer_ablation.evaluate import (
    METRICS, _check_runtime_identity, _extended_summary, _metric,
)
from mixer_ablation.selector import load_ablation_selector
from tsc_repaired.config import load_manifest, sha256_file
from tsc_repaired.rollout_evaluate import _freeze, _run_episode, _tensor_fingerprint
from tsc_repaired.task_protocol_v2 import load_protocol
from tsc_repaired.traffic_runtime import FrozenTrafficRuntime

EVALUATION_VERSION = "ma-usfa-paper-aligned-traffic-evaluation-v1"
LEGACY_EVAL_SEED = 10_000_000


def _load_paper_selector(path, device):
    """Load either a tagged mixer checkpoint or a standard V2 upper checkpoint.

    The latter is useful for a paper-aligned control trained with the original
    repaired attention-DDQN trainer.  Both variants still pass the same strict
    source/SF/protocol identity checks below.
    """
    try:
        return load_ablation_selector(path, device=device)
    except ValueError as ablation_error:
        from tsc_repaired.task_selector import load_selector
        try:
            return load_selector(path, device=device)
        except ValueError:
            raise ablation_error


def _training_budget(payload):
    metadata = payload.get("metadata", {})
    if metadata.get("test_accessed") is not False or metadata.get("test_updates") != 0:
        raise ValueError("selector must declare train-only fitting with zero test updates")
    for key in ("episodes", "transitions", "optimizer_updates"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("selector is missing positive training " + key)
    settings = metadata.get("training_configuration")
    if not isinstance(settings, dict) or not settings:
        # The original repaired trainer predates the mixer metadata field.  It
        # remains admissible when its positive budget counters are present.
        settings = {"configuration_source": "standard_v2_trainer_metadata"}
    return {
        "ablation_variant": metadata.get("ablation_variant", "standard_v2"),
        "initial_parameter_sha256": metadata.get("initial_parameter_sha256"),
        "seed": metadata.get("seed"), "episodes": metadata.get("episodes"),
        "transitions": metadata.get("transitions"),
        "optimizer_updates": metadata.get("optimizer_updates"),
        "training_configuration": settings,
        "checkpoint_selection": metadata.get("checkpoint_selection"),
        "training_split": metadata.get("training_split"),
    }


def _paired_differences(rows, checkpoint_methods, methods):
    grouped = {}
    for row in rows:
        key = (row["task_id"], row["traffic_seed"], row["replay_index"])
        pair = grouped.setdefault(key, {})
        if row["method"] in pair:
            raise ValueError("duplicate method in legacy task/seed/replay pair")
        pair[row["method"]] = row
    result = []
    for (task, seed, replay), pair in grouped.items():
        if set(pair) != set(methods):
            raise ValueError("incomplete legacy evaluation matrix")
        anchor = pair[methods[0]]
        for other in pair.values():
            for name in ("task_sha256", "node_count", "steps", "gamma"):
                if anchor[name] != other[name]:
                    raise ValueError("legacy evaluation pair differs in " + name)
        for method in checkpoint_methods:
            for reference in methods:
                if method == reference:
                    continue
                metrics = {}
                for name in METRICS:
                    value, base = _metric(pair[method], name), _metric(pair[reference], name)
                    metrics[name] = {
                        "candidate": value, "reference": base,
                        "candidate_minus_reference": None if value is None or base is None
                        else float(value - base),
                        "relative_percent": (None if value is None or base is None or base == 0
                                             else float(100 * (value - base) / abs(base))),
                    }
                result.append({"task_id": task, "traffic_seed": seed,
                               "replay_index": replay, "candidate_method": method,
                               "reference_method": reference, "metrics": metrics})
    return result


def evaluate_paper_aligned(manifest_path, source_dir, sf_checkpoint, checkpoints,
                           output, replays=5, device="cpu", guard=None,
                           include_baselines=True, traffic_seed=LEGACY_EVAL_SEED):
    """Evaluate repaired checkpoints on legacy Homo/Hetero tasks.

    ``replays`` repeats the historical deterministic traffic seed.  They are
    paired replays, not independent traffic draws; the report says so
    explicitly.  Any failed rollout suppresses aggregate scores.
    """
    report_path = Path(output).expanduser().resolve()
    if report_path.suffix.lower() != ".json":
        report_path = report_path / "evaluation_paper_aligned.json"
    artifacts = report_path.parent / (report_path.stem + "_episodes")
    report = {
        "version": EVALUATION_VERSION, "status": "NOT_READY",
        "evaluation_kind": "closed_loop_sumo_rollout", "split": "validation",
        "paper_alignment": "legacy_homo_hetero_task_seed_11",
        "policy_performance_evaluated": False, "evaluation_updates": 0,
        "test_updates": 0, "test_accessed": False, "ready_for_formal": False,
        "episodes": [], "summary": None, "paired_differences": None, "errors": [],
    }
    can_write = not report_path.exists()
    runtime, models = None, {}
    fingerprints, input_hashes = {}, {}
    try:
        if not can_write:
            raise FileExistsError("evaluation report exists; preserve it and choose another output")
        if artifacts.exists() and any(artifacts.iterdir()):
            raise FileExistsError("legacy episode evidence exists; choose another output")
        if isinstance(replays, bool) or not isinstance(replays, int) or replays <= 0 or replays > 10:
            raise ValueError("replays must be an integer in [1,10]")
        if not isinstance(include_baselines, bool):
            raise ValueError("include_baselines must be boolean")
        if not isinstance(checkpoints, Mapping) or not checkpoints:
            raise ValueError("checkpoints must be a nonempty method-to-path mapping")
        for label in checkpoints:
            if (not isinstance(label, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", label)
                    or label in {"independent_entry", "sync_entry", "ma_usfa_ddqn"}
                    or label.startswith("fixed_source_")):
                raise ValueError("checkpoint method label is invalid or reserved")
        manifest = load_manifest(manifest_path)
        manifest.require_stage("pilot")
        marker = manifest.raw.get("paper_alignment", {})
        if not isinstance(marker, dict) or marker.get("task_seed") != 11:
            raise ValueError("manifest is not marked as the recovered task_seed=11 legacy protocol")
        sf_path = Path(sf_checkpoint).expanduser().resolve()
        source_root = Path(source_dir).expanduser().resolve()
        checkpoint_paths = {name: Path(path).expanduser().resolve() for name, path in checkpoints.items()}
        source_files = sorted(source_root.glob("source_*.pt"))
        if not source_files:
            raise ValueError("no frozen source checkpoints found")
        files = [Path(manifest_path).expanduser().resolve(), sf_path,
                 *checkpoint_paths.values(), *source_files]
        input_hashes = {str(path): sha256_file(path) for path in files}
        runtime = FrozenTrafficRuntime(manifest_path, source_root, sf_path, device=device)
        scenario_files = getattr(getattr(runtime, "spec", None), "files", None)
        if scenario_files is not None:
            for item in (scenario_files.network, scenario_files.routes):
                path = Path(item)
                files.append(path)
                input_hashes[str(path)] = sha256_file(path)
        protocol = load_protocol(manifest_path, runtime.ids)
        tasks = protocol.tasks("validation")
        if len(tasks) != 2:
            raise ValueError("paper-aligned evaluation requires exactly Homo and Hetero tasks")
        ids = {str(task["id"]).lower() for task in tasks}
        if not any("homo" in ident for ident in ids) or not any("hetero" in ident for ident in ids):
            raise ValueError("validation tasks are not labeled Homo/Hetero")
        declared_seeds = tuple(protocol.traffic_seeds("validation"))
        if declared_seeds != (int(traffic_seed),):
            raise ValueError("legacy protocol must declare exactly the old evaluation traffic seed")
        protocol_path = getattr(protocol, "path", None)
        if protocol_path is not None:
            files.append(Path(protocol_path))
            input_hashes[str(Path(protocol_path))] = sha256_file(Path(protocol_path))
        payloads, budgets, gamma = {}, {}, None
        for name, path in checkpoint_paths.items():
            models[name], payloads[name] = _load_paper_selector(path, device=device)
            _freeze(models[name])
            budgets[name] = _training_budget(payloads[name])
            current = _check_runtime_identity(payloads[name], runtime, protocol, sf_path)
            if gamma is not None and current != gamma:
                raise ValueError("selectors have different discount factors")
            gamma = current
            fingerprints[name] = _tensor_fingerprint(models[name])
        for model in runtime.library.models:
            _freeze(model)
        if runtime.sf is not None:
            _freeze(runtime.sf)
        fixed_methods = [f"fixed_source_{i}" for i in range(runtime.K)] if include_baselines else []
        methods = list(models) + (["independent_entry", "sync_entry"] + fixed_methods
                                 if include_baselines else [])
        report.update({
            "methods": methods, "checkpoint_methods": list(models),
            "task_ids": [str(task["id"]) for task in tasks],
            "traffic_seed": int(traffic_seed), "replays": int(replays),
            "expected_episode_count": len(tasks) * int(replays) * len(methods),
            "checkpoint_sha256": {name: input_hashes[str(path)] for name, path in checkpoint_paths.items()},
            "training_budgets": budgets, "sf_sha256": input_hashes[str(sf_path)],
            "source_hashes": runtime.source_hashes, "protocol_fingerprint": protocol.fingerprint,
            "node_ids": list(runtime.ids), "gamma": gamma, "horizon_steps": runtime.horizon,
            "source_scenario_binding_verified": bool(runtime.source_scenario_binding_verified),
            "episode_evidence_directory": artifacts.name,
            "return_definition": "episode_team_return = sum_t sum_i reward_i(t); undiscounted",
            "replay_definition": "Five repetitions of the same legacy traffic seed; not independent traffic draws",
            "scientific_scope": "Paper-aligned task/seed development evaluation of the repaired selector; historical residual-composer rows remain separate",
            "baseline_definitions": {
                "independent_entry": "Each node chooses its highest conditional-SF source-action score.",
                "sync_entry": "One common source maximizes mean conditional-SF score.",
                "fixed_source_k": "Every node executes frozen source k.",
            },
        })
        artifacts.mkdir(parents=True, exist_ok=True)
        baseline_model = next(iter(models.values()))
        for task in tasks:
            for replay_index in range(int(replays)):
                for method in methods:
                    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task["id"]))[:60]
                    path = artifacts / (f"{len(report['episodes']):05d}_{safe_task}_"
                                        f"{method}_replay{replay_index:02d}_seed{traffic_seed}.xml")
                    original_method = "ma_usfa_ddqn" if method in models else method
                    try:
                        row = _run_episode(runtime, models.get(method, baseline_model), original_method,
                                           task, int(traffic_seed), path, gamma, device, guard)
                        row["method"] = method
                        row["replay_index"] = replay_index
                        report["episodes"].append(row)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        report["episodes"].append({"status": "NOT_READY", "method": method,
                            "task_id": str(task["id"]), "traffic_seed": int(traffic_seed),
                            "replay_index": replay_index, "error": str(exc),
                            "evaluation_updates": 0, "test_updates": 0})
                        raise
        audit = runtime.audit_frozen()
        if any(_tensor_fingerprint(models[name]) != before for name, before in fingerprints.items()):
            raise RuntimeError("selector parameters or buffers changed")
        if {str(path): sha256_file(path) for path in files} != input_hashes:
            raise RuntimeError("legacy input or checkpoint changed during evaluation")
        if sorted(source_root.glob("source_*.pt")) != source_files:
            raise RuntimeError("source checkpoint membership changed during evaluation")
        if guard is not None:
            guard.check()
        evidence, unavailable = {}, []
        for row in report["episodes"]:
            path = artifacts / row["tripinfo_file"]
            if path.is_file():
                evidence[row["tripinfo_file"]] = sha256_file(path)
            else:
                unavailable.append(row["tripinfo_file"])
        paired = _paired_differences(report["episodes"], list(models), methods)
        report.update({"status": "PASS", "policy_performance_evaluated": True,
            "completed_episode_count": len(report["episodes"]), "checkpoint_files_unchanged": True,
            "selector_parameters_unchanged": True, "runtime_frozen_audit": audit,
            "selector_state_sha256": fingerprints, "input_sha256": input_hashes,
            "episode_evidence_sha256": evidence, "unavailable_episode_evidence": unavailable,
            "summary": _extended_summary(report["episodes"], methods),
            "paired_differences": paired})
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "NOT_READY"
        report["policy_performance_evaluated"] = False
        report["errors"].append(str(exc))
        report["completed_episode_count"] = sum(row.get("status") == "PASS" for row in report["episodes"])
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
