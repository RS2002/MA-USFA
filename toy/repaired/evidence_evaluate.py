"""Frozen, evaluation-only synthetic evidence with an explicit sealed-test gate.

This module never trains a model.  ``freeze`` reads only train/validation tasks
and protocol metadata; it does not open the sealed test file.  Final test access
requires a complete, development-finished matrix, unchanged files and code,
and an explicit command-line flag.  A test-access receipt prevents silent retry.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .exact import (compose_fixed, evaluate_policy, evaluate_sources_sf,
                    legacy_sources, optimal_policy)
from .legacy_world import SFWorldExact
from .learning import model_hash
from .paper_aligned_attention import PaperAlignedAttention
from .paper_aligned_lower import load_context_sf
from .paper_aligned_run import evaluate_attention
from .task_protocol import digest, load_tasks

VERSION = "icml-frozen-synthetic-evaluation-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_METHODS = (
    "ma_usfa_ddqn", "entry_greedy", "no_attention", "no_sf",
    "no_teammate_conditioning", "shared_joint_ddqn", "shared_joint_ddqn_total",
)
KIND_BY_METHOD = {
    "ma_usfa_ddqn": "main_attention", "entry_greedy": "entry_greedy",
    "no_attention": "ablation", "no_sf": "ablation",
    "no_teammate_conditioning": "ablation",
    "shared_joint_ddqn": "shared_joint_ddqn",
    "shared_joint_ddqn_total": "shared_joint_ddqn",
}
CODE_NAMES = (
    "evidence_evaluate.py", "exact.py", "legacy_world.py", "learning.py",
    "task_protocol.py", "paper_aligned_run.py", "paper_aligned_attention.py",
    "paper_aligned_lower.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value, root=PROJECT_ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def path_ref(path, root=PROJECT_ROOT):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        # Unit-test fixtures can be outside the checkout.  Normal plan paths
        # and delivered command examples remain relative to the project root.
        return str(path)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation deliberately refuses any overwrite or silent retry.
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _canonical_array_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def make_sources(world):
    # This is the same kappa=0, infinite-horizon library used by the main runs.
    base = legacy_sources(SFWorldExact(horizon=world.horizon,
                                      gamma=world.gamma, kappa=0.0))["policies"]
    return np.broadcast_to(base[:, None],
                           (len(base), world.horizon, world.S, 2)).copy()


def _check_matrix(plan):
    matrix = plan.get("matrix", {})
    kappas = [float(x) for x in matrix.get("kappas", [])]
    seeds = [int(x) for x in matrix.get("seeds", [])]
    if (not kappas or not seeds or len(set(kappas)) != len(kappas)
            or len(set(seeds)) != len(seeds)
            or any(not math.isfinite(k) or not 0 <= k <= 1 for k in kappas)):
        raise ValueError("Declare unique finite kappas and training seeds in matrix")
    required = tuple(plan.get("required_methods", REQUIRED_METHODS))
    if set(required) != set(REQUIRED_METHODS):
        raise ValueError("The seven predeclared evidence methods must all be required")
    expected = {(k, seed) for k in kappas for seed in seeds}
    groups = defaultdict(set)
    for run in plan.get("runs", []):
        method = run.get("method")
        if method not in KIND_BY_METHOD or run.get("kind") != KIND_BY_METHOD[method]:
            raise ValueError(f"Unsupported or inconsistent method/kind: {method}")
        pair = (float(run["kappa"]), int(run["seed"]))
        if pair not in expected or pair in groups[method]:
            raise ValueError(f"Duplicate or out-of-matrix run: {method}, {pair}")
        groups[method].add(pair)
        if method in ("no_attention", "no_sf", "no_teammate_conditioning"):
            if run.get("variant", method) != method:
                raise ValueError("Ablation method and variant must agree")
        for key in ("budget_upper", "budget_lower"):
            if int(run.get(key, -1)) < 0:
                raise ValueError(f"Missing nonnegative {key}")
        if method == "entry_greedy":
            if run.get("upper") or int(run["budget_upper"]) != 0:
                raise ValueError("entry_greedy has no upper checkpoint or upper training")
        elif not run.get("upper") or int(run["budget_upper"]) <= 0:
            raise ValueError("A learned controller requires its upper checkpoint and budget")
        if method in ("no_sf", "shared_joint_ddqn", "shared_joint_ddqn_total"):
            if run.get("lower") or int(run["budget_lower"]) != 0:
                raise ValueError("no_sf/shared baseline must not load a lower SF")
        elif not run.get("lower") or int(run["budget_lower"]) <= 0:
            raise ValueError("SF-based controllers require a lower checkpoint and budget")
    if not groups:
        raise ValueError("No runs were declared")
    # A validation rehearsal can contain a subset of methods, but each method
    # that is included must cover the entire declared matrix.
    if any(pairs != expected for pairs in groups.values()):
        raise ValueError("Each included method must cover the complete declared matrix")
    complete = all(groups.get(method) == expected for method in required)
    eligible = bool(plan.get("development_complete", False)) and complete
    if plan.get("final_test_ready", False) and not eligible:
        raise ValueError("Final test requires all development complete and all seven method matrices")
    ready = bool(plan.get("final_test_ready", False)) and eligible
    return {"kappas": kappas, "seeds": seeds, "required_methods": list(required),
            "matrix_complete": complete, "final_test_ready": ready}


def _check_number(payload, key, expected, label):
    if key not in payload or not np.isclose(float(payload[key]), float(expected),
                                          rtol=0, atol=1e-12):
        raise ValueError(f"{label} {key} does not match frozen run")


def load_run_models(run, world, sources, root=PROJECT_ROOT):
    """Load only trusted local checkpoints; never instantiate an optimizer."""
    kind = run["kind"]
    lower, lower_payload = None, None
    if run.get("lower"):
        lower_path = resolve_path(run["lower"], root)
        if run["method"] == "no_teammate_conditioning":
            from .evidence_ablations import load_ablation_lower
            lower, lower_payload = load_ablation_lower(lower_path)
        else:
            lower, lower_payload = load_context_sf(lower_path)
        for key, expected in (("seed", run["seed"]), ("kappa", run["kappa"]),
                              ("gamma", world.gamma), ("steps", run["budget_lower"])):
            _check_number(lower_payload, key, expected, "lower")
        source_hash = lower_payload.get("source_sha256", lower_payload.get("sources_sha256"))
        if source_hash != _canonical_array_hash(sources):
            raise ValueError("Lower checkpoint source-library identity mismatch")
    if kind == "main_attention":
        payload = torch.load(resolve_path(run["upper"], root),
                             map_location="cpu", weights_only=False)
        if payload.get("mode") != "ddqn":
            raise ValueError("Main checkpoint must be the team-DDQN controller")
        model = PaperAlignedAttention(gamma=world.gamma, horizon=world.horizon)
        model.load_state_dict(payload["model"])
    elif kind == "entry_greedy":
        # A parameter-free controller makes both its behavior and frozen
        # identity independent of random initialization.
        from .evidence_ablations import make_ablation_model
        model = make_ablation_model("entry_greedy", gamma=world.gamma,
                                    horizon=world.horizon)
        payload = {"seed": run["seed"], "budget": 0,
                   "sources_sha256": _canonical_array_hash(sources)}
    elif kind == "ablation":
        from .evidence_ablations import load_ablation_upper
        model, payload = load_ablation_upper(resolve_path(run["upper"], root))
        if payload.get("variant") != run.get("variant", run["method"]):
            raise ValueError("Loaded ablation variant does not match the plan")
        if payload.get("lower_sha256") is not None:
            expected_lower = (sha256_file(resolve_path(run["lower"], root))
                              if run.get("lower") else None)
            if payload["lower_sha256"] != expected_lower:
                raise ValueError("Ablation upper/lower checkpoint binding mismatch")
    elif kind == "shared_joint_ddqn":
        from .evidence_shared_baseline import load_shared_baseline
        model, payload = load_shared_baseline(resolve_path(run["upper"], root))
    else:
        raise ValueError(kind)
    if kind == "shared_joint_ddqn" and "best validation" not in payload.get("selection", ""):
        raise ValueError("Shared-policy evaluation must load the validation-selected checkpoint")
    if kind == "ablation" and int(payload.get("steps", -1)) != int(run["budget_upper"]):
        raise ValueError("Ablation training did not complete the declared upper budget")
    if kind == "main_attention" and payload.get("logs"):
        if int(payload["logs"][-1]["steps"]) != int(run["budget_upper"]):
            raise ValueError("Main training did not complete the declared upper budget")
    cfg = payload.get("config", payload)
    _check_number(cfg, "seed", run["seed"], "upper")
    _check_number(cfg, "budget", run["budget_upper"], "upper")
    if kind != "shared_joint_ddqn":
        if payload.get("sources_sha256") != _canonical_array_hash(sources):
            raise ValueError("Upper checkpoint source-library identity mismatch")
    # New checkpoints record their world.  Main historical upper checkpoints
    # obtain this binding from their checked lower and source metadata instead.
    for key, expected in (("kappa", world.kappa), ("gamma", world.gamma),
                          ("horizon", world.horizon)):
        if key in cfg:
            _check_number(cfg, key, expected, "upper")
    for net in (model, lower):
        if net is not None:
            net.eval()
            for parameter in net.parameters():
                parameter.requires_grad_(False)
    return model, lower, payload


def freeze_plan(plan_path, output_path, root=PROJECT_ROOT):
    plan_path = resolve_path(plan_path, root)
    output_path = resolve_path(output_path, root)
    if output_path.exists():
        raise FileExistsError("A frozen plan is immutable; use a new explicitly named path")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    status = _check_matrix(plan)
    settings = plan.get("world", {})
    if settings != {"horizon": 40, "gamma": .95}:
        raise ValueError("This evidence protocol is the declared H40/gamma.95 SFWorld")
    selection = plan.get("selection", "")
    if not selection or "validation" not in selection.lower():
        raise ValueError("Declare the validation checkpoint-selection rule")
    protocol_dir = resolve_path(plan["protocol_dir"], root)
    protocol_manifest = protocol_dir / "manifest.json"
    meta = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    # Intentionally do not call verify_protocol here: it fingerprints sealed
    # bytes too.  No sealed-file open is necessary to create the frozen plan.
    for split in ("train", "validation"):
        load_tasks(protocol_dir, split)
    files = {path_ref(plan_path, root): sha256_file(plan_path),
             path_ref(protocol_manifest, root): sha256_file(protocol_manifest)}
    for split in ("train", "validation"):
        p = protocol_dir / meta["files"][split]["file"]
        files[path_ref(p, root)] = sha256_file(p)
    for item in plan.get("evidence_files", []):
        p = resolve_path(item, root)
        files[path_ref(p, root)] = sha256_file(p)
    code_names = list(CODE_NAMES)
    if any(r["kind"] in ("ablation", "entry_greedy") for r in plan["runs"]):
        code_names.append("evidence_ablations.py")
    if any(r["kind"] == "shared_joint_ddqn" for r in plan["runs"]):
        code_names.append("evidence_shared_baseline.py")
    for name in code_names:
        p = Path(__file__).with_name(name)
        files[path_ref(p, root)] = sha256_file(p)
    worlds = {}
    source_cache = {}
    runs = []
    for run in plan["runs"]:
        run = dict(run)
        for key in ("upper", "lower"):
            if run.get(key):
                p = resolve_path(run[key], root)
                files[path_ref(p, root)] = sha256_file(p)
                run[key] = path_ref(p, root)
        if run["kind"] == "shared_joint_ddqn":
            summary_path = resolve_path(run["upper"], root).with_name("summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (summary.get("status") != "PASS" or summary.get("complete") is not True
                    or int(summary.get("steps", -1)) != int(run["budget_upper"])):
                raise ValueError("Shared-policy development run is not complete")
            files[path_ref(summary_path, root)] = sha256_file(summary_path)
        kappa = float(run["kappa"])
        if kappa not in worlds:
            worlds[kappa] = SFWorldExact(kappa=kappa, **settings)
            source_cache[kappa] = make_sources(worlds[kappa])
        model, lower, payload = load_run_models(run, worlds[kappa], source_cache[kappa], root)
        run["parameter_sha256"] = model_hash(model)
        run["lower_parameter_sha256"] = model_hash(lower) if lower is not None else None
        run["source_array_sha256"] = _canonical_array_hash(source_cache[kappa])
        run["selected_validation_return"] = payload.get("best_validation", payload.get("validation", {}).get("mean_return"))
        run["training_transition_cost"] = {
            "lower": int(run["budget_lower"]), "upper": int(run["budget_upper"]),
            "total": int(run["budget_lower"]) + int(run["budget_upper"]),
        }
        runs.append(run)
    doc = {
        "version": VERSION, "created_utc": utc_now(), "root": path_ref(root, root),
        "status": "FROZEN", **status, "development_complete": bool(plan.get("development_complete", False)),
        "world": settings, "protocol_dir": path_ref(protocol_dir, root),
        "protocol": {"version": meta["version"], "counts": meta["counts"],
                     "declared_split_fingerprints": meta["files"]},
        "selection": selection, "runs": runs, "files_sha256": files,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__},
        "test_tasks_parsed": False, "optimizer_updates": 0,
        "evaluation": "exact finite-horizon discounted team return under the declared initial distribution",
        "fixed_comparators": ["exact_sync", "primitive_independent", "joint_gpi", "dp_optimal"],
        "scratch_scope": "Validation-target-specific scratch models are not reused on unseen test targets. No target-specific scratch comparator is claimed by this evaluation.",
        "historical_main_binding": "Legacy main upper checkpoints lack world/task metadata; the freeze checks their recorded seed, upper budget, source-array hash, paired lower world/seed/budget, and exact checkpoint bytes. Historical provenance manifests may be added as evidence_files.",
    }
    doc["frozen_sha256"] = digest(doc)
    write_new_json(output_path, doc)
    return doc


def verify_frozen(path, root=PROJECT_ROOT):
    doc = json.loads(resolve_path(path, root).read_text(encoding="utf-8"))
    own = dict(doc)
    claimed = own.pop("frozen_sha256", None)
    if doc.get("version") != VERSION or doc.get("status") != "FROZEN" or digest(own) != claimed:
        raise ValueError("Frozen manifest identity or digest mismatch")
    for value, expected in doc["files_sha256"].items():
        if sha256_file(resolve_path(value, root)) != expected:
            raise ValueError(f"Frozen input or code changed: {value}")
    return doc


def aggregate_rows(rows):
    """First average actual objectives, then summarize independent training seeds."""
    per_run = []
    grouped = defaultdict(list)
    for row in rows:
        if not np.isfinite(row["return"]):
            raise ValueError("Cannot aggregate nonfinite performance")
        grouped[(float(row["kappa"]), row["method"], row.get("seed"))].append(row)
    seen = set()
    for (kappa, method, seed), items in sorted(grouped.items(), key=lambda x: str(x[0])):
        ids = [r["task_id"] for r in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task within run")
        kinds = sorted({r["kind"] for r in items})
        for kind in ["all"] + kinds:
            values = [float(r["return"]) for r in items if kind == "all" or r["kind"] == kind]
            per_run.append({"kappa": kappa, "method": method, "seed": seed,
                            "group": kind, "objectives": len(values),
                            "mean_return": float(np.mean(values))})
        signature = tuple(sorted(ids))
        if seen and signature not in seen:
            raise ValueError("Methods or settings have different objective coverage")
        seen.add(signature)
    summary = []
    g = defaultdict(list)
    for row in per_run:
        g[(row["kappa"], row["method"], row["group"])].append(row)
    for (kappa, method, kind), items in sorted(g.items()):
        values = [r["mean_return"] for r in items]
        fixed = all(r["seed"] is None for r in items)
        summary.append({"kappa": kappa, "method": method, "group": kind,
                        "mean_return": float(np.mean(values)),
                        "sample_std_training_seed_mean": (float(np.std(values, ddof=1))
                                                          if len(values) > 1 else None),
                        "training_seeds": [] if fixed else [r["seed"] for r in items],
                        "deterministic_reference": fixed,
                        "objectives_per_run": items[0]["objectives"]})
    return {"per_run": per_run, "summary": summary,
            "aggregation": "Mean over every objective within each run; then mean/sample SD across training-seed means. Task groups are weighted by their actual counts. Exact fixed comparators have no independent training seeds."}


def _fixed_rows(world, sources, tasks, sf):
    rows = []
    for task in tasks:
        weights = np.asarray(task["weights"], dtype=np.float64)
        policies = compose_fixed(world, sources, sf, weights)
        named = {"exact_sync": policies["sync"],
                 "primitive_independent": policies["primitive_indep"],
                 "joint_gpi": policies["true_joint_gpi"]}
        for method, policy in named.items():
            value = evaluate_policy(world, policy, weights)["initial_return"]
            rows.append({"kappa": world.kappa, "method": method, "seed": None,
                         "task_id": task["id"], "kind": task["kind"], "return": value})
        rows.append({"kappa": world.kappa, "method": "dp_optimal", "seed": None,
                     "task_id": task["id"], "kind": task["kind"],
                     "return": optimal_policy(world, weights)["initial_return"]})
    return rows


def _evaluate_run(run, world, tasks, sources, root):
    source_hash = _canonical_array_hash(sources)
    model, lower, _ = load_run_models(run, world, sources, root)
    before = model_hash(model)
    lb = model_hash(lower) if lower is not None else None
    if before != run["parameter_sha256"] or lb != run["lower_parameter_sha256"]:
        raise ValueError("Reconstructed model differs from frozen parameter identity")
    if run["kind"] == "shared_joint_ddqn":
        from .evidence_shared_baseline import evaluate_shared_baseline
        result = evaluate_shared_baseline(model, world, tasks)
    elif run["kind"] in ("ablation", "entry_greedy"):
        from .evidence_ablations import evaluate_ablation
        result = evaluate_ablation(model, world, tasks, sources, lower)
    else:
        result = evaluate_attention(model, world, tasks, sources, lower)
    model.eval()
    if lower is not None:
        lower.eval()
    after = model_hash(model)
    la = model_hash(lower) if lower is not None else None
    if before != after or lb != la or source_hash != _canonical_array_hash(sources):
        raise RuntimeError("Frozen evaluation mutated a model or source library")
    expected_ids = {t["id"] for t in tasks}
    if len(result["rows"]) != len(tasks) or {r["task_id"] for r in result["rows"]} != expected_ids:
        raise ValueError("Incomplete or duplicated evaluation tasks")
    return [{"kappa": world.kappa, "method": run["method"], "seed": int(run["seed"]),
             "task_id": r["task_id"], "kind": r["kind"], "return": float(r["return"])}
            for r in result["rows"]], {
                "method": run["method"], "kappa": world.kappa, "seed": int(run["seed"]),
                "parameter_sha256_before": before, "parameter_sha256_after": after,
                "lower_parameter_sha256_before": lb, "lower_parameter_sha256_after": la,
                "source_array_sha256": source_hash, "optimizer_updates": 0,
            }


def evaluate_frozen(frozen_path, split, output_dir, allow_test=False, root=PROJECT_ROOT):
    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    if split == "test" and not allow_test:
        raise PermissionError("Test objectives require the explicit --allow-test flag")
    frozen_path = resolve_path(frozen_path, root)
    frozen = verify_frozen(frozen_path, root)
    if split == "test" and not frozen.get("final_test_ready", False):
        raise PermissionError("The final test is gated until all development matrices are complete")
    receipt_path = frozen_path.with_name(frozen_path.stem + ".test_access_started.json")
    if split == "test" and receipt_path.exists():
        raise FileExistsError("This frozen protocol already has a test-access receipt; no silent retry")
    output_dir = resolve_path(output_dir, root)
    if output_dir.exists():
        raise FileExistsError("Evaluation outputs are immutable; refusing overwrite or silent retry")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {"version": VERSION, "frozen_sha256": frozen["frozen_sha256"],
               "split": split, "started_utc": utc_now(),
               "output": path_ref(output_dir, root), "optimizer_updates": 0}
    if split == "test":
        # Receipt is created before opening the sealed task file.  A crash or
        # rejected run remains visible and cannot cause an automatic second look.
        write_new_json(receipt_path, receipt)
    write_new_json(output_dir / "evaluation_started.json", receipt)
    rows, run_receipts = [], []
    try:
        protocol_dir = resolve_path(frozen["protocol_dir"], root)
        tasks = load_tasks(protocol_dir, split, allow_test=allow_test)
        task_before = digest(tasks)
        declared_count = sum(frozen["protocol"]["counts"][split].values())
        if len(tasks) != declared_count:
            raise ValueError("Task count differs from frozen protocol manifest")
        torch.set_num_threads(min(2, max(1, int(os.environ.get("MA_USFA_THREAD_BUDGET", "2")))))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        for kappa in frozen["kappas"]:
            world = SFWorldExact(kappa=float(kappa), **frozen["world"])
            sources = make_sources(world)
            sf = evaluate_sources_sf(world, sources)
            rows.extend(_fixed_rows(world, sources, tasks, sf))
            for run in frozen["runs"]:
                if float(run["kappa"]) != float(kappa):
                    continue
                result, r = _evaluate_run(run, world, tasks, sources, root)
                rows.extend(result)
                run_receipts.append(r)
            # Incremental progress is exact raw evidence, never a successful-
            # subset aggregate.  It is retained if a later setting fails.
            write_new_json(output_dir / f"completed_kappa_{str(kappa).replace('.', 'p')}.json",
                           {"kappa": float(kappa), "rows": [r for r in rows if r["kappa"] == float(kappa)],
                            "optimizer_updates": 0})
        if digest(tasks) != task_before:
            raise RuntimeError("Evaluation mutated objective inputs")
        verify_frozen(frozen_path, root)
        aggregate = aggregate_rows(rows)
        report = {"version": VERSION, "status": "PASS", "split": split,
                  "frozen_sha256": frozen["frozen_sha256"], "test_accessed": split == "test",
                  "evaluation_updates": 0, "test_updates": 0, "optimizer_updates": 0,
                  "objective_count": len(tasks), "objective_values_sha256": task_before,
                  "rows": rows, "run_receipts": run_receipts, **aggregate,
                  "elapsed_seconds": time.perf_counter() - started,
                  "scratch_scope": frozen["scratch_scope"]}
        write_new_json(output_dir / "evaluation.json", report)
        return report
    except Exception as exc:
        write_new_json(output_dir / "evaluation_failed.json", {
            "version": VERSION, "status": "FAILED", "split": split,
            "frozen_sha256": frozen["frozen_sha256"], "error": str(exc),
            "rows_completed": rows, "run_receipts": run_receipts,
            "aggregate": None, "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates": 0,
        })
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--frozen", required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), required=True)
    evaluate.add_argument("--allow-test", action="store_true")
    evaluate.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--frozen", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_plan(args.plan, args.output)
        print(json.dumps({"status": result["status"], "final_test_ready": result["final_test_ready"],
                          "runs": len(result["runs"]), "test_tasks_parsed": False}))
    elif args.command == "verify":
        result = verify_frozen(args.frozen)
        print(json.dumps({"status": "PASS", "frozen_sha256": result["frozen_sha256"],
                          "final_test_ready": result["final_test_ready"]}))
    else:
        result = evaluate_frozen(args.frozen, args.split, args.output, args.allow_test)
        print(json.dumps({"status": result["status"], "split": result["split"],
                          "objective_count": result["objective_count"],
                          "rows": len(result["rows"]), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
