"""Numerically replay the released frozen synthetic models.

This evaluates already-published tasks. It is a reproduction check, not a new
independent test and never updates, selects or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "toy"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--method", default="ma_usfa_ddqn",
                   help="Frozen method name or all")
    p.add_argument("--kappa", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--output", type=Path, default=Path("runs/frozen_replay.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing output; choose a fresh --output")
    import numpy as np
    import torch
    from repaired.evidence_evaluate import make_sources, _evaluate_run
    from repaired.legacy_world import SFWorldExact
    from repaired.task_protocol import load_tasks
    from verify_evidence import verify

    verify()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    evidence = ROOT / "evidence/synthetic"
    portable = json.loads((evidence / "portable_models.json").read_text())
    runs = [r for r in portable["runs"]
            if (args.method == "all" or r["method"] == args.method)
            and (args.kappa is None or r["kappa"] == args.kappa)
            and (args.seed is None or r["seed"] == args.seed)]
    if not runs:
        raise ValueError("No frozen run matches the requested filters")
    tasks = load_tasks(ROOT / "toy/protocol", args.split,
                       allow_test=args.split == "test")
    original = json.loads((evidence / f"{args.split}_evaluation.json").read_text())
    expected = {(r["method"], r["kappa"], r["seed"], r["task_id"]): r["return"]
                for r in original["rows"]}
    started = time.monotonic()
    rows, receipts, maximum = [], [], 0.0
    world_cache = {}
    for run in runs:
        kappa = float(run["kappa"])
        if kappa not in world_cache:
            world = SFWorldExact(kappa=kappa, **portable["world"])
            world_cache[kappa] = (world, make_sources(world))
        world, sources = world_cache[kappa]
        result, receipt = _evaluate_run(run, world, tasks, sources, root=ROOT)
        receipts.append(receipt)
        for row in result:
            key = (run["method"], run["kappa"], run["seed"], row["task_id"])
            error = abs(float(row["return"]) - expected[key])
            maximum = max(maximum, error)
            rows.append({"method": run["method"], "kappa": run["kappa"],
                         "seed": run["seed"], **row,
                         "absolute_error_from_record": error})
            if not np.isfinite(error) or error > 1e-6:
                raise RuntimeError(f"Frozen replay mismatch: {key}, {error}")
        print(json.dumps({"status": "RUN_REPLAYED", "method": run["method"],
                          "kappa": kappa, "seed": run["seed"]}), flush=True)
    value = {"status": "PASS", "split": args.split,
             "controller_runs": len(runs), "objectives_per_run": len(tasks),
             "rows": rows, "run_receipts": receipts, "maximum_absolute_error": maximum,
             "optimizer_updates": 0, "elapsed_seconds": time.monotonic()-started,
             "source_tree_sha256": {x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                                    for x in (ROOT / "toy/repaired").glob("*.py")},
             "scope": "Replay of already-published frozen-model evidence; no new test, selection or training"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in value.items() if k not in ("rows", "run_receipts", "source_tree_sha256")}, indent=2))


if __name__ == "__main__":
    main()
