"""Verify the released frozen-test records without training or simulation."""
from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/synthetic"


def verify():
    inventory = json.loads((EVIDENCE / "SHA256SUMS.json").read_text())
    for relative, expected in inventory["files"].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError("Evidence path leaves the release")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence fingerprint changed: {relative}")
    data = json.loads((EVIDENCE / "test_evaluation.json").read_text())
    if data["status"] != "PASS" or any(data[k] for k in
            ("optimizer_updates", "evaluation_updates", "test_updates")):
        raise ValueError("The original evaluation is incomplete or updated a model")
    groups, all_runs = defaultdict(list), defaultdict(list)
    bounds = {}
    for row in data["rows"]:
        value = float(row["return"])
        if not math.isfinite(value):
            raise ValueError("Non-finite return")
        key = (float(row["kappa"]), row["method"], row["seed"])
        groups[(*key, row["kind"])].append(value)
        all_runs[key].append(value)
        if row["method"] == "dp_optimal":
            bounds[(row["kappa"], row["task_id"])] = value
    if len(data["rows"]) != 4000 or any(len(v) != 32 for v in all_runs.values()):
        raise ValueError("Incomplete 32-target evaluation matrix")
    for row in data["rows"]:
        if row["return"] > bounds[(row["kappa"], row["task_id"])] + 1e-6:
            raise ValueError("DP upper-bound violation")
    for key, values in all_runs.items():
        groups[(*key, "all")] = values
    aggregate = defaultdict(list)
    for (kappa, method, seed, group), values in groups.items():
        aggregate[(kappa, method, group)].append(statistics.mean(values))
    compared = 0
    with (EVIDENCE / "analysis/group_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = aggregate[(float(row["kappa"]), row["method"], row["group"])]
            if not math.isclose(statistics.mean(values), float(row["mean_return"]),
                                rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Mean mismatch: {row}")
            if row["sample_std_training_seed_mean"]:
                if not math.isclose(statistics.stdev(values),
                        float(row["sample_std_training_seed_mean"]),
                        rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"Training-seed SD mismatch: {row}")
            compared += 1
    return {"status": "PASS", "evidence_files": len(inventory["files"]),
            "test_rows": len(data["rows"]), "grouped_rows_recomputed": compared,
            "optimizer_updates": 0,
            "scope": "Integrity and arithmetic check of recorded evidence; no model training or simulator run"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
