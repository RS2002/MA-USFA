"""Tests of freezing, sealed-test gates and aggregation; no scientific training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import evidence_evaluate as ev
from .paper_aligned_attention import PaperAlignedAttention
from .paper_aligned_lower import ContextActionSF
from .task_protocol import canonical_json, digest


class FrozenEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mausfa_frozen_evidence_")
        self.root = Path(self.tmp.name)
        self.protocol = self.root / "protocol"
        self.protocol.mkdir()
        self.sources = np.zeros((4, 40, 625, 2), dtype=np.int64)
        self.task = {"id": "validation_homo_000", "kind": "homo",
                     "weights": [[1., .2, .1, .3], [1., .2, .1, .3]]}
        files = {}
        for split in ("train", "validation"):
            payload = {"split": split, "objectives": [dict(self.task, id=f"{split}_homo_000")]}
            name = split + ".json"
            (self.protocol / name).write_text(canonical_json(payload) + "\n")
            files[split] = {"file": name, "sha256": digest(payload)}
        # Deliberately absent.  freeze/verify/validation gates must not need it.
        files["test"] = {"file": "test.sealed.json", "sha256": "0" * 64}
        meta = {"version": "test-fixture", "files": files,
                "counts": {s: {"homo": 1, "distinct": 0, "overlap": 0}
                           for s in ("train", "validation", "test")}}
        (self.protocol / "manifest.json").write_text(canonical_json(meta))
        torch.manual_seed(6)
        lower = ContextActionSF()
        torch.save({"model": lower.state_dict(), "seed": 100, "kappa": 0., "gamma": .95,
                    "steps": 80, "source_sha256": ev._canonical_array_hash(self.sources)},
                   self.root / "lower.pt")
        upper = PaperAlignedAttention()
        torch.save({"model": upper.state_dict(), "mode": "ddqn", "seed": 100,
                    "budget": 80, "best_validation": 1.,
                    "sources_sha256": ev._canonical_array_hash(self.sources)},
                   self.root / "upper.pt")
        self.plan = {"protocol_dir": "protocol", "world": {"horizon": 40, "gamma": .95},
                     "matrix": {"kappas": [0.], "seeds": [100]},
                     "selection": "validation mean best within budget; first tie",
                     "development_complete": False, "final_test_ready": False,
                     "runs": [{"method": "ma_usfa_ddqn", "kind": "main_attention",
                               "kappa": 0., "seed": 100, "budget_upper": 80,
                               "budget_lower": 80, "upper": "upper.pt", "lower": "lower.pt"}]}
        self.plan_path = self.root / "plan.json"
        self.frozen_path = self.root / "frozen.json"
        self._write_plan()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def freeze(self):
        with patch.object(ev, "make_sources", return_value=self.sources.copy()):
            return ev.freeze_plan(self.plan_path, self.frozen_path, self.root)

    def test_freeze_does_not_open_sealed_file_and_refuses_overwrite(self):
        doc = self.freeze()
        self.assertFalse((self.protocol / "test.sealed.json").exists())
        self.assertFalse(doc["test_tasks_parsed"])
        self.assertFalse(doc["final_test_ready"])
        self.assertEqual(ev.verify_frozen(self.frozen_path, self.root)["frozen_sha256"], doc["frozen_sha256"])
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_test_requires_flag_and_complete_development_before_loading(self):
        self.freeze()
        with patch.object(ev, "load_tasks", side_effect=AssertionError("must stay sealed")):
            with self.assertRaises(PermissionError):
                ev.evaluate_frozen(self.frozen_path, "test", "out1", False, self.root)
            with self.assertRaises(PermissionError):
                ev.evaluate_frozen(self.frozen_path, "test", "out2", True, self.root)
        self.assertFalse((self.root / "out1").exists())
        self.assertFalse((self.root / "out2").exists())
        self.assertFalse(self.frozen_path.with_name("frozen.test_access_started.json").exists())

    def test_hash_change_rejected_before_task_loading(self):
        self.freeze()
        with (self.root / "upper.pt").open("ab") as f:
            f.write(b"changed")
        with patch.object(ev, "load_tasks", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "Frozen input or code changed"):
                ev.evaluate_frozen(self.frozen_path, "validation", "changed", False, self.root)
        self.assertFalse((self.root / "changed").exists())

    def test_matrix_rejects_missing_pairs_duplicate_runs_and_early_final(self):
        bad = copy.deepcopy(self.plan)
        bad["matrix"]["seeds"].append(101)
        with self.assertRaisesRegex(ValueError, "complete declared matrix"):
            ev._check_matrix(bad)
        bad = copy.deepcopy(self.plan)
        bad["runs"].append(copy.deepcopy(bad["runs"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ev._check_matrix(bad)
        bad = copy.deepcopy(self.plan)
        bad["development_complete"] = bad["final_test_ready"] = True
        with self.assertRaisesRegex(ValueError, "all seven method matrices"):
            ev._check_matrix(bad)

    def test_final_readiness_requires_explicit_plan_flag_even_after_development(self):
        plan = copy.deepcopy(self.plan)
        plan["development_complete"] = True
        plan["runs"] = []
        for method in ev.REQUIRED_METHODS:
            run = dict(self.plan["runs"][0], method=method, kind=ev.KIND_BY_METHOD[method])
            if method == "entry_greedy":
                run.pop("upper")
                run["budget_upper"] = 0
            if method in ("no_sf", "shared_joint_ddqn", "shared_joint_ddqn_total"):
                run.pop("lower")
                run["budget_lower"] = 0
            plan["runs"].append(run)
        self.assertTrue(ev._check_matrix(plan)["matrix_complete"])
        self.assertFalse(ev._check_matrix(plan)["final_test_ready"])
        plan["final_test_ready"] = True
        self.assertTrue(ev._check_matrix(plan)["final_test_ready"])

    def test_loading_freezes_parameters_and_preserves_checkpoint_bytes(self):
        before = {p.name: ev.sha256_file(p) for p in (self.root / "upper.pt", self.root / "lower.pt")}
        world = ev.SFWorldExact(kappa=0.)
        model, lower, _ = ev.load_run_models(self.plan["runs"][0], world, self.sources, self.root)
        self.assertTrue(all(not p.requires_grad for m in (model, lower) for p in m.parameters()))
        self.assertTrue(not model.training and not lower.training)
        self.assertEqual(before, {p.name: ev.sha256_file(p) for p in (self.root / "upper.pt", self.root / "lower.pt")})

    def test_aggregation_weights_objectives_and_seed_means(self):
        rows = []
        for seed, offset in ((100, 0.), (101, 2.)):
            for kind, n, value in (("homo", 4, 1.), ("distinct", 6, 3.), ("overlap", 6, 5.)):
                for j in range(n):
                    rows.append({"kappa": 1., "method": "m", "seed": seed,
                                 "task_id": f"{kind}{j}", "kind": kind, "return": value + offset})
        out = ev.aggregate_rows(rows)
        all_group = next(x for x in out["summary"] if x["group"] == "all")
        self.assertAlmostEqual(all_group["mean_return"], (4 + 18 + 30) / 16 + 1.)
        self.assertAlmostEqual(all_group["sample_std_training_seed_mean"], np.sqrt(2.))
        self.assertEqual(all_group["objectives_per_run"], 16)
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            ev.aggregate_rows(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "different objective coverage"):
            ev.aggregate_rows(rows[:-1])

    def test_failed_evaluation_keeps_raw_rows_but_no_success_subset_summary(self):
        self.freeze()
        def fail(*args):
            raise RuntimeError("fixture failure")
        with patch.object(ev, "make_sources", return_value=self.sources.copy()), \
             patch.object(ev, "evaluate_sources_sf", return_value=np.empty(0)), \
             patch.object(ev, "_fixed_rows", return_value=[]), \
             patch.object(ev, "_evaluate_run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                ev.evaluate_frozen(self.frozen_path, "validation", "failed", False, self.root)
        report = json.loads((self.root / "failed/evaluation_failed.json").read_text())
        self.assertIsNone(report["aggregate"])
        self.assertFalse((self.root / "failed/evaluation.json").exists())
        with self.assertRaises(FileExistsError):
            ev.evaluate_frozen(self.frozen_path, "validation", "failed", False, self.root)


    def test_actual_evaluation_wrapper_checks_model_and_source_immutability(self):
        doc = self.freeze()
        run = doc["runs"][0]
        world = ev.SFWorldExact(kappa=0.)
        expected = {"rows": [{"task_id": self.task["id"], "kind": "homo", "return": 1.}]}
        before = ev.sha256_file(self.root / "upper.pt")
        with patch.object(ev, "evaluate_attention", return_value=expected):
            rows, receipt = ev._evaluate_run(run, world, [self.task], self.sources, self.root)
        self.assertEqual(receipt["parameter_sha256_before"], receipt["parameter_sha256_after"])
        self.assertEqual(before, ev.sha256_file(self.root / "upper.pt"))
        self.assertEqual(len(rows), 1)
        def bad_evaluator(model, *args):
            with torch.no_grad():
                next(model.parameters()).add_(1.)
            return expected
        with patch.object(ev, "evaluate_attention", side_effect=bad_evaluator):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                ev._evaluate_run(run, world, [self.task], self.sources, self.root)

    def test_entry_greedy_has_reproducible_identity_without_upper_parameters(self):
        run = dict(self.plan["runs"][0], method="entry_greedy", kind="entry_greedy", budget_upper=0)
        run.pop("upper")
        world = ev.SFWorldExact(kappa=0.)
        model1, _, _ = ev.load_run_models(run, world, self.sources, self.root)
        torch.manual_seed(999)
        model2, _, _ = ev.load_run_models(run, world, self.sources, self.root)
        self.assertEqual(sum(p.numel() for p in model1.parameters()), 0)
        self.assertEqual(ev.model_hash(model1), ev.model_hash(model2))
        obs = torch.zeros(2, 5)
        weights = torch.zeros(2, 2, 4)
        actions = torch.zeros(2, 2, 4, dtype=torch.long)
        values = torch.tensor([[[0., 2., 1., 1.], [0., 1., 3., 2.]]]).expand(2, -1, -1)
        self.assertTrue(torch.equal(model1.greedy_entries(obs, weights, actions, values),
                                    torch.tensor([[1, 2], [1, 2]])))

    def test_fixture_test_access_receipt_prevents_automatic_second_access(self):
        # This test uses one synthetic in-memory task, never the project test.
        doc = self.freeze()
        doc["final_test_ready"] = True
        row = {"kappa": 0., "method": "ma_usfa_ddqn", "seed": 100,
               "task_id": self.task["id"], "kind": "homo", "return": 1.}
        with patch.object(ev, "verify_frozen", return_value=doc), \
             patch.object(ev, "load_tasks", return_value=[self.task]), \
             patch.object(ev, "make_sources", return_value=self.sources.copy()), \
             patch.object(ev, "evaluate_sources_sf", return_value=np.empty(0)), \
             patch.object(ev, "_fixed_rows", return_value=[]), \
             patch.object(ev, "_evaluate_run", return_value=([row], {"optimizer_updates": 0})):
            result = ev.evaluate_frozen(self.frozen_path, "test", "fixture_test", True, self.root)
            self.assertEqual(result["status"], "PASS")
            with self.assertRaisesRegex(FileExistsError, "no silent retry"):
                ev.evaluate_frozen(self.frozen_path, "test", "second", True, self.root)
        self.assertTrue(self.frozen_path.with_name("frozen.test_access_started.json").exists())
        self.assertFalse((self.root / "second").exists())
        self.assertFalse((self.protocol / "test.sealed.json").exists())


if __name__ == "__main__":
    unittest.main()
