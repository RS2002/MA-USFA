"""Fake-environment checks for the legacy-aligned evaluator."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mixer_ablation.selector import NORMALIZED_VERSION
from tsc_repaired.config import sha256_file
from tsc_repaired.test_rollout_evaluate import FakeRuntime as BaseRuntime
from tsc_repaired.test_rollout_evaluate import FakeSelector
from . import evaluate


class LegacyProtocol:
    fingerprint = "legacy-paper-protocol"

    def tasks(self, split):
        if split != "validation":
            raise AssertionError("legacy evaluator accessed non-validation tasks")
        return [
            {"id": "legacy_homo", "weights": np.ones((2, 5), np.float32)},
            {"id": "legacy_hetero", "weights": np.asarray(
                [[1, 1, 1, 1, 0], [2, 2, 2, 2, 0]], np.float32)},
        ]

    def traffic_seeds(self, split):
        if split != "validation":
            raise AssertionError("legacy evaluator accessed non-validation seeds")
        return (10_000_000,)


class LegacyRuntime(BaseRuntime):
    instances = []
    source_count = 2

    def __init__(self, manifest_path, source_dir, sf_checkpoint=None, device="cpu"):
        super().__init__(manifest_path, source_dir, sf_checkpoint, device)
        self.source_scenario_binding_verified = False


class PaperAlignedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text("{}")
        self.sources = self.root / "sources"
        self.sources.mkdir()
        for index in range(2):
            (self.sources / f"source_{index}.pt").write_bytes(f"source{index}".encode())
        self.sf = self.root / "task_sf.pt"
        self.sf.write_bytes(b"legacy sf")
        self.checkpoints = {"normalized_legacy": self.root / "normalized_legacy.pt"}
        self.checkpoints["normalized_legacy"].write_bytes(b"legacy upper")
        self.protocol = LegacyProtocol()
        self.manifest = SimpleNamespace(
            raw={"paper_alignment": {"task_seed": 11}},
            require_stage=lambda stage: self.assertEqual(stage, "pilot"),
        )
        self.model = FakeSelector(0)
        LegacyRuntime.instances.clear()

    def payload(self):
        return {
            "version": NORMALIZED_VERSION,
            "metadata": {
                "ablation_variant": "normalized",
                "initial_parameter_sha256": "a" * 64,
                "training_configuration": {"batch_size": 16, "epsilon_decay_transitions": 480},
                "seed": 100, "episodes": 200, "transitions": 12_000,
                "optimizer_updates": 11_985,
                "checkpoint_selection": "predeclared legacy-aligned budget; no validation selection",
                "source_hashes": {p.name: sha256_file(p) for p in sorted(self.sources.glob("source_*.pt"))},
                "sf_sha256": sha256_file(self.sf), "protocol_fingerprint": self.protocol.fingerprint,
                "node_ids": ["node_a", "node_b"], "training_split": "train",
                "test_accessed": False, "test_updates": 0,
            },
            "config": {"observation_dim": 3, "source_count": 2, "action_count": 2,
                       "task_dim": 5, "horizon": 3, "gamma": .5},
            "adjacency": np.ones((2, 2), bool),
        }

    def run_eval(self, name="result.json", **kwargs):
        def load(path, device="cpu"):
            return self.model, self.payload()
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluate, "FrozenTrafficRuntime", LegacyRuntime))
            stack.enter_context(patch.object(evaluate, "load_manifest", return_value=self.manifest))
            stack.enter_context(patch.object(evaluate, "load_protocol", return_value=self.protocol))
            stack.enter_context(patch.object(evaluate, "load_ablation_selector", side_effect=load))
            stack.enter_context(patch.object(torch.optim, "Adam",
                                              side_effect=AssertionError("optimizer started")))
            return evaluate.evaluate_paper_aligned(
                self.manifest_path, self.sources, self.sf, self.checkpoints,
                self.root / name, **kwargs)

    def test_two_legacy_tasks_and_five_same_seed_replays_are_paired(self):
        report = self.run_eval()
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["paper_alignment"], "legacy_homo_hetero_task_seed_11")
        self.assertEqual(report["traffic_seed"], 10_000_000)
        self.assertEqual(report["replays"], 5)
        self.assertEqual(report["expected_episode_count"], 50)
        self.assertEqual(report["completed_episode_count"], 50)
        self.assertEqual(len(report["paired_differences"]), 5 * 2 * 4)
        self.assertTrue(all(row["traffic_seed"] == 10_000_000 for row in report["episodes"]))
        self.assertEqual({row["task_id"] for row in report["episodes"]},
                         {"legacy_homo", "legacy_hetero"})
        self.assertEqual(len(report["episode_evidence_sha256"]), 50)
        self.assertEqual(report["unavailable_episode_evidence"], [])

    def test_rejects_nonlegacy_manifest_or_wrong_seed_before_rollout(self):
        self.manifest.raw = {"paper_alignment": {"task_seed": 7}}
        report = self.run_eval("wrong_manifest.json")
        self.assertEqual(report["status"], "NOT_READY")
        self.assertFalse(LegacyRuntime.instances)
        self.manifest.raw = {"paper_alignment": {"task_seed": 11}}
        report = self.run_eval("wrong_seed.json", traffic_seed=999)
        self.assertEqual(report["status"], "NOT_READY")
        self.assertFalse(LegacyRuntime.instances[-1].reset_calls)

    def test_partial_rollout_never_reports_subset_summary(self):
        original = LegacyRuntime.step

        def fail_after_first(runtime, entries):
            if len(runtime.reset_calls) > 1:
                raise RuntimeError("legacy deliberate failure")
            return original(runtime, entries)

        with patch.object(LegacyRuntime, "step", fail_after_first):
            report = self.run_eval("partial.json")
        self.assertEqual(report["status"], "NOT_READY")
        self.assertIsNone(report["summary"])
        self.assertIsNone(report["paired_differences"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
