"""Action-to-return regression tests for closed-loop traffic evaluation.

The fake environment rewards the actions actually executed, so these tests
would fail if evaluation merely reused fixed behavior-data rewards. No SUMO
process or optimizer is started.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import rollout_evaluate as evaluation
from .config import sha256_file


class FakeSelector(nn.Module):
    def __init__(self, selected=0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(selected)))

    def choose_entries(self, obs, weights, candidate_actions, sf_values, time_fraction):
        assert not self.training
        assert not self.bias.requires_grad
        assert not torch.is_grad_enabled()
        return torch.full(obs.shape[:2], int(self.bias), dtype=torch.long, device=obs.device)


class FakeProtocol:
    fingerprint = "frozen-protocol-fingerprint"

    def tasks(self, split):
        if split != "validation":
            raise AssertionError("unexpected test/train task access")
        return [{"id": "task/A", "weights": np.ones((2, 5), np.float32)},
                {"id": "task/B", "weights": 2 * np.ones((2, 5), np.float32)}]

    def traffic_seeds(self, split):
        if split != "validation":
            raise AssertionError("unexpected test/train seed access")
        return [10000, 10001]


class FakeRuntime:
    instances = []
    source_count = 2
    fail_on_step = False

    def __init__(self, manifest_path, source_dir, sf_checkpoint=None, device="cpu"):
        self.ids = ("node_a", "node_b")
        self.N, self.K, self.observation_dim, self.action_count, self.horizon = 2, self.source_count, 3, 2, 3
        self.weights = np.ones((self.N, 5), np.float32)
        self.adjacency = np.ones((2, 2), dtype=bool)
        self.source_hashes = {path.name: sha256_file(path) for path in sorted(Path(source_dir).glob("source_*.pt"))}
        self.sf_payload = {"config": {"gamma": .5}}
        self.sf = nn.Linear(1, 1)
        self.library = SimpleNamespace(models=(nn.Linear(1, 1),))
        self.reset_calls, self.step_calls, self.close_calls = [], 0, 0
        self.in_episode = False
        self.instances.append(self)

    def reset(self, task, traffic_seed, tripinfo_path=None):
        assert not self.in_episode
        self.weights = np.asarray(task["weights"], np.float32)
        self.current_step, self.traffic_seed = 0, traffic_seed
        self.in_episode = True
        self.tripinfo = Path(tripinfo_path)
        self.reset_calls.append((task["id"], traffic_seed, self.tripinfo))
        return np.zeros((2, 3), np.float32)

    def candidate_actions(self, state, step):
        return np.broadcast_to(np.arange(self.K), (2, self.K)).copy()

    def sf_values(self, state, step):
        return np.broadcast_to(np.arange(self.K, dtype=np.float32), (2, self.K)).copy()

    def step(self, entries):
        if not self.in_episode:
            raise AssertionError("step after terminal/close")
        self.current_step += 1
        self.step_calls += 1
        if self.fail_on_step:
            raise RuntimeError("fake rollout failure")
        actions = np.asarray(entries, np.int64)
        rewards = (actions + 1) * np.arange(1, 3)
        done = self.current_step == 2
        if done:
            self.in_episode = False
        return SimpleNamespace(
            state=np.full((2, 3), self.current_step, np.float32),
            actions=actions, rewards=rewards, done=done, info={},
            features=np.zeros((2, 5), np.float32))

    def close(self):
        self.close_calls += 1
        self.in_episode = False
        # Assert parsing must happen only after close writes a valid file.
        if hasattr(self, "tripinfo") and not self.tripinfo.is_file():
            self.tripinfo.parent.mkdir(parents=True, exist_ok=True)
            self.tripinfo.write_text(
                '<tripinfos><tripinfo arrival="20" duration="20" waitingTime="4" timeLoss="5"/>'
                '<tripinfo arrival="-1" duration="15" waitingTime="99" timeLoss="99"/></tripinfos>')

    def audit_frozen(self):
        return {"status": "PASS", "membership_failures": 0}


class ClosedLoopEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        for index in range(2):
            (self.sources / f"source_{index}.pt").write_bytes(f"source{index}".encode())
        self.sf = self.root / "sf.pt"
        self.sf.write_bytes(b"SF checkpoint")
        self.checkpoint = self.root / "upper.pt"
        self.checkpoint.write_bytes(b"upper checkpoint")
        self.protocol = FakeProtocol()
        self.manifest = SimpleNamespace(require_stage=lambda stage: None)
        FakeRuntime.instances.clear()
        FakeRuntime.source_count = 2
        FakeRuntime.fail_on_step = False
        self.model = FakeSelector(0)

    def payload(self):
        return {
            "version": evaluation.SELECTOR_VERSION,
            "metadata": {
                "source_hashes": {p.name: sha256_file(p) for p in sorted(self.sources.glob("source_*.pt"))},
                "sf_sha256": sha256_file(self.sf),
                "protocol_fingerprint": self.protocol.fingerprint,
                "node_ids": ["node_a", "node_b"],
                "training_split": "train",
            },
            "config": {"observation_dim": 3, "source_count": FakeRuntime.source_count,
                       "action_count": 2, "task_dim": 5, "horizon": 3, "gamma": .5},
            "adjacency": np.ones((2, 2), bool),
        }

    def run_evaluation(self, name="result.json", **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, "FrozenTrafficRuntime", FakeRuntime))
            stack.enter_context(patch.object(evaluation, "load_manifest", return_value=self.manifest))
            stack.enter_context(patch.object(evaluation, "load_protocol", return_value=self.protocol))
            stack.enter_context(patch.object(evaluation, "load_selector", return_value=(self.model, self.payload())))
            return evaluation.evaluate_rollouts("manifest.json", self.sources, self.sf,
                                                self.checkpoint, self.root / name, **kwargs)

    def test_actual_actions_change_actual_returns(self):
        first = self.run_evaluation("first.json", methods=["ma_usfa_ddqn"])
        self.model = FakeSelector(1)
        second = self.run_evaluation("second.json", methods=["ma_usfa_ddqn"])
        self.assertEqual(first["status"], "PASS", first["errors"])
        self.assertEqual(second["status"], "PASS", second["errors"])
        for report, expected in ((first, 6.0), (second, 12.0)):
            self.assertTrue(report["policy_performance_evaluated"])
            self.assertEqual(report["evaluation_updates"], 0)
            self.assertEqual(report["test_updates"], 0)
            self.assertEqual(report["completed_episode_count"], 2)
            for row in report["episodes"]:
                self.assertEqual(row["steps"], 2)
                self.assertEqual(row["episode_team_return"], expected)
                self.assertEqual(row["episode_mean_agent_return"], expected / 2)
                self.assertEqual(row["mean_step_reward"], expected / 4)
                self.assertEqual(row["discounted_team_return"], expected * .75)
                self.assertIsNone(row["mean_queue_per_node"])
                self.assertIsNone(row["mean_speed_m_per_second"])
                self.assertEqual(row["tripinfo"]["completed_trip_count"], 1)
                self.assertEqual(row["tripinfo"]["unfinished_trip_count"], 1)
                self.assertEqual(row["tripinfo"]["mean_completed_duration_seconds"], 20)
                self.assertNotIn("selected_entries", row)
            self.assertTrue(report["checkpoint_files_unchanged"])
            self.assertTrue(report["selector_parameters_unchanged"])
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.bias.requires_grad)
        self.assertIsNone(self.model.bias.grad)
        self.assertTrue(all(runtime.step_calls == 4 for runtime in FakeRuntime.instances))

    def test_all_methods_are_paired_and_every_fixed_source_reported(self):
        report = self.run_evaluation(episodes_per_task=2)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["expected_episode_count"], 20)
        self.assertEqual(set(report["summary"]), {"ma_usfa_ddqn", "sync_entry", "independent_entry", "fixed_source_0", "fixed_source_1"})
        pairs = {}
        paths = []
        for row in report["episodes"]:
            pairs.setdefault(row["method"], set()).add((row["task_id"], row["traffic_seed"]))
            paths.append(row["tripinfo_file"])
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(pairset == pairs["ma_usfa_ddqn"] for pairset in pairs.values()))
        self.assertEqual(pairs["ma_usfa_ddqn"], {("task/A", 10000), ("task/A", 10001), ("task/B", 10000), ("task/B", 10001)})

    def test_k1_all_controllers_match_the_only_source(self):
        FakeRuntime.source_count = 1
        (self.sources / "source_1.pt").unlink()
        report = self.run_evaluation()
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertTrue(all(row["episode_team_return"] == 6 for row in report["episodes"]))
        self.assertTrue(all(row["source_selection_histogram"] == {"0": 4} for row in report["episodes"]))

    def test_failure_is_explicit_and_does_not_produce_subset_summary(self):
        FakeRuntime.fail_on_step = True
        report = self.run_evaluation()
        self.assertEqual(report["status"], "NOT_READY")
        self.assertFalse(report["policy_performance_evaluated"])
        self.assertIsNone(report["summary"])
        self.assertIn("fake rollout failure", report["errors"][0])
        self.assertEqual(report["episodes"][0]["status"], "NOT_READY")
        self.assertGreater(FakeRuntime.instances[0].close_calls, 0)

    def test_checkpoint_mutation_invalidates_completed_rollouts(self):
        original_step = FakeRuntime.step
        checkpoint = self.checkpoint

        def mutate(runtime, entries):
            transition = original_step(runtime, entries)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"unexpected mutation")
            return transition

        with patch.object(FakeRuntime, "step", mutate):
            report = self.run_evaluation(methods=["ma_usfa_ddqn"])
        self.assertEqual(report["status"], "NOT_READY")
        self.assertIsNone(report["summary"])
        self.assertFalse(report["policy_performance_evaluated"])
        self.assertTrue(any("checkpoint file changed" in error for error in report["errors"]))

    def test_nonpersistent_graph_buffer_mutation_is_detected(self):
        class MutatingSelector(FakeSelector):
            def __init__(self):
                super().__init__()
                self.register_buffer("graph_counter", torch.zeros(1), persistent=False)

            def choose_entries(self, *args):
                result = super().choose_entries(*args)
                self.graph_counter.add_(1)
                return result

        self.model = MutatingSelector()
        report = self.run_evaluation(methods=["ma_usfa_ddqn"])
        self.assertEqual(report["status"], "NOT_READY")
        self.assertIsNone(report["summary"])
        self.assertTrue(any("buffers changed" in error for error in report["errors"]))

    def test_graph_identity_mismatch_is_rejected_before_rollout(self):
        original_payload = self.payload

        def wrong_graph():
            payload = original_payload()
            payload["adjacency"] = np.eye(2, dtype=bool)
            return payload

        with patch.object(self, "payload", side_effect=wrong_graph):
            report = self.run_evaluation()
        self.assertEqual(report["status"], "NOT_READY")
        self.assertFalse(FakeRuntime.instances[0].reset_calls)
        self.assertTrue(any("adjacency differs" in error for error in report["errors"]))

    def test_insufficient_declared_seeds_rejected_without_a_rollout(self):
        report = self.run_evaluation(episodes_per_task=3)
        self.assertEqual(report["status"], "NOT_READY")
        self.assertFalse(FakeRuntime.instances[0].reset_calls)
        self.assertIsNone(report["summary"])

    def test_saved_report_is_not_overwritten(self):
        path = self.root / "kept.json"
        path.write_text("original evidence")
        report = self.run_evaluation("kept.json")
        self.assertEqual(report["status"], "NOT_READY")
        self.assertEqual(path.read_text(), "original evidence")
        self.assertFalse(FakeRuntime.instances)

    def test_missing_tripinfo_and_missing_attributes_are_not_zero(self):
        missing = evaluation.parse_tripinfo(self.root / "missing.xml")
        self.assertIsNone(missing["completed_trip_count"])
        self.assertIsNone(missing["mean_completed_duration_seconds"])
        empty = self.root / "empty.xml"
        empty.write_text("<tripinfos/>")
        parsed = evaluation.parse_tripinfo(empty)
        self.assertEqual(parsed["completed_trip_count"], 0)
        self.assertIsNone(parsed["mean_completed_duration_seconds"])


def main() -> int:
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClosedLoopEvaluationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
