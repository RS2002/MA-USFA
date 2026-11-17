"""Small synthetic contract tests; no paper data or sealed objectives are read."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from .evidence_shared_baseline import (
    SharedConfig, SharedJointDDQN, double_dqn_targets,
    evaluate_shared_baseline, load_shared_baseline, shared_ddqn_update,
    train_shared_baseline, main,
)
from .exact import evaluate_policy
from .learning import model_hash
from .legacy_world import SFWorldExact


def tasks(prefix, offset):
    return [{"id": f"{prefix}_{i}", "kind": "unit",
             "weights": [[1. + offset + .1 * i, .2, .3, .4],
                         [.3, 1. + offset, .2, .1]]} for i in range(2)]


class FixedScores(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.calls = 0

    def forward(self, obs, weights):
        self.calls += 1
        return self.scores[None].expand(len(obs), -1)


class SharedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_task_input_and_all_primitive_actions(self):
        model = SharedJointDDQN()
        self.assertEqual(model.spec()["parameters"], 6681)
        self.assertFalse(model.spec()["source_inputs"])
        self.assertFalse(model.spec()["sf_inputs"])
        obs = torch.zeros(2, 5)
        weights = torch.zeros(2, 2, 4)
        weights[1, 1, 3] = 1.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # Make output action 24 depend only on agent 1's final task weight.
            model.net[0].weight[0, 12] = 1.
            model.net[2].weight[0, 0] = 1.
            model.net[4].weight[24, 0] = 1.
        q = model(obs, weights)
        self.assertEqual(tuple(q.shape), (2, 25))
        self.assertEqual(q.argmax(-1).tolist(), [0, 24])
        world = SFWorldExact(horizon=2)
        for action in range(25):
            nxt, feature, _ = world.step(24, action, rng=np.random.RandomState(action))
            self.assertTrue(0 <= nxt < world.S)
            self.assertEqual(feature.shape, (2, 5))

    def test_ddqn_targets_use_online_argmax_and_ignore_terminal(self):
        online_scores = np.zeros(25); online_scores[7] = 3.
        target_scores = np.zeros(25); target_scores[7] = 4.; target_scores[2] = 99.
        online, target = FixedScores(online_scores), FixedScores(target_scores)
        batch = {"reward": np.array([1., 2.]), "done": np.array([0., 1.]),
                 "time": np.array([0, 1]), "next_state": np.array([6, 999999]),
                 "weights": np.zeros((2, 2, 4), dtype=np.float32)}
        y = double_dqn_targets(online, target, batch, gamma=.5, horizon=2)
        self.assertTrue(torch.equal(y, torch.tensor([3., 2.])))
        terminal = {k: v[1:] for k, v in batch.items()}
        y_terminal = double_dqn_targets(online, target, terminal, gamma=.5, horizon=2)
        self.assertTrue(torch.equal(y_terminal, torch.tensor([2.])))
        self.assertEqual(online.calls, 1)
        self.assertEqual(target.calls, 1)

    def test_real_update_changes_online_and_preserves_target(self):
        torch.manual_seed(1)
        model, target = SharedJointDDQN(8, 2), SharedJointDDQN(8, 2)
        target.load_state_dict(model.state_dict())
        before, target_before = model_hash(model), model_hash(target)
        batch = {"state": np.array([0, 3, 19, 24]), "time": np.array([0, 1, 0, 1]),
                 "next_state": np.array([1, 2, 20, 25]), "action": np.array([0, 4, 20, 24]),
                 "reward": np.array([1., 2., 3., 4.]), "done": np.array([0., 1., 0., 1.]),
                 "weights": np.ones((4, 2, 4), dtype=np.float32)}
        cfg = SharedConfig(budget=4, horizon=2, hidden=8, eval_every=2)
        loss = shared_ddqn_update(model, target, batch, torch.optim.Adam(model.parameters(), lr=.01), cfg)
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(before, model_hash(model))
        self.assertEqual(target_before, model_hash(target))

    def test_exact_evaluation_is_frozen_and_executes_selected_action(self):
        model = SharedJointDDQN(8, horizon=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[24] = 2.
        model.train()
        before = model_hash(model)
        world = SFWorldExact(horizon=2, kappa=.5)
        task = tasks("validation", .7)[0]
        result = evaluate_shared_baseline(model, world, [task])
        direct = evaluate_policy(world, np.full((2, world.S), 24), np.asarray(task["weights"], dtype=np.float32))
        self.assertAlmostEqual(result["mean_return"], direct["initial_return"], places=12)
        self.assertEqual(model_hash(model), before)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertTrue(model.training)

    def test_small_runs_are_deterministic_and_preserve_best_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"MA_USFA_THREAD_BUDGET": "1"}):
            outputs = []
            for index in range(2):
                out = Path(temporary) / str(index)
                model, summary = train_shared_baseline(out, SFWorldExact(horizon=2, kappa=.5),
                    tasks("train", .0), tasks("validation", .7), seed=4, budget=16,
                    hidden=8, batch=4, warmup=4, update_every=2, target_every=2,
                    eval_every=8, replay_capacity=32)
                outputs.append((model_hash(model), summary))
                self.assertTrue((out / "best.pt").exists())
                self.assertTrue((out / "endpoint.pt").exists())
                self.assertEqual(summary["steps"], 16)
                self.assertEqual(summary["updates"], 7)
                self.assertEqual(len(summary["logs"]), 2)
                self.assertFalse(summary["test_objectives_loaded"])
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                best, best_payload = load_shared_baseline(out / "best.pt")
                endpoint, endpoint_payload = load_shared_baseline(out / "endpoint.pt")
                self.assertEqual(best_payload["selection"], "best validation mean over all objectives")
                self.assertEqual(endpoint_payload["selection"], "fixed-budget endpoint")
                self.assertEqual(endpoint_payload["steps"], 16)
                self.assertEqual(best_payload["steps"], summary["best_step"])
                frozen = evaluate_shared_baseline(best, SFWorldExact(horizon=2, kappa=.5), tasks("validation", .7))
                self.assertAlmostEqual(frozen["mean_return"], summary["best_validation"], places=12)
                with self.assertRaises(FileExistsError):
                    train_shared_baseline(out, SFWorldExact(horizon=2, kappa=.5),
                        tasks("train", .0), tasks("validation", .7), budget=16,
                        hidden=8, eval_every=8)
                corrupted = copy.deepcopy(best_payload)
                corrupted["model"]["net.0.weight"][0, 0] += .1
                torch.save(corrupted, out / "corrupted.pt")
                with self.assertRaisesRegex(ValueError, "parameter hash"):
                    load_shared_baseline(out / "corrupted.pt")
            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1]["best_validation"], outputs[1][1]["best_validation"])
            self.assertEqual(outputs[0][1]["endpoint_validation"], outputs[1][1]["endpoint_validation"])

    def test_guard_and_split_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"MA_USFA_THREAD_BUDGET": "0"}):
                with self.assertRaisesRegex(RuntimeError, "resource guardian"):
                    train_shared_baseline(Path(temporary) / "guard", SFWorldExact(horizon=2),
                                          tasks("train", 0.), tasks("validation", .7))
            with mock.patch.dict(os.environ, {"MA_USFA_THREAD_BUDGET": "1"}):
                with self.assertRaisesRegex(ValueError, "identical objectives"):
                    train_shared_baseline(Path(temporary) / "overlap", SFWorldExact(horizon=2),
                                          tasks("train", 0.), tasks("validation", 0.))

    def test_longer_budget_has_its_own_full_validation_schedule(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"MA_USFA_THREAD_BUDGET": "1"}):
            _, summary = train_shared_baseline(Path(temporary) / "longer", SFWorldExact(horizon=2, kappa=.5),
                tasks("train", .0), tasks("validation", .7), seed=4, budget=32,
                hidden=8, batch=4, warmup=4, update_every=2, target_every=2,
                eval_every=8, replay_capacity=32)
            self.assertEqual([row["steps"] for row in summary["logs"]], [8, 16, 24, 32])
            self.assertEqual(summary["config"]["budget"], 32)
            self.assertEqual(summary["steps"], 32)
            self.assertEqual(summary["logs"][0]["epsilon"], max(.02, .2 * (1. - 7 / (32 * .8))))
        cfg = SharedConfig(budget=163840)
        cfg.validate()
        self.assertEqual(cfg.budget // cfg.eval_every, 64)

    def test_interrupted_cli_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary)
            for name in ("manifest.json", "train.json", "validation.json"):
                (protocol / name).write_text("{}\n")
            interrupted = {"status": "INTERRUPTED", "steps": 40, "updates": 0, "complete": False}
            argv = ["evidence_shared_baseline", "--protocol-dir", temporary,
                    "--output-dir", str(protocol / "output"), "--kappa", "1"]
            with mock.patch("sys.argv", argv), mock.patch(
                "repaired.evidence_shared_baseline.load_tasks",
                side_effect=[tasks("train", 0.) * 32, tasks("validation", .7) * 8]), mock.patch(
                "repaired.evidence_shared_baseline.train_shared_baseline",
                return_value=(None, interrupted)):
                self.assertEqual(main(), 2)


if __name__ == "__main__":
    unittest.main()
