"""Seed, terminal-target and identity tests with small fake traffic envs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .config import REQUIRED_CONTRACT
from .source_policy import SourceCheckpoint
from .task_protocol_v2 import VERSION
from .train_sources_v2 import (TRAINING_VERSION, _dqn_target,
                                train_sources_v2)


class FakeEnv:
    possible_agents = ["a", "b"]

    def __init__(self, seed, terminal_after=None, fail_step=False,
                 missing_reward=False, inconsistent_actions=False):
        self.seed = seed
        self.reset_seeds = []
        self.closed = 0
        self.steps = 0
        self.actions = []
        self.reward_rows = []
        self.terminal_after = terminal_after
        self.fail_step = fail_step
        self.missing_reward = missing_reward
        self.action_spaces = {"a": SimpleNamespace(n=2),
                              "b": SimpleNamespace(n=3 if inconsistent_actions else 2)}

    def reset(self, seed):
        self.reset_seeds.append(seed)
        return {"a": np.array([0, 0, seed % 7], dtype=np.float32),
                "b": np.array([0, 1, seed % 7], dtype=np.float32)}, {}

    def step(self, actions):
        if self.fail_step:
            raise RuntimeError("fake environment step failed")
        self.steps += 1
        self.actions.append(actions.copy())
        rewards = {node: float(action + 1) for node, action in actions.items()}
        self.reward_rows.append(rewards.copy())
        if self.missing_reward:
            del rewards["b"]
        done = self.terminal_after is not None and self.steps >= self.terminal_after
        obs = {} if done else {
            node: np.array([self.steps, action, self.seed % 7], dtype=np.float32)
            for node, action in actions.items()}
        return obs, rewards, {node: done for node in self.possible_agents}, {
            node: False for node in self.possible_agents}, {}

    def close(self):
        self.closed += 1


class FakeGuard:
    def __init__(self):
        self.calls = 0

    def check(self):
        self.calls += 1


class SourceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.network = self.root / "network.xml"
        self.routes = self.root / "routes.xml"
        self.network.write_text('<net><tlLogic id="a"/><tlLogic id="b"/></net>')
        self.routes.write_text('<routes/>')
        anchors = [[-1, -0.3, -1.5, 3, 0], [-2, -0.3, -1.5, 3, 0]]
        self.protocol_path = self.root / "tasks.json"
        self.protocol_path.write_text(json.dumps({
            "version": VERSION, "node_order": ["a", "b"],
            "source_anchors": anchors,
            "source_sampling": {"temperature": 20.0, "resample": "each_step"},
            "tasks": {"train": [{"id": "train", "weights": anchors[0]}],
                      "validation": [{"id": "validation", "weights": [-1, -0.3, -1.5, 4, 0]}]},
            "traffic_seeds": {"train": [11, 12], "validation": [31], "test": [41]},
            "sealed_test_tasks": {"path": "must_stay_unopened.json", "sha256": "0" * 64},
        }))
        self.manifest_path = self.root / "manifest.json"
        manifest = {
            "version": "ma-usfa-tsc-server-manifest-v1", "status": "READY_FOR_PILOT",
            "scenario_id": "source-test", "source_anchors": anchors,
            "scenario": {"horizon_steps": 2, "delta_time": 10},
            "files": [{"role": "network", "path": "network.xml", "sha256": self.sha(self.network)},
                      {"role": "routes", "path": "routes.xml", "sha256": self.sha(self.routes)}],
            "method_contract": {key: True for key in REQUIRED_CONTRACT},
            "protocol": {"protocol_id": "v2-unit-test", "code_sha256": "x",
                         "data_manifest_sha256": "x", "train_split_sha256": "x",
                         "validation_split_sha256": "y", "test_split_sha256": "z",
                         "training_seeds": [0], "traffic_seeds": [11]},
            "resource_limits": {"threads": 1, "max_wall_seconds": 30,
                                "max_rss_gib": 1, "min_free_disk_gib": 1},
            "test_accessed_before_formal_run": False,
            "traffic_task_protocol": {"version": VERSION, "path": "tasks.json",
                                      "sha256": self.sha(self.protocol_path)},
            "source_training": {"hidden": 8, "target_every_episodes": 2},
        }
        self.manifest_path.write_text(json.dumps(manifest))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_training(self, out, **env_options):
        envs = []

        def builder(_spec, reward_fn, **kwargs):
            self.assertTrue(callable(reward_fn))
            env = FakeEnv(kwargs["sumo_seed"], **env_options)
            envs.append(env)
            return env

        guard = FakeGuard()
        with patch("tsc_repaired.train_sources_v2.ScenarioSpec.build_parallel_env", new=builder):
            report = train_sources_v2(self.manifest_path, out, seed=7,
                                      episodes=2, device="cpu", guard=guard)
        return report, envs, guard

    def test_fresh_seeded_envs_action_rewards_and_checkpoint_identity(self):
        output = self.root / "first"
        result, envs, guard = self.run_training(output)
        self.assertEqual(result["source_count"], 2)
        self.assertEqual([env.seed for env in envs], [12, 11, 11, 12])
        self.assertTrue(all(env.reset_seeds == [env.seed] for env in envs))
        self.assertTrue(all(env.closed == 1 and env.steps == 2 for env in envs))
        self.assertGreaterEqual(guard.calls, 16)
        self.assertFalse(result["test_objectives_loaded"])
        self.assertFalse(result["validation_objectives_used_for_training"])
        self.assertFalse((self.root / "must_stay_unopened.json").exists())
        for index, row in enumerate(result["sources"]):
            path = output / f"source_{index:03d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["source_index"], index)
            self.assertEqual(payload["seed"], 7 + index)
            self.assertEqual(payload["updates"], 4)
            self.assertEqual(payload["training_version"], TRAINING_VERSION)
            self.assertEqual(payload["source_training_traffic_seeds"], [12, 11] if index == 0 else [11, 12])
            self.assertEqual(payload["algorithm"], "one_step_per_node_dqn")
            self.assertEqual(payload["target_copies"], 1)
            self.assertEqual(payload["target_copy_unit"], "episodes")
            self.assertEqual(payload["scenario_hashes"], {
                "network": self.sha(self.network), "routes": self.sha(self.routes)})
            self.assertEqual(payload["scenario_settings"], {
                "delta_time": 10, "horizon_steps": 2, "yellow_time": 2,
                "min_green": 5, "max_green": 50, "scale": 2.0})
            self.assertEqual(row["scenario_settings"], payload["scenario_settings"])
            self.assertEqual(result["scenario_settings"], payload["scenario_settings"])
            self.assertEqual(payload["protocol_fingerprint"], self.sha(self.protocol_path))
            self.assertEqual(payload["node_ids"], ["a", "b"])
            self.assertEqual(row["checkpoint_sha256"], self.sha(path))
            actual_returns = [sum(np.mean(list(reward.values())) for reward in env.reward_rows)
                              for env in envs[index * 2: index * 2 + 2]]
            self.assertAlmostEqual(row["mean_training_episode_mean_agent_return"], np.mean(actual_returns))
            model = SourceCheckpoint(path, index, 3, 2, 8).load("cpu")
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        before = {path.name: path.read_bytes() for path in output.glob("source_*.pt")}
        with self.assertRaisesRegex(FileExistsError, "already contains"):
            self.run_training(output)
        for name, data in before.items():
            self.assertEqual((output / name).read_bytes(), data)

    def test_seed_reproduces_actions_and_source_parameters(self):
        first, first_envs, _ = self.run_training(self.root / "a")
        second, second_envs, _ = self.run_training(self.root / "b")
        self.assertEqual([env.actions for env in first_envs], [env.actions for env in second_envs])
        for index in range(2):
            a = torch.load(first["sources"][index]["checkpoint"], weights_only=False)
            b = torch.load(second["sources"][index]["checkpoint"], weights_only=False)
            for name, value in a["state_dict"].items():
                self.assertTrue(torch.equal(value, b["state_dict"][name]))

    def test_environment_closes_on_step_exception_and_missing_reward(self):
        for option in ({"fail_step": True}, {"missing_reward": True}, {"inconsistent_actions": True}):
            envs = []

            def builder(_spec, reward_fn, **kwargs):
                env = FakeEnv(kwargs["sumo_seed"], **option)
                envs.append(env)
                return env

            with patch("tsc_repaired.train_sources_v2.ScenarioSpec.build_parallel_env", new=builder):
                with self.assertRaises((RuntimeError, ValueError)):
                    train_sources_v2(self.manifest_path, self.root / str(len(option)) / next(iter(option)),
                                      episodes=1)
            self.assertEqual(len(envs), 1)
            self.assertEqual(envs[0].closed, 1)

    def test_terminal_target_never_bootstraps_and_dqn_preserves_node_rewards(self):
        class Exploding:
            def __call__(self, *args):
                raise AssertionError("terminal bootstrap was evaluated")

        state = torch.zeros((1, 2, 3))
        time = torch.zeros(1)
        target = _dqn_target(Exploding(), state, time, [2., 3.], True, .99)
        self.assertEqual(target.tolist(), [2., 3.])

        class ConstantQ:
            def __init__(self, values):
                self.values = torch.tensor([values], dtype=torch.float32)
            def __call__(self, *args):
                return self.values

        target = _dqn_target(ConstantQ([[10, 20], [30, 40]]), state, time,
                             [2., 3.], False, .5)
        self.assertEqual(target.tolist(), [12., 23.])

    def test_horizon_cap_is_terminal_even_if_environment_never_signals_done(self):
        seen = []
        original_target = _dqn_target

        def record_target(*args, **kwargs):
            seen.append(bool(args[4]))
            return original_target(*args, **kwargs)

        with patch("tsc_repaired.train_sources_v2._dqn_target", side_effect=record_target):
            _, envs, _ = self.run_training(self.root / "cap")
        self.assertEqual(seen, [False, True] * 4)
        self.assertTrue(all(env.steps == 2 for env in envs))


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SourceTrainingTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
