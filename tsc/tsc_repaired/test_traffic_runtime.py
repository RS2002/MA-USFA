"""Action-sensitive runtime tests; no SUMO process or training is started."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import traffic_runtime as runtime_module
from .config import sha256_file
from .features import FEATURE_NAMES
from .scenario import ScenarioFiles, ScenarioSpec
from .source_policy import SourceQNet
from .traffic_runtime import FrozenTrafficRuntime


class FakeSignal:
    def __init__(self, env, node):
        self.id = node
        self.action = 0
        self.lanes = [node + "_in"]
        self.out_lanes = [node + "_out"]
        self.delta_time = 10
        lane = SimpleNamespace(getWaitingTime=lambda lane: 0.,
                               getLastStepVehicleNumber=lambda lane: 0)
        self.sumo = SimpleNamespace(lane=lane, simulation=SimpleNamespace(
            getTime=lambda: float(env.calls * 10)))

    def get_total_queued(self):
        return 0.

    def get_average_speed(self):
        # The feature extractor divides by MAX_SPEED=15.
        return 15. * self.action


class FakeEnv:
    def __init__(self, reward_fn, ids, action_count, seed, tripinfo_path=None):
        self.reward_fn = reward_fn
        self.possible_agents = list(reversed(ids))
        self.ids = ids
        self.action_spaces = {node: SimpleNamespace(n=action_count) for node in ids}
        self.sumo_seed = seed
        self.tripinfo_path = tripinfo_path
        self.calls = 0
        self.closed = False
        self.omit_reward = False
        self.omit_observation = False
        self.empty_terminal = False
        self.terminate_at = None
        self.last_actions = None
        self.signals = {node: FakeSignal(self, node) for node in ids}

    def reset(self, seed):
        self.reset_seed = seed
        self.calls = 0
        return {node: np.zeros(3, dtype=np.float32) for node in self.ids}, {}

    def step(self, actions):
        if self.closed:
            raise AssertionError("closed environment was stepped")
        self.calls += 1
        self.last_actions = actions.copy()
        rewards = {}
        for node in self.ids:
            self.signals[node].action = actions[node]
            rewards[node] = self.reward_fn(self.signals[node])
        observations = {node: np.full(3, actions[node], dtype=np.float32)
                        for node in self.ids}
        done = self.terminate_at is not None and self.calls >= self.terminate_at
        if self.omit_reward:
            rewards.pop(self.ids[0])
        if self.omit_observation:
            observations.pop(self.ids[0])
        if done and self.empty_terminal:
            observations = {}
        return (observations, rewards, {node: done for node in self.ids},
                {node: False for node in self.ids}, {})

    def close(self):
        self.closed = True


class FakeTaskSF(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.register_buffer("identity", torch.ones(1))
        self.config = SimpleNamespace(state_dim=3, source_count=k,
                                      task_dim=5, feature_dim=5)

    def candidate_values(self, state, weights, times, adjacency=None):
        return weights[..., 3:4].expand(-1, -1, self.config.source_count)


class FrozenTrafficRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.network = self.root / "network.xml"
        self.network.write_text(
            '<net><tlLogic id="a"/><tlLogic id="b"/>'
            '<edge id="ia" from="x" to="a"/>'
            '<edge id="ab" from="a" to="b"/>'
            '<edge id="ob" from="b" to="y"/>'
            '<connection tl="a" from="ia" to="ab"/>'
            '<connection tl="b" from="ab" to="ob"/></net>')
        self.routes = self.root / "routes.xml"
        self.routes.write_text('<routes/>')
        self.protocol_path = self.root / "task_protocol.json"
        self.protocol_path.write_text('{"immutable":"protocol fixture"}')
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text('{}')
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.ids = ("a", "b")
        self.task = np.array([0., 0., 0., 1., 0.], dtype=np.float32)
        self.envs = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runtime_module, 'load_manifest',
            return_value=SimpleNamespace(path=self.manifest, require_stage=lambda stage: None)))
        self.spec = ScenarioSpec(ScenarioFiles(self.network, self.routes), horizon_steps=2)
        self.stack.enter_context(patch.object(ScenarioSpec, 'from_manifest', return_value=self.spec))
        case = self

        def builder(spec, reward_fn, tripinfo_path=None, use_gui=False, sumo_seed=None):
            env = FakeEnv(reward_fn, case.ids, 2, sumo_seed, tripinfo_path)
            case.envs.append(env)
            return env

        self.stack.enter_context(patch.object(ScenarioSpec, 'build_parallel_env', new=builder))

    def make(self, k=2, sf=False, source_settings=None, source_file_binding=False):
        anchors = np.stack([self.task * (index + 1) for index in range(k)])
        for index in range(k):
            model = SourceQNet(3, 2, hidden=8)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.net[-1].bias[min(index, 1)] = 1.
            metadata = {"source_index": index, "observation_dim": 3, "action_count": 2,
                        "hidden": 8, "weights": anchors[index].tolist(), "episodes": 5,
                        "state_dict": model.state_dict()}
            if source_settings is not None:
                metadata["scenario_settings"] = source_settings
            if source_file_binding:
                metadata["scenario_hashes"] = self.spec.files.hashes()
            torch.save(metadata, self.sources / f"source_{index:03d}.pt")
        protocol = SimpleNamespace(anchors=anchors, path=self.protocol_path,
                                   fingerprint=sha256_file(self.protocol_path))
        self.stack.enter_context(patch.object(runtime_module, 'load_protocol', return_value=protocol))
        sf_file = None
        if sf:
            sf_file = self.root / "task_sf.pt"
            sf_file.write_bytes(b'frozen SF checkpoint fixture')
            metadata = {"source_hashes": {p.name: sha256_file(p) for p in sorted(self.sources.glob('*.pt'))},
                        "protocol_sha256": protocol.fingerprint,
                        "scenario_hashes": self.spec.files.hashes(),
                        "manifest_sha256": sha256_file(self.manifest),
                        "feature_names": list(FEATURE_NAMES),
                        "adjacency": np.ones((2, 2), dtype=bool)}
            self.stack.enter_context(patch('tsc_repaired.task_sf.load_task_sf',
                                           return_value=(FakeTaskSF(k), metadata)))
        runtime = FrozenTrafficRuntime(self.manifest, self.sources, sf_file)
        self.addCleanup(runtime.close)
        return runtime

    def test_actual_source_actions_change_observed_rewards(self):
        runtime = self.make()
        self.assertEqual(self.envs, [])  # Constructor must not start SUMO.
        runtime.reset(self.task, traffic_seed=123)
        first = runtime.step(np.array([0, 0]))
        self.assertEqual(first.rewards.tolist(), [0., 0.])
        runtime.reset(self.task, traffic_seed=123)
        second = runtime.step(np.array([1, 1]))
        self.assertEqual(second.rewards.tolist(), [1., 1.])
        self.assertEqual(second.actions.tolist(), [1, 1])
        self.assertEqual(runtime.env.last_actions, {"a": 1, "b": 1})
        self.assertEqual(runtime.env.sumo_seed, 123)
        self.assertEqual(runtime.env.reset_seed, 123)
        self.assertTrue(runtime.audit_frozen()["frozen_models_unchanged"])

    def test_scenario_files_are_frozen_during_rollouts(self):
        runtime = self.make()
        self.routes.write_text('<routes><route id="changed" edges="ab"/></routes>')
        with self.assertRaisesRegex(RuntimeError, "changed"):
            runtime.audit_frozen()

    def test_saved_source_node_and_feature_identity_must_match(self):
        runtime = self.make()
        path = runtime.source_paths[0]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["node_ids"] = ["b", "a"]
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "node identities"):
            FrozenTrafficRuntime(self.manifest, self.sources)
        payload["node_ids"] = list(self.ids)
        payload["feature_names"] = list(reversed(FEATURE_NAMES))
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "feature definitions"):
            FrozenTrafficRuntime(self.manifest, self.sources)

    def test_heterogeneous_tasks_bind_before_first_reward(self):
        runtime = self.make()
        tasks = np.stack((self.task, self.task * 2.))
        runtime.reset({"weights": tasks}, 42)
        result = runtime.step(np.array([1, 1]))
        self.assertEqual(result.rewards.tolist(), [1., 2.])
        np.testing.assert_allclose(result.rewards, (result.features * tasks).sum(-1))

    def test_k1_consistency_and_terminal_cap(self):
        runtime = self.make(k=1)
        state = runtime.reset(self.task, 3)
        np.testing.assert_array_equal(runtime.candidate_actions(state, 0), np.zeros((2, 1)))
        self.assertFalse(runtime.step(np.zeros(2, dtype=np.int64)).done)
        self.assertTrue(runtime.step(np.zeros(2, dtype=np.int64)).done)
        with self.assertRaisesRegex(RuntimeError, 'termination'):
            runtime.step(np.zeros(2, dtype=np.int64))
        self.assertEqual(runtime.env.calls, 2)

    def test_terminal_missing_obs_allowed_but_nonterminal_rejected(self):
        runtime = self.make()
        runtime.reset(self.task, 3)
        runtime.env.omit_observation = True
        with self.assertRaisesRegex(RuntimeError, 'nonterminal observation'):
            runtime.step(np.array([1, 1]))
        runtime.reset(self.task, 3)
        runtime.env.terminate_at = 1
        runtime.env.empty_terminal = True
        self.assertTrue(runtime.step(np.array([1, 1])).done)

    def test_missing_reward_and_invalid_source_fail_instead_of_zero_fill(self):
        runtime = self.make()
        runtime.reset(self.task, 3)
        with self.assertRaises(ValueError):
            runtime.step(np.array([0.5, 1.0]))
        with self.assertRaises(ValueError):
            runtime.step(np.array([0, 2]))
        self.assertEqual(runtime.env.calls, 0)
        runtime.env.omit_reward = True
        with self.assertRaisesRegex(RuntimeError, 'reward is missing'):
            runtime.step(np.array([1, 1]))

    def test_close_then_reset_reopens_and_preserves_checkpoint_bytes(self):
        runtime = self.make()
        before = {p.name: p.read_bytes() for p in runtime.source_paths}
        runtime.reset(self.task, 4, tripinfo_path=self.root / "trips" / "episode0.xml")
        previous = runtime.env
        runtime.close()
        runtime.close()
        self.assertTrue(previous.closed)
        runtime.reset(self.task, 5, tripinfo_path=self.root / "trips" / "episode1.xml")
        self.assertIsNot(runtime.env, previous)
        self.assertNotEqual(previous.tripinfo_path, runtime.env.tripinfo_path)
        self.assertEqual(before, {p.name: p.read_bytes() for p in runtime.source_paths})
        self.assertFalse(runtime.source_scenario_binding_verified)
        with runtime.source_paths[0].open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            runtime.audit_frozen()

    def test_sf_queries_use_task_weights_not_selected_entries(self):
        runtime = self.make(sf=True)
        state = runtime.reset(self.task, 11)
        first = runtime.sf_values(state, 0)
        runtime.step(np.array([1, 0]))
        np.testing.assert_array_equal(first, np.ones((2, 2)))
        runtime.reset(self.task * 2., 11)
        second = runtime.sf_values(state, 0)
        np.testing.assert_array_equal(second, 2. * np.ones((2, 2)))
        np.testing.assert_array_equal(runtime.sf_values(state, runtime.horizon), np.zeros((2, 2)))
        self.assertTrue(runtime.audit_frozen()["frozen_models_unchanged"])

    def test_sf_rejects_manifest_horizon_or_scale_drift(self):
        self.make(sf=True)
        for field, value in (("horizon_steps", 3), ("scale", 7.0)):
            self.manifest.write_text('{"scenario":{"' + field + '":' + str(value) + '}}')
            settings = {name: getattr(self.spec, name) for name in (
                "delta_time", "horizon_steps", "yellow_time", "min_green", "max_green", "scale")}
            settings[field] = value
            changed_spec = ScenarioSpec(self.spec.files, **settings)
            with patch.object(ScenarioSpec, 'from_manifest', return_value=changed_spec):
                with self.assertRaisesRegex(ValueError, 'manifest_sha256'):
                    FrozenTrafficRuntime(self.manifest, self.sources, self.root / 'task_sf.pt')

    def test_source_binding_requires_files_and_all_scenario_settings(self):
        runtime = self.make(source_file_binding=True)
        self.assertFalse(runtime.source_scenario_binding_verified)
        settings = {name: getattr(self.spec, name) for name in (
            "delta_time", "horizon_steps", "yellow_time", "min_green", "max_green", "scale")}
        for path in runtime.source_paths:
            payload = torch.load(path, weights_only=False)
            payload['scenario_settings'] = settings.copy()
            torch.save(payload, path)
        bound = FrozenTrafficRuntime(self.manifest, self.sources)
        self.addCleanup(bound.close)
        self.assertTrue(bound.source_scenario_binding_verified)
        for field, value in (("horizon_steps", 9), ("scale", 4.0)):
            payload = torch.load(runtime.source_paths[0], weights_only=False)
            payload['scenario_settings'] = dict(settings, **{field: value})
            torch.save(payload, runtime.source_paths[0])
            with self.assertRaisesRegex(ValueError, 'source scenario settings mismatch'):
                FrozenTrafficRuntime(self.manifest, self.sources)


def main():
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrozenTrafficRuntimeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
