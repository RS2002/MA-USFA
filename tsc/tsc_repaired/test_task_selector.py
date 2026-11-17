"""Selector/Double-DQN regression tests; no SUMO or scientific run is started."""
from __future__ import annotations

from dataclasses import asdict
from types import ModuleType, SimpleNamespace
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .task_selector import (SELECTOR_VERSION, TaskAttentionSelector, TaskDDQNTrainer,
                            TaskSelectorConfig, TaskTransition, bounded_replay_capacity,
                            load_selector)


class TaskSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.config = TaskSelectorConfig(3, 2, 3, hidden_dim=16, attention_heads=2,
                                         gamma=.5, horizon=4, rho=20.)
        self.model = TaskAttentionSelector(self.config).set_adjacency(np.eye(2, dtype=bool))
        self.obs = torch.zeros((1, 2, 3))
        self.weights = torch.ones((1, 2, 5))
        self.actions = torch.tensor([[[0, 1], [1, 2]]])
        self.values = torch.tensor([[[3., 1.], [2., 5.]]])
        self.times = torch.zeros(1)
        self.inputs = (self.obs, self.weights, self.actions, self.values, self.times)

    def row(self, *, reward=1., done=False, invalid_next=False):
        obs = np.zeros((2, 3), dtype=np.float32)
        nxt = obs.copy()
        if invalid_next:
            nxt[:] = np.nan
        return TaskTransition(
            obs=obs, weights=np.ones((2, 5), dtype=np.float32),
            candidate_actions=np.zeros((2, 2), dtype=np.int64),
            sf_values=np.zeros((2, 2), dtype=np.float32), time_fraction=0.,
            entries=np.array([0, 1]), reward=reward, next_obs=nxt,
            next_candidate_actions=np.zeros((2, 2), dtype=np.int64),
            next_sf_values=np.zeros((2, 2), dtype=np.float32),
            next_time_fraction=.25, done=done)

    def test_zero_initialization_and_exact_additive_argmax(self):
        chosen = self.model.choose_entries(*self.inputs)
        self.assertTrue(torch.equal(chosen, self.values.argmax(-1)))
        parts = self.model.components(*self.inputs)
        self.assertTrue(torch.equal(parts['mixing'], torch.ones((1, 2))))
        self.assertTrue(torch.equal(self.model.q_for_entries(*self.inputs, chosen), torch.tensor([4.])))
        profile_values = []
        for first in range(2):
            for second in range(2):
                entry = torch.tensor([[first, second]])
                profile_values.append(float(self.model.q_for_entries(*self.inputs, entry).detach()))
        self.assertEqual(float(self.model.q_for_entries(*self.inputs, chosen).detach()), max(profile_values))

    def test_primitive_candidates_are_inputs_without_bypassing_source_choice(self):
        changed = self.actions.clone(); changed[0, 0, 0] = 2
        a = self.model.tokens(*self.inputs)
        b = self.model.tokens(self.obs, self.weights, changed, self.values, self.times)
        self.assertFalse(torch.equal(a, b))
        entries = self.model.choose_entries(self.obs, self.weights, changed, self.values, self.times)
        self.assertEqual(tuple(entries.shape), (1, 2))
        self.assertTrue(bool(((entries >= 0) & (entries < 2)).all()))

    def test_executed_profile_only_receives_direct_value_label(self):
        values = self.values.clone().requires_grad_(True)
        entries = torch.tensor([[0, 1]])
        q = self.model.q_for_entries(self.obs, self.weights, self.actions, values, self.times, entries)
        q.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.tensor([[[.5, 0.], [0., .5]]])))

    def test_terminal_target_does_not_read_next_inputs(self):
        trainer = TaskDDQNTrainer(self.model, seed=1, replay_capacity=8)
        with patch.object(trainer.online, 'choose_entries', side_effect=AssertionError('terminal bootstrap')):
            target = trainer.targets([self.row(reward=7., done=True, invalid_next=True)])
        self.assertTrue(torch.equal(target, torch.tensor([7.])))

    def test_double_dqn_uses_online_selection_and_target_evaluation(self):
        with torch.no_grad():
            self.model.residual_head[-1].bias[:] = torch.tensor([1., 0.])
        trainer = TaskDDQNTrainer(self.model, seed=1, replay_capacity=8)
        with torch.no_grad():
            trainer.target.residual_head[-1].bias[:] = torch.tensor([0., 1.])
        target = trainer.targets([self.row(reward=2., done=False)])
        self.assertTrue(torch.equal(target, torch.tensor([2.])))
        # Maximizing the target network instead would give a larger target.
        next_inputs = trainer._inputs([self.row()], next_state=True)
        self.assertTrue(bool((trainer.target(*next_inputs).max(-1).values.mean(-1) > 0.).all()))

    def test_actual_update_is_finite_and_target_is_frozen(self):
        trainer = TaskDDQNTrainer(self.model, seed=1, replay_capacity=8)
        for reward in range(4):
            trainer.add(self.row(reward=float(reward), done=True))
        before = {k: v.detach().clone() for k, v in trainer.online.state_dict().items()}
        metrics = trainer.update(batch_size=2)
        self.assertEqual(metrics['updates'], 1)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in trainer.online.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for p in trainer.target.parameters()))
        self.assertTrue(all(p.grad is None for p in trainer.target.parameters()))

    def test_non_neighbor_task_does_not_change_local_policy_scores(self):
        model = TaskAttentionSelector(self.config).set_adjacency(np.eye(3, dtype=bool))
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(.1)
        obs = torch.zeros((1, 3, 3)); weights = torch.ones((1, 3, 5))
        actions = torch.zeros((1, 3, 2), dtype=torch.long); values = torch.ones((1, 3, 2))
        before = model(obs, weights, actions, values, self.times).detach()
        weights[:, 2, :] *= 100.
        after = model(obs, weights, actions, values, self.times).detach()
        self.assertTrue(torch.equal(before[:, 0], after[:, 0]))
        with self.assertRaisesRegex(ValueError, 'node count'):
            model(*self.inputs)

    def test_lower_and_upper_composition_does_not_rebroadcast_second_hop_tasks(self):
        from .task_sf import TaskConditionedSF, TaskSFConfig
        # A--B--C. C's task changes B's task-conditioned SF, but the complete
        # frozen-lower/upper pipeline at A may only see A and B's raw inputs.
        adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
        lower = TaskConditionedSF(TaskSFConfig(
            state_dim=3, source_count=2, hidden_dim=16, attention_dim=8))
        with torch.no_grad():
            for parameter in lower.parameters():
                parameter.fill_(.1)
        lower.eval().requires_grad_(False)
        torch.manual_seed(17)
        model = TaskAttentionSelector(self.config).set_adjacency(adjacency)
        with torch.no_grad():
            model.residual_head[-1].weight.normal_(0., .2)
            model.residual_head[-1].bias.normal_(0., .1)
            model.mixer_head.weight.normal_(0., .1)
        obs = torch.zeros((1, 3, 3))
        weights = torch.ones((1, 3, 5))
        changed_weights = weights.clone(); changed_weights[:, 2] = 4.
        actions = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
        with torch.no_grad():
            before_sf = lower.candidate_values(obs, weights, self.times, adjacency)
            after_sf = lower.candidate_values(obs, changed_weights, self.times, adjacency)
            before = model.components(obs, weights, actions, before_sf, self.times)
            after = model.components(obs, changed_weights, actions, after_sf, self.times)
        self.assertFalse(torch.allclose(before_sf[:, 1], after_sf[:, 1]))
        self.assertTrue(torch.equal(before_sf[:, 0], after_sf[:, 0]))
        self.assertTrue(torch.equal(before['utilities'][:, 0], after['utilities'][:, 0]))
        self.assertTrue(torch.equal(before['mixing'][:, 0], after['mixing'][:, 0]))
        self.assertFalse(model.spec()['neighbor_sf_rebroadcast'])
        self.assertTrue(torch.equal(before['neighbor_tokens'][..., -2:], torch.zeros((1, 3, 2))))

        # Explicitly recreate the former SF-bearing K/V route. The same real
        # lower model and nonzero selector parameters then leak C's task to A,
        # demonstrating that this test would detect the original two-hop bug.
        def former_leaky_node_zero(task, values):
            encoded = model.encoder(model.tokens(obs, task, actions, values, self.times))
            attended, _ = model.attention(encoded, encoded, encoded,
                                           attn_mask=~model._adjacency, need_weights=False)
            return values[:, 0] + model.config.rho * torch.tanh(
                model.residual_head(encoded + attended))[:, 0]
        with torch.no_grad():
            leaky_before = former_leaky_node_zero(weights, before_sf)
            leaky_after = former_leaky_node_zero(changed_weights, after_sf)
        self.assertFalse(torch.allclose(leaky_before, leaky_after, atol=1e-7, rtol=1e-7))

    def test_save_load_freezes_and_rejects_legacy_or_missing_topology(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'upper.pt'
            payload = {'version': SELECTOR_VERSION, 'config': asdict(self.config),
                       'online': self.model.state_dict(), 'adjacency': np.eye(2, dtype=bool).tolist(),
                       'metadata': {'test_updates': 0}, 'spec': self.model.spec()}
            torch.save(payload, path)
            before = path.read_bytes()
            restored, _ = load_selector(path)
            self.assertFalse(restored.training)
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            self.assertTrue(torch.equal(restored.choose_entries(*self.inputs), self.model.choose_entries(*self.inputs)))
            self.assertEqual(before, path.read_bytes())
            original_spec = payload.pop('spec')
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'no-SF-rebroadcast'):
                load_selector(path)
            payload['spec'] = {**original_spec, 'neighbor_sf_rebroadcast': True}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'no-SF-rebroadcast'):
                load_selector(path)
            payload['spec'] = original_spec
            payload.pop('adjacency'); torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'topology'):
                load_selector(path)
            torch.save({'online': self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                load_selector(path)

    def test_online_training_controls_environment_and_uses_only_train_split(self):
        from . import train_online_upper as training
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); sources = root / 'sources'; sources.mkdir()
            for i in range(2):
                (sources / f'source_{i}.pt').write_bytes(f'frozen source {i}'.encode())
            sf_path = root / 'sf.pt'; torch.save({'config': {'gamma': .5}}, sf_path)
            sf_before = sf_path.read_bytes()
            source_before = {p.name: p.read_bytes() for p in sources.glob('*.pt')}
            source_hashes = training._source_hashes(sources)
            instances = []
            visited_splits = []

            class FakeRuntime:
                def __init__(self, *args, **kwargs):
                    self.ids = ('a', 'b'); self.N = 2; self.K = 2
                    self.observation_dim = 3; self.action_count = 3; self.horizon = 3
                    self.source_hashes = source_hashes
                    self.adjacency = np.eye(2, dtype=bool)
                    self.weights = np.ones((2, 5), dtype=np.float32)
                    self.resets = []; self.executions = []; self.closed = False
                    instances.append(self)

                def reset(self, task, seed):
                    self.resets.append((task['id'], seed)); self.step_index = 0
                    self.weights = np.asarray(task['weights'], dtype=np.float32)
                    return np.zeros((2, 3), dtype=np.float32)

                def candidate_actions(self, state, step):
                    return np.array([[0, 1], [1, 0]], dtype=np.int64)

                def sf_values(self, state, step):
                    return np.array([[1., 2.], [1., 2.]], dtype=np.float32)

                def step(self, entries):
                    self.executions.append(entries.copy())
                    actions = self.candidate_actions(None, self.step_index)[np.arange(2), entries]
                    self.step_index += 1
                    return SimpleNamespace(state=np.full((2, 3), self.step_index, dtype=np.float32),
                                           rewards=actions.astype(np.float32) + 1.,
                                           done=self.step_index == 3)

                def audit_frozen(self):
                    return {'frozen_models_unchanged': True, 'membership_failures': 0}

                def close(self):
                    self.closed = True

            class FakeProtocol:
                fingerprint = 'f' * 64
                def tasks(self, split):
                    visited_splits.append(('tasks', split))
                    if split != 'train':
                        raise AssertionError('non-training task access')
                    return [{'id': 'train-a', 'weights': np.ones((2, 5), dtype=np.float32)},
                            {'id': 'train-b', 'weights': np.full((2, 5), 2., dtype=np.float32)}]
                def traffic_seeds(self, split):
                    visited_splits.append(('traffic', split))
                    if split != 'train':
                        raise AssertionError('non-training traffic access')
                    return [11, 12]

            runtime_module = ModuleType('tsc_repaired.traffic_runtime')
            runtime_module.FrozenTrafficRuntime = FakeRuntime
            protocol_module = ModuleType('tsc_repaired.task_protocol_v2')
            protocol_module.load_protocol = lambda *args: FakeProtocol()
            manifest = SimpleNamespace(require_stage=lambda stage: None,
                                       resource_limits={'max_rss_gib': 1})
            guard = SimpleNamespace(check=lambda: None)
            with patch.dict(sys.modules, {'tsc_repaired.traffic_runtime': runtime_module,
                                          'tsc_repaired.task_protocol_v2': protocol_module}), \
                 patch.object(training, 'load_manifest', return_value=manifest):
                report = training.train_online_upper('manifest', sources, sf_path, root / 'upper',
                                                      seed=7, episodes=2, batch_size=2,
                                                      replay_capacity=16, guard=guard)
            self.assertEqual(report['transitions'], 6)
            self.assertEqual(report['optimizer_updates'], 5)
            self.assertEqual(len(instances[0].executions), 6)
            self.assertEqual(instances[0].resets, [('train-a', 11), ('train-b', 12)])
            self.assertEqual(visited_splits, [('tasks', 'train'), ('traffic', 'train')])
            self.assertTrue(instances[0].closed)
            self.assertFalse(report['policy_validation_evaluated'])
            restored, payload = load_selector(root / 'upper' / 'upper_ddqn_v2.pt')
            self.assertEqual(payload['metadata']['training_split'], 'train')
            self.assertEqual(payload['metadata']['optimizer_updates'], 5)
            self.assertFalse(restored.training)
            self.assertEqual(sf_path.read_bytes(), sf_before)
            self.assertEqual({p.name: p.read_bytes() for p in sources.glob('*.pt')}, source_before)

    def test_replay_memory_budget_is_enforced(self):
        row = self.row()
        capacity, per_row = bounded_replay_capacity(1024, row, 512_000, 16)
        self.assertLess(capacity, 1024)
        self.assertGreaterEqual(capacity, 16)
        self.assertLessEqual((capacity + 64) * per_row, 512_000)
        with self.assertRaisesRegex(ValueError, 'too small'):
            bounded_replay_capacity(1024, row, 100, 16)


def main():
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaskSelectorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
