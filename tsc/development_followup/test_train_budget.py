"""Continuous fixed-schedule DDQN checks with executed fake-environment actions."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Resolve lazy registration before optional-dependency doubles.

from mixer_ablation import train_upper as old_training
from mixer_ablation.selector import load_ablation_selector, parameter_sha256
from . import train_budget as training


class BudgetTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / 'sources'; self.sources.mkdir()
        for index in range(2):
            (self.sources / f'source_{index}.pt').write_bytes(f'frozen source {index}'.encode())
        self.sf = self.root / 'sf.pt'
        torch.save({'config': {'gamma': .5}}, self.sf)
        self.before = {path: path.read_bytes() for path in [self.sf, *self.sources.glob('*.pt')]}
        hashes = training._source_hashes(self.sources)
        self.instances, self.visited = [], []
        owner = self

        class FakeRuntime:
            def __init__(self, *args, **kwargs):
                self.ids = ('a', 'b'); self.N = 2; self.K = 2
                self.observation_dim = 3; self.action_count = 3; self.horizon = 3
                self.source_hashes = hashes; self.adjacency = np.eye(2, dtype=bool)
                self.weights = np.ones((2, 5), dtype=np.float32)
                self.resets = []; self.actions = []; self.closed = False
                owner.instances.append(self)
            def reset(self, task, seed):
                self.resets.append((task['id'], seed)); self.step_index = 0
                self.weights = np.asarray(task['weights'], dtype=np.float32)
                return np.zeros((2, 3), dtype=np.float32)
            def candidate_actions(self, state, step):
                return np.array([[0, 1], [1, 0]], dtype=np.int64)
            def sf_values(self, state, step):
                return np.array([[1., 2.], [1., 2.]], dtype=np.float32)
            def step(self, entries):
                actions = self.candidate_actions(None, self.step_index)[np.arange(2), entries]
                self.actions.append(actions.copy()); self.step_index += 1
                return SimpleNamespace(state=np.full((2, 3), self.step_index, dtype=np.float32),
                                       rewards=actions.astype(np.float32) + 1.,
                                       done=self.step_index == self.horizon)
            def audit_frozen(self):
                return {'frozen_models_unchanged': True, 'membership_failures': 0}
            def close(self):
                self.closed = True

        self.runtime_class = FakeRuntime
        class FakeProtocol:
            fingerprint = 'f' * 64
            def tasks(self, split):
                owner.visited.append(('tasks', split))
                assert split == 'train', 'held-out task access'
                return [{'id': 'a', 'weights': np.ones((2, 5), dtype=np.float32)},
                        {'id': 'b', 'weights': np.full((2, 5), 2., dtype=np.float32)}]
            def traffic_seeds(self, split):
                owner.visited.append(('traffic', split))
                assert split == 'train', 'held-out traffic access'
                return [11, 12]

        runtime_module = ModuleType('tsc_repaired.traffic_runtime')
        runtime_module.FrozenTrafficRuntime = FakeRuntime
        protocol_module = ModuleType('tsc_repaired.task_protocol_v2')
        protocol_module.load_protocol = lambda *args: FakeProtocol()
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, {'tsc_repaired.traffic_runtime': runtime_module,
                                                     'tsc_repaired.task_protocol_v2': protocol_module}))
        manifest = SimpleNamespace(require_stage=lambda stage: None, resource_limits={'max_rss_gib': 1.})
        stack.enter_context(patch.object(training, 'load_manifest', return_value=manifest))
        stack.enter_context(patch.object(old_training, 'load_manifest', return_value=manifest))

    def train(self, name='budget', **kwargs):
        defaults = dict(seed=7, checkpoints=(5, 8), epsilon_decay_transitions=12,
                        batch_size=2, replay_capacity=16)
        defaults.update(kwargs)
        return training.train_budget_upper('manifest', self.sources, self.sf, self.root/name, **defaults)

    def test_prefix_matches_original_and_snapshots_leave_continuous_trajectory_unchanged(self):
        original = old_training.train_ablation_upper(
            'manifest', self.sources, self.sf, self.root/'old_prefix', 'normalized',
            seed=7, episodes=5, batch_size=2, replay_capacity=16)
        old, old_saved = load_ablation_selector(original['checkpoint'])
        trainer_instances = []
        real_trainer = training.TaskDDQNTrainer
        def make_trainer(*args, **kwargs):
            trainer_instances.append(real_trainer(*args, **kwargs))
            return trainer_instances[-1]
        with patch.object(training, 'TaskDDQNTrainer', side_effect=make_trainer):
            report = self.train()
        self.assertEqual(len(trainer_instances), 1)
        self.assertEqual(report['transitions'], 24)
        self.assertEqual(report['optimizer_updates'], 23)
        self.assertEqual(trainer_instances[0].updates, 23)
        self.assertEqual(len(trainer_instances[0].replay), 16)
        snapshots = report['checkpoints']
        self.assertEqual(set(snapshots), {'5', '8'})
        prefix, saved = load_ablation_selector(snapshots['5']['checkpoint'])
        self.assertEqual(parameter_sha256(prefix), parameter_sha256(old))
        self.assertEqual(snapshots['5']['last_metrics'], original['last_metrics'])
        self.assertEqual(saved['metadata']['transitions'], 15)
        self.assertEqual(saved['metadata']['optimizer_updates'], 14)
        self.assertEqual(saved['metadata']['episodes'], 5)
        self.assertEqual(report['initial_parameter_sha256'], old_saved['metadata']['initial_parameter_sha256'])
        for actual, prior in zip(self.instances[1].actions[:15], self.instances[0].actions):
            self.assertTrue(np.array_equal(actual, prior))
        endpoint, _ = load_ablation_selector(snapshots['8']['checkpoint'])
        self.assertNotEqual(parameter_sha256(prefix), parameter_sha256(endpoint))
        for ep, record in snapshots.items():
            self.assertEqual(training.sha256(record['checkpoint']), record['checkpoint_sha256'])
            loaded, payload = load_ablation_selector(record['checkpoint'])
            self.assertEqual(parameter_sha256(loaded), record['parameter_sha256'])
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertEqual(payload['metadata']['training_configuration']['epsilon_decay_transitions'], 12)
        # Omitting intermediate serialization must leave the final trajectory
        # unchanged: it is one optimizer/replay/RNG stream, not separate runs.
        unsplit = self.train('without_snapshot', checkpoints=(8,))
        self.assertEqual(unsplit['checkpoints']['8']['parameter_sha256'], snapshots['8']['parameter_sha256'])
        self.assertEqual(unsplit['episode_reports'], report['episode_reports'])
        self.assertEqual(self.before, {path: path.read_bytes() for path in self.before})
        self.assertTrue(all(instance.closed for instance in self.instances))
        self.assertTrue(all(split == 'train' for _, split in self.visited))
        self.assertFalse(report['test_accessed']); self.assertFalse(report['policy_validation_evaluated'])
        with self.assertRaisesRegex(FileExistsError, 'nonempty'):
            self.train()

    def test_interrupted_run_keeps_marker_and_cannot_silently_restart(self):
        with patch.object(self.runtime_class, 'step', side_effect=RuntimeError('controlled interruption')):
            with self.assertRaisesRegex(RuntimeError, 'controlled interruption'):
                self.train()
        marker = self.root/'budget'/'training_started.json'
        self.assertTrue(marker.is_file())
        self.assertFalse((self.root/'budget'/'train_budget.json').exists())
        self.assertTrue(self.instances[-1].closed)
        with self.assertRaisesRegex(FileExistsError, 'nonempty'):
            self.train()

    def test_declared_budgets_and_schedule_reject_invalid_inputs(self):
        for kwargs in ({'checkpoints': ()}, {'checkpoints': (2, 2)}, {'checkpoints': (3, 1)},
                       {'checkpoints': (True, 2)}, {'checkpoints': (1, 31)},
                       {'epsilon_decay_transitions': 0}, {'epsilon_decay_transitions': 1.5},
                       {'replay_capacity': 0}, {'updates_per_step': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.train(**kwargs)
        self.assertFalse((self.root/'budget').exists())

    def test_cpu_clone_does_not_alias_live_training_storage(self):
        live = torch.tensor([1., 2.], requires_grad=True)
        snapshot = training._cpu_clone({'online': [live], 'nested': ({'v': live},)})
        with torch.no_grad():
            live.add_(10.)
        self.assertTrue(torch.equal(snapshot['online'][0], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(snapshot['nested'][0]['v'], torch.tensor([1., 2.])))
        self.assertFalse(snapshot['online'][0].requires_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
