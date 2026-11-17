"""Meaningful selector and paired-training checks, without SUMO."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Resolve lazy imports before optional-dependency test doubles.

from tsc_repaired.task_selector import (
    TaskAttentionSelector, TaskSelectorConfig, load_selector,
)
from tsc_repaired import train_online_upper as original_training
from . import train_upper as training
from .selector import (
    NORMALIZED_VERSION, NormalizedTaskAttentionSelector, checkpoint_spec,
    load_ablation_selector, parameter_sha256, version_for_variant,
)


class SelectorNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TaskSelectorConfig(3, 2, 3, hidden_dim=16, attention_heads=2,
                                         gamma=.5, horizon=3)
        torch.manual_seed(11)
        self.control = TaskAttentionSelector(self.config).set_adjacency(np.eye(2, dtype=bool))
        torch.manual_seed(11)
        self.normalized = NormalizedTaskAttentionSelector(self.config).set_adjacency(np.eye(2, dtype=bool))
        self.inputs = (torch.tensor([[[.1, 0., .2], [.3, .2, 0.]]]),
                       torch.ones((1, 2, 5)), torch.tensor([[[0, 1], [1, 2]]]),
                       torch.tensor([[[3., 1.], [2., 5.]]]), torch.tensor([0.]))

    def test_initial_parameters_utilities_q_and_choices_are_identical(self):
        self.assertEqual(parameter_sha256(self.control), parameter_sha256(self.normalized))
        c = self.control.components(*self.inputs)
        n = self.normalized.components(*self.inputs)
        self.assertTrue(torch.equal(c['utilities'], n['utilities']))
        self.assertTrue(torch.equal(n['mixing'], torch.ones((1, 2))))
        choices = self.control.choose_entries(*self.inputs)
        self.assertTrue(torch.equal(choices, self.normalized.choose_entries(*self.inputs)))
        self.assertTrue(torch.equal(choices, self.inputs[3].argmax(-1)))
        self.assertTrue(torch.equal(self.control.q_for_entries(*self.inputs, choices),
                                    self.normalized.q_for_entries(*self.inputs, choices)))

    def test_normalization_changes_team_q_and_gradients_but_not_same_model_argmax(self):
        with torch.no_grad():
            self.control.mixer_head.bias.fill_(.8)
            self.control.residual_head[-1].bias[:] = torch.tensor([.2, -.1])
        self.normalized.load_state_dict(self.control.state_dict())
        c = self.control.components(*self.inputs)
        n = self.normalized.components(*self.inputs)
        self.assertTrue(torch.equal(c['utilities'], n['utilities']))
        self.assertTrue(bool((n['mixing'] > 0).all()))
        self.assertTrue(torch.allclose(n['mixing'].mean(-1), torch.ones(1), atol=1e-7))
        choices = self.control.choose_entries(*self.inputs)
        self.assertTrue(torch.equal(choices, self.normalized.choose_entries(*self.inputs)))
        q_control = self.control.q_for_entries(*self.inputs, choices)
        q_normalized = self.normalized.q_for_entries(*self.inputs, choices)
        self.assertFalse(torch.allclose(q_control, q_normalized))
        q_control.sum().backward(); q_normalized.sum().backward()
        self.assertGreater(float(self.control.mixer_head.bias.grad.abs()), .1)
        self.assertLess(float(self.normalized.mixer_head.bias.grad.abs()), 1e-6)
        self.assertFalse(torch.allclose(self.control.residual_head[-1].bias.grad,
                                        self.normalized.residual_head[-1].bias.grad))
        # Choice-independent positive mixing still gives exact additive greedy.
        qs = []
        for a in range(2):
            for b in range(2):
                qs.append(float(self.normalized.q_for_entries(*self.inputs, torch.tensor([[a, b]])).detach()))
        self.assertAlmostEqual(float(q_normalized.detach()), max(qs), places=6)

    def test_mean_one_for_nonuniform_positive_weights_and_isolated_nodes(self):
        with torch.no_grad():
            self.control.mixer_head.weight.normal_(0., .7)
            self.control.mixer_head.bias.fill_(-1.)
        self.normalized.load_state_dict(self.control.state_dict())
        raw = self.control.components(*self.inputs)['mixing']
        actual = self.normalized.components(*self.inputs)['mixing']
        self.assertTrue(torch.allclose(actual, raw / raw.mean(-1, keepdim=True), atol=1e-6))
        self.assertTrue(torch.allclose(actual.mean(-1), torch.ones(1), atol=1e-7))
        changed = list(self.inputs); changed[1] = changed[1].clone(); changed[1][:, 1] *= 100.
        before = self.normalized.choose_entries(*self.inputs)
        after = self.normalized.choose_entries(*changed)
        self.assertEqual(int(before[0, 0]), int(after[0, 0]))

    def test_variant_loader_freezes_and_rejects_wrong_loader_or_tag(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'upper.pt'
            for variant, model in [('control', self.control), ('normalized', self.normalized)]:
                payload = {'version': version_for_variant(variant), 'config': asdict(self.config),
                           'online': model.state_dict(), 'adjacency': np.eye(2, dtype=bool).tolist(),
                           'metadata': {'ablation_variant': variant}, 'spec': checkpoint_spec(model, variant)}
                torch.save(payload, path)
                original_bytes = path.read_bytes()
                loaded, saved = load_ablation_selector(path)
                self.assertEqual(saved['metadata']['ablation_variant'], variant)
                self.assertEqual(isinstance(loaded, NormalizedTaskAttentionSelector), variant == 'normalized')
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                self.assertFalse(loaded.training)
                self.assertTrue(torch.equal(loaded.choose_entries(*self.inputs), model.choose_entries(*self.inputs)))
                self.assertEqual(path.read_bytes(), original_bytes)
                if variant == 'normalized':
                    self.assertEqual(payload['version'], NORMALIZED_VERSION)
                    with self.assertRaisesRegex(ValueError, 'legacy'):
                        load_selector(path)
                    payload['metadata']['ablation_variant'] = 'control'
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, 'version'):
                        load_ablation_selector(path)


class PairedTrainingTests(unittest.TestCase):
    def test_real_updates_pair_initialization_preserve_frozen_inputs_and_match_old_control(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); sources = root / 'sources'; sources.mkdir()
            for i in range(2):
                (sources / f'source_{i}.pt').write_bytes(f'frozen source {i}'.encode())
            sf = root / 'sf.pt'; torch.save({'config': {'gamma': .5}}, sf)
            frozen_before = {str(p): p.read_bytes() for p in [sf, *sources.glob('*.pt')]}
            hashes = training._source_hashes(sources)
            instances, visited = [], []

            class FakeRuntime:
                def __init__(self, *args, **kwargs):
                    self.ids = ('a', 'b'); self.N = 2; self.K = 2
                    self.observation_dim = 3; self.action_count = 3; self.horizon = 3
                    self.source_hashes = hashes; self.adjacency = np.eye(2, dtype=bool)
                    self.weights = np.ones((2, 5), dtype=np.float32)
                    self.resets = []; self.actions = []; self.closed = False
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
                    actions = self.candidate_actions(None, self.step_index)[np.arange(2), entries]
                    self.actions.append(actions.copy()); self.step_index += 1
                    return SimpleNamespace(state=np.full((2, 3), self.step_index, dtype=np.float32),
                                           rewards=actions.astype(np.float32) + 1.,
                                           done=self.step_index == self.horizon)
                def audit_frozen(self):
                    return {'frozen_models_unchanged': True, 'membership_failures': 0}
                def close(self):
                    self.closed = True

            class FakeProtocol:
                fingerprint = 'f' * 64
                def tasks(self, split):
                    visited.append(('tasks', split))
                    assert split == 'train', 'held-out task access'
                    return [{'id': 'a', 'weights': np.ones((2, 5), dtype=np.float32)},
                            {'id': 'b', 'weights': np.full((2, 5), 2., dtype=np.float32)}]
                def traffic_seeds(self, split):
                    visited.append(('traffic', split))
                    assert split == 'train', 'held-out traffic access'
                    return [11, 12]

            runtime_module = ModuleType('tsc_repaired.traffic_runtime')
            runtime_module.FrozenTrafficRuntime = FakeRuntime
            protocol_module = ModuleType('tsc_repaired.task_protocol_v2')
            protocol_module.load_protocol = lambda *args: FakeProtocol()
            manifest = SimpleNamespace(require_stage=lambda stage: None, resource_limits={'max_rss_gib': 1.})
            kwargs = dict(seed=7, episodes=2, batch_size=2, replay_capacity=16)
            with patch.dict(sys.modules, {'tsc_repaired.traffic_runtime': runtime_module,
                                          'tsc_repaired.task_protocol_v2': protocol_module}), \
                 patch.object(training, 'load_manifest', return_value=manifest), \
                 patch.object(original_training, 'load_manifest', return_value=manifest):
                reports = {variant: training.train_ablation_upper('manifest', sources, sf, root/variant,
                                                                 variant, **kwargs)
                           for variant in ('control', 'normalized')}
                original = original_training.train_online_upper('manifest', sources, sf, root/'original', **kwargs)
            self.assertEqual(reports['control']['initial_parameter_sha256'], reports['normalized']['initial_parameter_sha256'])
            self.assertEqual(reports['control']['training_configuration'], reports['normalized']['training_configuration'])
            for index, variant in enumerate(('control', 'normalized')):
                report = reports[variant]
                self.assertEqual(report['transitions'], 6)
                self.assertEqual(report['optimizer_updates'], 5)
                self.assertEqual(instances[index].resets, [('a', 11), ('b', 12)])
                self.assertTrue(instances[index].closed)
                self.assertFalse(report['test_accessed'])
                self.assertFalse(report['policy_validation_evaluated'])
                loaded, saved = load_ablation_selector(root/variant/'upper_ddqn_v2.pt')
                self.assertEqual(saved['metadata']['ablation_variant'], variant)
                self.assertEqual(saved['metadata']['optimizer_updates'], 5)
                self.assertFalse(loaded.training)
            for ep in reports['normalized']['episode_reports']:
                self.assertAlmostEqual(ep['observed_mixing_mean'], 1., places=6)
            # A true unchanged-control check: endpoint tensors and executed
            # actions match the original frozen-code trainer under one recipe.
            c, _ = load_ablation_selector(root/'control'/'upper_ddqn_v2.pt')
            old, _ = load_selector(root/'original'/'upper_ddqn_v2.pt')
            self.assertEqual(parameter_sha256(c), parameter_sha256(old))
            self.assertEqual(reports['control']['last_metrics'], original['last_metrics'])
            self.assertTrue(all(np.array_equal(a, b) for a,b in zip(instances[0].actions, instances[2].actions)))
            self.assertEqual(frozen_before, {str(p): p.read_bytes() for p in [sf, *sources.glob('*.pt')]})
            self.assertTrue(all(split == 'train' for _, split in visited))


if __name__ == '__main__':
    unittest.main(verbosity=2)
