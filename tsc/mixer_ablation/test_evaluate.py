"""Paired A/B evaluation checks; only the environment and loaders are replaced."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tsc_repaired.config import sha256_file
from tsc_repaired.test_rollout_evaluate import FakeRuntime, FakeSelector, FakeProtocol
from . import evaluate as evaluation
from .selector import NORMALIZED_VERSION


class PairedMixerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text('{}')
        self.sources = self.root / "sources"
        self.sources.mkdir()
        for index in range(2):
            (self.sources / f"source_{index}.pt").write_bytes(f"source{index}".encode())
        self.sf = self.root / "sf.pt"
        self.sf.write_bytes(b'SF checkpoint')
        self.checkpoints = {variant: self.root / f"{variant}.pt" for variant in ('control', 'normalized')}
        for variant, path in self.checkpoints.items():
            path.write_bytes(variant.encode())
        self.protocol = FakeProtocol()
        self.manifest = SimpleNamespace(require_stage=lambda stage: self.assertEqual(stage, 'pilot'))
        self.models = {'control': FakeSelector(0), 'normalized': FakeSelector(1)}
        FakeRuntime.instances.clear()
        FakeRuntime.source_count = 2
        FakeRuntime.fail_on_step = False
        self.payloads = {variant: self.payload(variant) for variant in self.models}

    def payload(self, variant):
        return {
            'version': evaluation.SELECTOR_VERSION if variant == 'control' else NORMALIZED_VERSION,
            'metadata': {
                'ablation_variant': variant,
                'initial_parameter_sha256': 'a' * 64,
                'training_configuration': {'batch_size': 16, 'updates_per_step': 1, 'learning_rate': 3e-4},
                'seed': 100, 'episodes': 10, 'transitions': 600, 'optimizer_updates': 585,
                'checkpoint_selection': 'fixed-budget endpoint; no validation selection',
                'training_reward_reduction': 'mean node reward',
                'source_hashes': {p.name: sha256_file(p) for p in sorted(self.sources.glob('source_*.pt'))},
                'sf_sha256': sha256_file(self.sf),
                'protocol_fingerprint': self.protocol.fingerprint,
                'node_ids': ['node_a', 'node_b'], 'training_split': 'train',
                'test_accessed': False, 'test_updates': 0,
            },
            'config': {'observation_dim': 3, 'source_count': 2, 'action_count': 2,
                       'task_dim': 5, 'horizon': 3, 'gamma': .5},
            'adjacency': np.ones((2, 2), bool),
        }

    def run_evaluation(self, output='result.json', **kwargs):
        def load(path, device='cpu'):
            variant = Path(path).stem
            return self.models[variant], self.payloads[variant]
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, 'FrozenTrafficRuntime', FakeRuntime))
            stack.enter_context(patch.object(evaluation, 'load_manifest', return_value=self.manifest))
            stack.enter_context(patch.object(evaluation, 'load_protocol', return_value=self.protocol))
            stack.enter_context(patch.object(evaluation, 'load_ablation_selector', side_effect=load))
            stack.enter_context(patch.object(torch.optim, 'Adam', side_effect=AssertionError('optimizer started')))
            return evaluation.evaluate_ablation(self.manifest_path, self.sources, self.sf,
                self.checkpoints['control'], self.checkpoints['normalized'], self.root / output, **kwargs)

    def test_real_actions_produce_paired_returns_without_updates(self):
        before = {p.name: p.read_bytes() for p in [self.sf, *self.checkpoints.values()]}
        report = self.run_evaluation()
        self.assertEqual(report['status'], 'PASS', report['errors'])
        self.assertEqual(report['completed_episode_count'], 10)
        self.assertEqual(len(report['episode_evidence_sha256']), 10)
        self.assertEqual(report['unavailable_episode_evidence'], [])
        self.assertEqual(report['evaluation_updates'], 0)
        self.assertFalse(report['test_accessed'])
        self.assertEqual(report['summary']['control_ddqn']['mean_episode_team_return'], 6.)
        self.assertEqual(report['summary']['normalized_ddqn']['mean_episode_team_return'], 12.)
        self.assertIsNone(report['summary']['control_ddqn']['mean_mean_waiting_time_seconds'])
        self.assertEqual(len(report['paired_differences']), 2)
        for row in report['paired_differences']:
            self.assertEqual(row['traffic_seed'], 10000)
            self.assertEqual(row['metrics']['episode_team_return']['normalized_minus_control'], 6.)
            self.assertEqual(row['metrics']['discounted_team_return']['normalized_minus_control'], 4.5)
            self.assertIsNone(row['metrics']['mean_waiting_time_seconds']['normalized_minus_control'])
        for model in self.models.values():
            self.assertFalse(model.training)
            self.assertFalse(model.bias.requires_grad)
            self.assertIsNone(model.bias.grad)
        self.assertEqual(before, {p.name: p.read_bytes() for p in [self.sf, *self.checkpoints.values()]})
        for row in report['episodes']:
            path = self.root / report['episode_evidence_directory'] / row['tripinfo_file']
            self.assertEqual(report['episode_evidence_sha256'][path.name], sha256_file(path))

    def test_all_methods_share_exact_task_seed_pairs(self):
        report = self.run_evaluation(episodes_per_task=2, all_fixed_sources=True)
        self.assertEqual(report['status'], 'PASS', report['errors'])
        self.assertEqual(report['completed_episode_count'], 24)
        pairs = {}
        for row in report['episodes']:
            pairs.setdefault(row['method'], set()).add((row['task_id'], row['traffic_seed']))
        expected = {('task/A', 10000), ('task/A', 10001), ('task/B', 10000), ('task/B', 10001)}
        self.assertEqual(set(pairs), {'control_ddqn', 'normalized_ddqn', 'independent_entry',
                                    'sync_entry', 'fixed_source_0', 'fixed_source_1'})
        self.assertTrue(all(value == expected for value in pairs.values()))

    def test_training_or_topology_mismatch_refused_before_any_rollout(self):
        self.payloads['normalized']['metadata']['initial_parameter_sha256'] = 'b' * 64
        report = self.run_evaluation('wrong_init.json')
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertFalse(FakeRuntime.instances[-1].reset_calls)
        self.assertIn('initial_parameter_sha256', report['errors'][0])
        self.payloads['normalized']['metadata']['initial_parameter_sha256'] = 'a' * 64
        for payload in self.payloads.values():
            payload['adjacency'] = np.eye(2, dtype=bool)
        report = self.run_evaluation('wrong_graph.json')
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertFalse(FakeRuntime.instances[-1].reset_calls)
        self.assertIn('adjacency differs', report['errors'][0])

    def test_later_rollout_failure_retains_rows_without_subset_average(self):
        original = FakeRuntime.step
        def fail_after_control(runtime, entries):
            if len(runtime.reset_calls) > 1:
                raise RuntimeError('deliberate second-arm failure')
            return original(runtime, entries)
        with patch.object(FakeRuntime, 'step', fail_after_control):
            report = self.run_evaluation()
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertIsNone(report['summary'])
        self.assertIsNone(report['paired_differences'])
        self.assertEqual(report['completed_episode_count'], 1)
        self.assertEqual([row['status'] for row in report['episodes']], ['PASS', 'NOT_READY'])
        self.assertTrue((self.root / 'result.json').is_file())
        self.assertGreater(FakeRuntime.instances[-1].close_calls, 0)

    def test_checkpoint_or_nonpersistent_buffer_mutation_invalidates_result(self):
        original = FakeRuntime.step
        def mutate_file(runtime, entries):
            transition = original(runtime, entries)
            path = self.checkpoints['normalized']
            path.write_bytes(path.read_bytes() + b'changed')
            return transition
        with patch.object(FakeRuntime, 'step', mutate_file):
            report = self.run_evaluation('changed_file.json')
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertIsNone(report['summary'])
        self.assertIn('checkpoint file changed', report['errors'][0])
        class MutatingSelector(FakeSelector):
            def __init__(self):
                super().__init__(1)
                self.register_buffer('counter', torch.zeros(1), persistent=False)
            def choose_entries(self, *args):
                result = super().choose_entries(*args)
                self.counter.add_(1)
                return result
        self.models['normalized'] = MutatingSelector()
        report = self.run_evaluation('changed_buffer.json')
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertIsNone(report['summary'])
        self.assertIn('parameters or buffers changed', report['errors'][0])

    def test_existing_report_is_not_overwritten_or_started(self):
        path = self.root / 'existing.json'
        path.write_text('keep original')
        report = self.run_evaluation('existing.json')
        self.assertEqual(report['status'], 'NOT_READY')
        self.assertEqual(path.read_text(), 'keep original')
        self.assertFalse(FakeRuntime.instances)


if __name__ == '__main__':
    unittest.main(verbosity=2)
