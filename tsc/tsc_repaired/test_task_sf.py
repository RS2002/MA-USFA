"""Meaningful V2 SF checks; uses a fake traffic environment, never SUMO."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
# Adam lazily imports torch._dynamo. Load before patch.dict(sys.modules), so
# test cleanup does not undo registrations owned by PyTorch itself.
import torch._dynamo

from . import collect_task_reference, train_task_sf
from .config import sha256_file
from .features import FEATURE_NAMES
from .task_sf import (REFERENCE_VERSION, TaskConditionedSF, TaskSFConfig,
                      load_task_sf, task_sf_checkpoint, task_sf_td_target)


class TaskSFContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = TaskConditionedSF(TaskSFConfig(
            state_dim=3, source_count=3, task_dim=5, hidden_dim=16,
            attention_dim=8))
        self.state = torch.randn(2, 3, 3)
        self.weights = torch.randn(2, 3, 5)
        self.sources = torch.tensor([[0, 1, 2], [1, 2, 0]])
        self.time = torch.tensor([0.0, 0.5])

    def test_own_task_cannot_change_own_sf_but_teammate_task_can(self):
        base = self.model(self.state, self.sources, self.weights, self.time)
        own_changed = self.weights.clone()
        own_changed[:, 0] += 10.0
        changed = self.model(self.state, self.sources, own_changed, self.time)
        torch.testing.assert_close(base[:, 0], changed[:, 0], rtol=0, atol=0)
        other_changed = self.weights.clone()
        other_changed[:, 1] += 10.0
        other = self.model(self.state, self.sources, other_changed, self.time)
        self.assertGreater(float((base[:, 0] - other[:, 0]).detach().abs().max()), 1e-6)

    def test_mask_excludes_non_neighbors_and_handles_isolated_nodes(self):
        graph = torch.eye(3, dtype=torch.bool)
        base = self.model(self.state, self.sources, self.weights, self.time, graph)
        other = self.model(self.state, self.sources, self.weights * 100., self.time, graph)
        torch.testing.assert_close(base, other, rtol=0, atol=0)
        one = self.model(self.state[:, :1], self.sources[:, :1], self.weights[:, :1], self.time)
        self.assertEqual(one.shape, (2, 1, 5))
        self.assertTrue(torch.isfinite(one).all())

    def test_candidate_values_use_own_reward_weights_without_entry_context(self):
        values = self.model.candidate_values(self.state, self.weights, self.time)
        expected = []
        for source in range(3):
            own = torch.full((2, 3), source)
            expected.append((self.model(self.state, own, self.weights, self.time) * self.weights).sum(-1))
        torch.testing.assert_close(values, torch.stack(expected, -1))
        self.assertEqual(values.shape, (2, 3, 3))
        # K=3 is deliberately different from the five task/feature dimensions.
        self.assertEqual(self.model.config.source_count, 3)
        self.assertEqual(self.model.config.task_dim, 5)

    def test_node_permutation_is_equivariant(self):
        order = torch.tensor([2, 0, 1])
        graph = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.bool)
        result = self.model(self.state, self.sources, self.weights, self.time, graph)
        permuted = self.model(self.state[:, order], self.sources[:, order],
                              self.weights[:, order], self.time,
                              graph[order][:, order])
        torch.testing.assert_close(permuted, result[:, order], rtol=1e-5, atol=1e-6)

    def test_terminal_rows_have_exactly_zero_bootstrap(self):
        phi = torch.arange(30.).reshape(2, 3, 5)
        bootstrap = torch.full_like(phi, 100.)
        target = task_sf_td_target(phi, bootstrap, torch.tensor([True, False]), .9)
        torch.testing.assert_close(target[0], phi[0], rtol=0, atol=0)
        torch.testing.assert_close(target[1], phi[1] + 90., rtol=0, atol=0)
        bootstrap[0] = float("nan")
        target = task_sf_td_target(phi, bootstrap, torch.tensor([True, False]), .9)
        torch.testing.assert_close(target[0], phi[0], rtol=0, atol=0)

    def test_loader_freezes_v2_and_refuses_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pt"
            torch.save(task_sf_checkpoint(self.model), path)
            loaded, metadata = load_task_sf(path)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            torch.testing.assert_close(
                loaded(self.state, self.sources, self.weights, self.time),
                self.model(self.state, self.sources, self.weights, self.time))
            torch.save({"state_dict": self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "legacy SF needs retraining"):
                load_task_sf(path)


class ReferenceTrainingChainTests(unittest.TestCase):
    def test_train_only_reference_to_frozen_checkpoint_without_sumo(self):
        """Prove task labels, finite-horizon terminals and identities survive."""
        protocol_hash = "protocol-identity"
        instances = []

        class FakeProtocol:
            fingerprint = protocol_hash
            sampling_temperature = .5
            sampling_resample = "each_step"

            def tasks(self, split):
                if split != "train":
                    raise AssertionError("non-training tasks accessed")
                return [{"id": "train-A", "weights": [1., 2., 3., 4., 5.]}]

            def traffic_seeds(self, split):
                if split != "train":
                    raise AssertionError("non-training traffic seeds accessed")
                return [11, 12]

        class FakeRuntime:
            def __init__(self, *args, **kwargs):
                self.N, self.K, self.horizon = 2, 3, 2
                self.ids = ("a", "b")
                self.anchors = np.ones((3, 5), dtype=np.float32)
                self.source_hashes = {"source_0.pt": "sha0", "source_1.pt": "sha1", "source_2.pt": "sha2"}
                self.adjacency = np.ones((2, 2), dtype=bool)
                self.spec = types.SimpleNamespace(files=types.SimpleNamespace(hashes=lambda: {"network": "net", "routes": "rou"}))
                self.closed = False
                instances.append(self)

            def reset(self, task, traffic_seed):
                self.steps = 0
                self.weights = task
                return np.zeros((2, 3), dtype=np.float32)

            def step(self, entries):
                self.steps += 1
                return types.SimpleNamespace(
                    state=np.full((2, 3), self.steps, dtype=np.float32),
                    # Never announces done: collector must mark horizon end.
                    done=False,
                    features=np.repeat(np.asarray(entries)[:, None], 5, axis=-1).astype(np.float32))

            def close(self):
                self.closed = True

            def audit_frozen(self):
                return {"frozen_models_unchanged": True, "optimizer_updates": 0}

        def validated(weights, count):
            value = np.asarray(weights, dtype=np.float32)
            return np.repeat(value[None], count, axis=0) if value.ndim == 1 else value

        fake_protocol_module = types.ModuleType("tsc_repaired.task_protocol_v2")
        fake_protocol_module.load_protocol = lambda *args, **kwargs: FakeProtocol()
        fake_protocol_module.validate_task_weights = validated
        fake_protocol_module.source_sampling_probabilities = lambda weights, anchors, temperature: np.tile([.2, .3, .5], (len(weights), 1))
        fake_runtime_module = types.ModuleType("tsc_repaired.traffic_runtime")
        fake_runtime_module.FrozenTrafficRuntime = FakeRuntime
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            fake_manifest = types.SimpleNamespace(
                raw={"traffic_task_protocol": {"sha256": protocol_hash}},
                require_stage=lambda stage: None)
            with patch.dict(sys.modules, {
                "tsc_repaired.task_protocol_v2": fake_protocol_module,
                "tsc_repaired.traffic_runtime": fake_runtime_module,
            }), patch.object(collect_task_reference, "load_manifest", return_value=fake_manifest), patch.object(train_task_sf, "load_manifest", return_value=fake_manifest):
                reference = root / "reference.npz"
                report = collect_task_reference.collect_reference_v2(
                    manifest, root / "sources", reference, seed=7, episodes=2)
                self.assertEqual(report["transitions"], 4)
                self.assertFalse(report["test_accessed"])
                self.assertTrue(instances[0].closed)
                with np.load(reference, allow_pickle=False) as data:
                    self.assertNotIn("context", data)
                    self.assertEqual(str(data["version"].item()), REFERENCE_VERSION)
                    np.testing.assert_array_equal(data["done"], [False, True, False, True])
                    self.assertEqual(data["weights"].shape, (4, 2, 5))
                    np.testing.assert_allclose(data["weights"][0, 0], [1, 2, 3, 4, 5])
                trained = train_task_sf.train_task_sf(
                    manifest, reference, root / "sf", seed=3, steps=3,
                    batch_size=2, hidden_dim=16, attention_dim=8)
                self.assertEqual(trained["optimizer_updates"], 3)
                self.assertEqual(trained["source_count"], 3)
                self.assertEqual(trained["task_dim"], 5)
                model, checkpoint = load_task_sf(root / "sf" / "task_sf.pt")
                self.assertEqual(checkpoint["protocol_sha256"], protocol_hash)
                self.assertEqual(checkpoint["source_hashes"], instances[0].source_hashes)
                self.assertEqual(checkpoint["scenario_hashes"], {"network": "net", "routes": "rou"})
                self.assertEqual(checkpoint["feature_names"], list(FEATURE_NAMES))
                self.assertEqual(checkpoint["references_sha256"], sha256_file(reference))
                self.assertFalse(model.training)
                with np.load(reference, allow_pickle=False) as raw:
                    insufficient = {name: raw[name].copy() for name in raw.files}
                insufficient["entries"][:] = 0
                bad_reference = root / "insufficient.npz"
                np.savez_compressed(bad_reference, **insufficient)
                with self.assertRaisesRegex(ValueError, "did not observe sources"):
                    train_task_sf.train_task_sf(
                        manifest, bad_reference, root / "insufficient_sf", steps=3)
                self.assertFalse((root / "insufficient_sf").exists())
                with self.assertRaises(FileExistsError):
                    collect_task_reference.collect_reference_v2(
                        manifest, root / "sources", reference, seed=7, episodes=2)


def main():
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
