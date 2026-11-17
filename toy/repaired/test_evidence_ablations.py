"""Short contract tests; no production experiment or sealed task is loaded."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .evidence_ablations import (
    EntryGreedy, NoSFAttention, NoTeammateContextSF, SelfOnlyAttention,
    ablation_tensor_inputs, evaluate_ablation, load_ablation_lower,
    load_ablation_upper, make_ablation_model, model_sha256,
    train_ablation_lower, train_ablation_upper,
)
from .legacy_world import SFWorldExact
from .learning import observation
from .paper_aligned_attention import PaperAlignedAttention
from .paper_aligned_lower import ContextActionSF
from .paper_aligned_run import train_upper


class AblationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.world = SFWorldExact(horizon=4, kappa=.5)
        cells = cls.world.state_cells
        # Four fixed deterministic source components suffice for contracts;
        # the production runner creates the unmodified kappa=0 source library.
        base = np.empty((4, cls.world.S, 2), dtype=np.int64)
        for k in range(4):
            base[k, :, 0] = (cells[:, 0] + k) % 5
            base[k, :, 1] = (cells[:, 1] + 3-k) % 5
        cls.sources = np.broadcast_to(base[:, None], (4, 4, cls.world.S, 2)).copy()
        cls.tasks = [{"id": "contract_train", "kind": "overlap",
                      "weights": [[1.3, .2, .4, .1], [.9, .2, .3, .6]]}]
        cls.val_tasks = [{**cls.tasks[0], "id": "contract_validation"}]

    def inputs(self, batch=3):
        rng = np.random.RandomState(4)
        return (torch.as_tensor(observation(np.arange(batch), np.zeros(batch, dtype=int))),
                torch.as_tensor(rng.rand(batch, 2, 4), dtype=torch.float32),
                torch.as_tensor(rng.randint(5, size=(batch, 2, 4)), dtype=torch.long),
                torch.as_tensor(rng.randn(batch, 2, 4), dtype=torch.float32))

    def test_initial_entry_argmax_and_first_index_ties(self):
        inputs = self.inputs()
        expected = inputs[-1].argmax(-1)
        for variant in ("entry_greedy", "no_attention", "attention_control"):
            model = make_ablation_model(variant)
            torch.testing.assert_close(model.greedy_entries(*inputs), expected)
        tied = (*inputs[:-1], torch.zeros_like(inputs[-1]))
        entry = EntryGreedy()
        self.assertEqual(sum(p.numel() for p in entry.parameters()), 0)
        torch.testing.assert_close(entry.greedy_entries(*tied),
                                   torch.zeros((3, 2), dtype=torch.long))

    def test_self_only_has_no_cross_token_value_exchange(self):
        torch.manual_seed(4)
        model = SelfOnlyAttention()
        torch.nn.init.normal_(model.residual_head[-1].weight)
        inputs = self.inputs()
        other_actions = inputs[2].clone()
        other_actions[:, 1] = (other_actions[:, 1] + 1) % 5
        other_values = inputs[3].clone()
        other_values[:, 1] += 50.
        a = model.components(*inputs)
        b = model.components(inputs[0], inputs[1], other_actions, other_values)
        torch.testing.assert_close(a["utilities"][:, 0], b["utilities"][:, 0])
        model(*inputs).sum().backward()
        self.assertIsNone(model.query.weight.grad)
        self.assertIsNone(model.key.weight.grad)
        self.assertGreater(model.spec()["active_parameters"], 0)

    def test_no_sf_ignores_values_and_never_queries_lower(self):
        model = NoSFAttention()
        torch.nn.init.normal_(model.residual_head[-1].weight)
        inputs = self.inputs()
        torch.testing.assert_close(model(*inputs), model(*inputs[:-1], inputs[-1] + 100.))
        self.assertEqual(model.tokens(*inputs).shape[-1], 33)
        with patch("repaired.evidence_ablations.candidate_values",
                   side_effect=AssertionError("SF must not be queried")):
            out = ablation_tensor_inputs(model, np.array([0]), np.array([0]),
                np.array(self.tasks[0]["weights"], dtype=np.float32)[None], self.sources, None)
            self.assertTrue(torch.equal(out[-1], torch.zeros((1, 2, 4))))
        torch.nn.init.zeros_(model.residual_head[-1].weight)
        with torch.no_grad():
            model.residual_head[-1].bias.fill_(25.)
        self.assertTrue(torch.all(model(*inputs) > 8.))

    def test_lower_context_is_removed_in_training_and_inference(self):
        lower = NoTeammateContextSF()
        inputs = self.inputs()
        obs = inputs[0]
        agent = torch.tensor([0, 1, 0])
        source = torch.tensor([1, 2, 0])
        first = lower(obs, agent, source, inputs[1][:, 0])
        second = lower(obs, agent, source, inputs[1][:, 0] + 500.)
        torch.testing.assert_close(first, second)
        self.assertEqual(lower.net[0].in_features, 11)

    def test_attention_control_matches_original_training_prefix(self):
        torch.manual_seed(7)
        lower = ContextActionSF()
        with tempfile.TemporaryDirectory() as td:
            old, old_summary = train_upper(Path(td) / "old", self.world, self.sources,
                lower, self.tasks, self.val_tasks, seed=8, budget=16, batch=4,
                warmup=4, update_every=4, target_every=2, eval_every=8)
            new, summary = train_ablation_upper(Path(td) / "new", self.world,
                self.sources, lower, self.tasks, self.val_tasks, "attention_control",
                seed=8, budget=16, batch=4, warmup=4, update_every=4,
                target_every=2, eval_every=8)
            self.assertEqual(model_sha256(old), model_sha256(new))
            self.assertEqual(old_summary["logs"], summary["logs"])
            self.assertEqual(old_summary["best_validation"], summary["best_validation"])
            self.assertEqual(summary["updates"], 4)
            self.assertNotEqual(summary["initial_model_sha256"], model_sha256(new))

    def test_short_actual_optimization_freezing_loading_and_no_overwrite(self):
        torch.manual_seed(1)
        context_lower = ContextActionSF()
        source_before = self.sources.copy()
        lower_before = model_sha256(context_lower)
        with tempfile.TemporaryDirectory() as td:
            no_context, meta = train_ablation_lower(Path(td) / "lower", self.sources,
                self.tasks, seed=9, steps=16, horizon=4, gamma=.95, kappa=.5,
                batch_size=4, warmup=4, update_every=4, target_every=2)
            self.assertEqual(meta["updates"], 4)
            self.assertNotEqual(meta["initial_model_sha256"], meta["model_sha256"])
            loaded_lower, _ = load_ablation_lower(Path(td) / "lower" / "lower.pt")
            self.assertEqual(model_sha256(loaded_lower), model_sha256(no_context))
            for variant, lower in (("no_attention", context_lower), ("no_sf", None),
                                    ("no_teammate_conditioning", no_context)):
                out = Path(td) / variant
                model, summary = train_ablation_upper(out, self.world, self.sources,
                    lower, self.tasks, self.val_tasks, variant, seed=10, budget=16,
                    batch=4, warmup=4, update_every=4, target_every=2, eval_every=8)
                self.assertEqual(summary["status"], "COMPLETE")
                self.assertFalse(summary["test_accessed"])
                self.assertNotEqual(summary["initial_model_sha256"], model_sha256(model))
                loaded, _ = load_ablation_upper(out / "upper.pt")
                self.assertEqual(model_sha256(loaded), model_sha256(model))
                was_training = loaded.training
                with torch.no_grad():
                    evaluation = evaluate_ablation(loaded, self.world, self.val_tasks,
                                                  self.sources, lower)
                self.assertEqual(was_training, loaded.training)
                self.assertEqual(len(evaluation["rows"]), 1)
                with self.assertRaises(FileExistsError):
                    train_ablation_upper(out, self.world, self.sources, lower,
                        self.tasks, self.val_tasks, variant, budget=16)
            self.assertEqual(lower_before, model_sha256(context_lower))
            np.testing.assert_array_equal(source_before, self.sources)
            with self.assertRaises(FileExistsError):
                train_ablation_lower(Path(td) / "lower", self.sources, self.tasks,
                                      steps=16, horizon=4)

    def test_partial_checkpoints_are_not_loadable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partial.pt"
            torch.save({"version": "icml-component-ablation-v1", "status": "RUNNING"}, path)
            with self.assertRaises(ValueError):
                load_ablation_upper(path)


if __name__ == "__main__":
    unittest.main()
