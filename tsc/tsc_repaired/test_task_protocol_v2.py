"""Task/traffic split and deployment-input checks; no SUMO or training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .config import REQUIRED_CONTRACT
from .task_protocol_v2 import (VERSION, default_anchors, load_protocol,
                               source_sampling_probabilities,
                               validate_task_weights)


class TrafficTaskProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.protocol_path = self.root / "tasks.json"
        self.manifest_path = self.root / "manifest.json"
        self.weights = [-1.0, -0.3, -1.5, 3.0, 0.0]
        self.protocol = {
            "version": VERSION,
            "node_order": ["a", "b"],
            "source_anchors": default_anchors().tolist(),
            "source_sampling": {
                "temperature": 0.5, "score": "anchor_dot_task", "resample": "each_step"},
            "tasks": {
                "train": [{"id": "train-default", "weights": self.weights}],
                "validation": [{"id": "validation-speed", "weights": [-1.0, -0.3, -1.5, 4.0, 0.0]}],
            },
            "traffic_seeds": {"train": [0, 1], "validation": [10000, 10001], "test": [20000]},
            # This path deliberately does not exist: default loads must not touch it.
            "sealed_test_tasks": {"path": "unopened_test.json", "sha256": "0" * 64},
        }
        self.manifest = {
            "version": "ma-usfa-tsc-server-manifest-v1",
            "status": "READY_FOR_PILOT", "scenario_id": "fake-test-only",
            "method_contract": {key: True for key in REQUIRED_CONTRACT},
            "protocol": {"protocol_id": "v2-unit-test", "code_sha256": "x",
                         "data_manifest_sha256": "x", "train_split_sha256": "x",
                         "validation_split_sha256": "y", "test_split_sha256": "z",
                         "training_seeds": [0], "traffic_seeds": [0]},
            "resource_limits": {"threads": 1, "max_wall_seconds": 10,
                                "max_rss_gib": 1, "min_free_disk_gib": 1},
            "test_accessed_before_formal_run": False,
        }

    def write(self, protocol=None, manifest=None):
        raw = self.protocol if protocol is None else protocol
        self.protocol_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        parent = copy.deepcopy(self.manifest if manifest is None else manifest)
        parent["traffic_task_protocol"] = {
            "version": VERSION, "path": "tasks.json",
            "sha256": hashlib.sha256(self.protocol_path.read_bytes()).hexdigest(),
        }
        self.manifest_path.write_text(json.dumps(parent), encoding="utf-8")
        return self.manifest_path

    def test_real_split_broadcast_and_sealed_test_is_not_opened(self):
        protocol = load_protocol(self.write(), node_ids=("a", "b"))
        self.assertEqual(protocol.version, VERSION)
        self.assertEqual(protocol.fingerprint, hashlib.sha256(self.protocol_path.read_bytes()).hexdigest())
        train, validation = protocol.tasks("train"), protocol.tasks("validation")
        self.assertEqual(train[0]["weights"].shape, (2, 5))
        np.testing.assert_array_equal(train[0]["weights"][0], np.asarray(self.weights, dtype=np.float32))
        self.assertFalse(np.array_equal(train[0]["weights"], validation[0]["weights"]))
        self.assertFalse(set(protocol.traffic_seeds("train")) & set(protocol.traffic_seeds("validation")))
        self.assertFalse(protocol.test_objectives_loaded)
        with self.assertRaisesRegex(RuntimeError, "cannot run stage 'formal'"):
            protocol.tasks("test")
        self.assertFalse(protocol.test_objectives_loaded)
        self.assertFalse((self.root / "unopened_test.json").exists())
        train[0]["weights"][0, 0] = 999.0
        self.assertEqual(float(protocol.tasks("train")[0]["weights"][0, 0]), -1.0)

    def test_without_runtime_ids_vector_is_retained(self):
        protocol = load_protocol(self.write())
        self.assertEqual(protocol.tasks("train")[0]["weights"].shape, (5,))

    def test_identical_weights_across_split_are_rejected(self):
        raw = copy.deepcopy(self.protocol)
        raw["tasks"]["validation"][0]["weights"] = [self.weights, self.weights]
        with self.assertRaisesRegex(ValueError, "duplicate task weights"):
            load_protocol(self.write(raw))
        with self.assertRaisesRegex(ValueError, "duplicate task weights"):
            load_protocol(self.write(raw), node_ids=("a", "b"))

    def test_duplicate_ids_and_traffic_seeds_are_rejected(self):
        raw = copy.deepcopy(self.protocol)
        raw["tasks"]["validation"][0]["id"] = "train-default"
        with self.assertRaisesRegex(ValueError, "duplicate task id"):
            load_protocol(self.write(raw))
        raw = copy.deepcopy(self.protocol)
        raw["traffic_seeds"]["validation"][0] = 0
        with self.assertRaisesRegex(ValueError, "overlaps"):
            load_protocol(self.write(raw))
        raw = copy.deepcopy(self.protocol)
        raw["traffic_seeds"]["test"][0] = 1
        with self.assertRaisesRegex(ValueError, "overlaps"):
            load_protocol(self.write(raw))

    def test_hash_tampering_and_node_reordering_are_rejected(self):
        path = self.write()
        self.protocol_path.write_text(self.protocol_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_protocol(path)
        with self.assertRaisesRegex(ValueError, "node_order differs"):
            load_protocol(self.write(), node_ids=("b", "a"))

    def test_shape_nonfinite_anchor_mismatch_and_inline_test_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_task_weights([self.weights, self.weights], 3)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_task_weights([1, 2, float("nan"), 4, 5], 2)
        raw = copy.deepcopy(self.protocol)
        raw["tasks"]["validation"][0]["weights"] = [self.weights] * 3
        with self.assertRaisesRegex(ValueError, "node_order"):
            load_protocol(self.write(raw))
        manifest = copy.deepcopy(self.manifest)
        manifest["source_anchors"] = [[1, 1, 1, 1, 1]]
        with self.assertRaisesRegex(ValueError, "source_anchors differ"):
            load_protocol(self.write(manifest=manifest))
        raw = copy.deepcopy(self.protocol)
        raw["tasks"]["test"] = []
        with self.assertRaisesRegex(ValueError, "inline test"):
            load_protocol(self.write(raw))

    def test_declared_task_sampling_is_stable_and_uses_signed_dot_product(self):
        weights = np.asarray([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]], dtype=np.float32)
        anchors = np.asarray([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]], dtype=np.float32)
        probabilities = source_sampling_probabilities(weights, anchors, temperature=0.5)
        np.testing.assert_allclose(probabilities.sum(-1), 1.0)
        self.assertGreater(probabilities[0, 0], probabilities[0, 1])
        self.assertGreater(probabilities[1, 1], probabilities[1, 0])
        np.testing.assert_allclose(probabilities[0], probabilities[1, ::-1])
        huge = source_sampling_probabilities(weights * 1e30, anchors * 1e30, 1e-100)
        self.assertTrue(np.isfinite(huge).all())
        np.testing.assert_allclose(huge.sum(-1), 1.0)
        for temperature in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                source_sampling_probabilities(weights, anchors, temperature)

    def test_sampling_temperature_and_continuation_must_be_declared(self):
        raw = copy.deepcopy(self.protocol)
        del raw["source_sampling"]["temperature"]
        with self.assertRaisesRegex(ValueError, "temperature"):
            load_protocol(self.write(raw))
        raw = copy.deepcopy(self.protocol)
        raw["source_sampling"]["resample"] = "per_episode"
        with self.assertRaisesRegex(ValueError, "each_step"):
            load_protocol(self.write(raw))

    def test_formal_gate_then_hash_bound_test_loading(self):
        raw = copy.deepcopy(self.protocol)
        test_path = self.root / "sealed.json"
        test_path.write_text(json.dumps({"version": VERSION,
            "tasks": [{"id": "test-unseen", "weights": [-1, -0.3, -1.5, 5, 0]}]}))
        raw["sealed_test_tasks"] = {"path": "sealed.json",
            "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
        manifest = copy.deepcopy(self.manifest)
        manifest["status"] = "READY_FOR_FORMAL"
        protocol = load_protocol(self.write(raw, manifest), node_ids=("a", "b"))
        self.assertFalse(protocol.test_objectives_loaded)
        rows = protocol.tasks("test")
        self.assertEqual(rows[0]["id"], "test-unseen")
        self.assertTrue(protocol.test_objectives_loaded)
        self.assertEqual(rows[0]["weights"].shape, (2, 5))


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TrafficTaskProtocolTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
