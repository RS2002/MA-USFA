from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from paper_aligned_followup.legacy_tasks import (
    build_legacy_protocol, old_heterogeneous_weights, old_homogeneous_weight,
    prepare_from_parent,
)


class LegacyTaskTests(unittest.TestCase):
    def test_homogeneous_exact_old_seed(self):
        expected = np.asarray([-.68026969, -.15584257, -1.44482779, 3.67480179, 0.], dtype=np.float32)
        np.testing.assert_allclose(old_homogeneous_weight(11), expected, rtol=0, atol=2e-7)

    def test_heterogeneous_shape_and_rng(self):
        value = old_heterogeneous_weights(196, 11)
        self.assertEqual(value.shape, (196, 5))
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(value, old_heterogeneous_weights(196, 11))
        self.assertTrue(np.all(value[:, 4] == 0))

    def test_protocol_keeps_old_validation_separate_from_train(self):
        raw = build_legacy_protocol(["a", "b"], np.zeros((5, 5), dtype=np.float32))
        self.assertEqual([x["id"] for x in raw["tasks"]["validation"]],
                         ["legacy_homo_seed11", "legacy_heterogeneous_seed11"])
        self.assertEqual(raw["traffic_seeds"]["validation"], [10_000_000])
        self.assertNotEqual(raw["tasks"]["train"][0]["weights"], raw["tasks"]["validation"][0]["weights"])

    def test_prepare_does_not_modify_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root / "parent"; parent.mkdir()
            net, routes = parent / "x.net.xml", parent / "x.rou.xml"
            net.write_text("net", encoding="utf-8"); routes.write_text("route", encoding="utf-8")
            protocol = parent / "protocol.json"
            protocol.write_text(json.dumps({"version":"ma-usfa-traffic-task-v2",
                "source_anchors":np.zeros((5,5)).tolist(),"node_order":["a","b"],
                "tasks":{"train":[{"id":"x","weights":[0,0,0,0,0]}],"validation":[{"id":"y","weights":[1,1,1,1,0]}]},
                "traffic_seeds":{"train":[0],"validation":[1],"test":[2]}}), encoding="utf-8")
            import hashlib
            h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = parent / "manifest.json"
            manifest.write_text(json.dumps({"version":"ma-usfa-tsc-server-manifest-v2","status":"READY_FOR_PILOT",
                "scenario_id":"test","scenario":{"delta_time":10,"horizon_steps":60},
                "files":[{"role":"network","path":net.name,"sha256":h(net)},{"role":"routes","path":routes.name,"sha256":h(routes)}],
                "source_anchors":np.zeros((5,5)).tolist(),"traffic_task_protocol":{"version":"ma-usfa-traffic-task-v2","path":protocol.name,"sha256":h(protocol)},
                "method_contract":{"strict_entry":True,"task_weight_conditioned_sf":True,"frozen_source_library":True,"declared_reference_distribution":True,"zero_evaluation_updates":True,"held_out_validation":True,"k1_entry_consistency":True},
                "protocol":{"protocol_id":"p","code_sha256":"x","data_manifest_sha256":"x","train_split_sha256":"x","validation_split_sha256":"x","training_seeds":[0],"traffic_seeds":{"train":[0],"validation":[1],"test":[2]}},
                "resource_limits":{"threads":1,"max_wall_seconds":60,"max_rss_gib":4,"min_free_disk_gib":1},"test_accessed_before_formal_run":False}), encoding="utf-8")
            before = manifest.read_bytes(); result = prepare_from_parent(manifest, root / "new")
            self.assertTrue(Path(result["manifest"]).is_file()); self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
