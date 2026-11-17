"""Release integration contracts; no SUMO process or training is started."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run as release


def settings():
    return dict(source_seed=0, upper_seed=100, source_episodes=5,
                reference_episodes=20, sf_steps=200, upper_episodes=30,
                replays=5, threads=2, max_wall_seconds=43200,
                max_rss_gib=8., min_free_disk_gib=20., device="cpu")


def scene(root):
    path = root / "scene"
    path.mkdir()
    (path / "manhattan_28x7.net.xml").write_text(
        '<net><junction id="a" type="traffic_light"/>'
        '<junction id="b" type="traffic_light"/>'
        '<tlLogic id="a" type="static" programID="0" offset="0"/>'
        '<tlLogic id="b" type="static" programID="0" offset="0"/></net>')
    (path / "manhattan_28x7.rou.xml").write_text('<routes/>')
    return path


class Guard:
    def __init__(self, *args, **kwargs): pass
    def check(self, **kwargs): return {"rss_gib": .01}
    def close(self): pass


class Runtime:
    source_scenario_binding_verified = True
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class ReleasePipelineTests(unittest.TestCase):
    def test_prepare_freezes_train_validation_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = scene(root); out = root / "result"
            manifest, identity = release.prepare(source, out, settings())
            raw = json.loads(manifest.read_text())
            self.assertFalse(raw["formal_evaluation_enabled"])
            self.assertFalse(raw["test_accessed_before_formal_run"])
            self.assertTrue(all(not Path(row["path"]).is_absolute() for row in raw["files"]))
            tasks = json.loads((out / "protocol/tasks.json").read_text())
            self.assertEqual(set(tasks["tasks"]), {"train", "validation"})
            self.assertEqual(len(tasks["tasks"]["train"]), 2)
            self.assertEqual(len(tasks["tasks"]["validation"]), 2)
            self.assertEqual(release.prepare(source, out, settings())[1], identity)
            (source / "manhattan_28x7.rou.xml").write_text('<routes><route id="x" edges="x"/></routes>')
            with self.assertRaisesRegex(RuntimeError, "changed"):
                release.prepare(source, out, settings())

    def test_completed_stage_reuses_hashes_and_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name); artifact = out / "saved.bin"; calls = []
            def action():
                calls.append(1); artifact.write_bytes(b"frozen"); return {"status": "PASS"}
            release._stage("one", out, "f", {"a": 1}, [artifact], action)
            release._stage("one", out, "f", {"a": 1}, [artifact], action)
            self.assertEqual(len(calls), 1)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                release._stage("one", out, "f", {"a": 1}, [artifact], action)

    def test_all_pipeline_receipts_without_sumo_and_no_duplicate_training(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = scene(root); out = root / "run"
            args = SimpleNamespace(scene_dir=str(source), output=str(out), stage="all", **settings())
            calls = []
            def sources(manifest, output, **kwargs):
                calls.append("sources"); output.mkdir()
                for i in range(5): (output / ("source_%03d.pt" % i)).write_bytes(b"source")
                (output / "train_sources_v2.json").write_text('{}')
                return {"status": "PASS"}
            def reference(manifest, sources, output, **kwargs):
                calls.append("reference"); output.write_bytes(b"reference")
                output.with_suffix(".json").write_text('{}'); return {"status": "PASS"}
            def sf(manifest, refs, output, **kwargs):
                calls.append("sf"); output.mkdir(); (output / "task_sf.pt").write_bytes(b"sf")
                (output / "train_task_sf.json").write_text('{}'); return {"status": "PASS"}
            def upper(manifest, sources, sf, output, **kwargs):
                calls.append("upper"); output.mkdir(); (output / "upper_ep030.pt").write_bytes(b"upper")
                (output / "train_budget.json").write_text('{}'); return {"status": "PASS"}
            def evaluate(manifest, sources, sf, checkpoints, output, **kwargs):
                calls.append("evaluate"); output.write_text('{}'); return {"status": "PASS"}
            patches = {
                "tsc_repaired.resources.ResourceGuard": Guard,
                "tsc_repaired.resources.configure_torch_threads": lambda *a: None,
                "tsc_repaired.train_sources_v2.train_sources_v2": sources,
                "tsc_repaired.collect_task_reference.collect_reference_v2": reference,
                "tsc_repaired.train_task_sf.train_task_sf": sf,
                "tsc_repaired.traffic_runtime.FrozenTrafficRuntime": Runtime,
                "development_followup.train_budget.train_budget_upper": upper,
                "paper_aligned_followup.evaluate.evaluate_paper_aligned": evaluate,
            }
            with ExitStack() as stack:
                for target, value in patches.items(): stack.enter_context(patch(target, value))
                report = release.execute(args)
                self.assertEqual(report["status"], "PASS")
                self.assertFalse(report["test_accessed"])
                self.assertEqual(report["evaluation_updates"], 0)
                self.assertEqual(calls, ["sources", "reference", "sf", "upper", "evaluate"])
                release.execute(args)
                self.assertEqual(len(calls), 5)

    def test_failure_before_first_artifact_cannot_restart_silently(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name); calls = []
            def interrupted():
                calls.append(1)
                self.assertTrue((out / "stages/early.json").is_file())
                raise RuntimeError("interrupted before any artifact")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                release._stage("early", out, "same", {}, [out / "never.pt"], interrupted)
            marker = json.loads((out / "stages/early.json").read_text())
            self.assertEqual(marker["status"], "FAILED")
            with self.assertRaisesRegex(RuntimeError, "receipt/input mismatch"):
                release._stage("early", out, "same", {}, [out / "never.pt"], interrupted)
            self.assertEqual(len(calls), 1)

    def test_unbound_sources_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = scene(root); out = root / "run"
            args = SimpleNamespace(scene_dir=str(source), output=str(out), stage="reference", **settings())
            release.prepare(source, out, settings())
            (out / "sources").mkdir()
            for i in range(5): (out / "sources" / ("source_%03d.pt" % i)).write_bytes(b"historical")
            class Unbound(Runtime): source_scenario_binding_verified = False
            with patch("tsc_repaired.resources.ResourceGuard", Guard), \
                 patch("tsc_repaired.resources.configure_torch_threads"), \
                 patch("tsc_repaired.traffic_runtime.FrozenTrafficRuntime", Unbound):
                with self.assertRaisesRegex(ValueError, "scenario-bound"):
                    release.execute(args)
            self.assertFalse((out / "reference_train.npz").exists())


if __name__ == "__main__":
    unittest.main()
