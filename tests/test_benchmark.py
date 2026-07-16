# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the benchmark framework (Phase 8). No rdkit/gnina/network."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark  # noqa: E402
from generators.base import GenerationResult  # noqa: E402
from pose.base import PoseResult, PoseScore  # noqa: E402


class FakeGen:
    def __init__(self, name, smiles, fail=False):
        self.name = name
        self._smiles = smiles
        self._fail = fail

    def generate(self, target=None, n=20, seeds=None, **kw):
        if self._fail:
            raise RuntimeError(f"{self.name} unavailable")
        return GenerationResult(smiles=self._smiles[:n], source=self.name, requested=n)


class FakePose:
    def __init__(self, name, scores, fail=False):
        self.name = name
        self._scores = scores
        self._fail = fail

    def predict_pose(self, molecules, **kw):
        if self._fail:
            raise RuntimeError(f"{self.name} unavailable")
        return PoseResult(engine=self.name, scores=self._scores, rows=[])


class TestGeneratorBenchmark(unittest.TestCase):
    def test_metrics_and_error_row(self):
        gens = {
            "reinvent4": FakeGen("reinvent4", ["CCO", "CCN", "CCC", "c1ccccc1"]),
            "molmim": FakeGen("molmim", [], fail=True),  # e.g. no API key
        }
        # deterministic ADMET: pass if length<=3
        report = benchmark.run_generator_benchmark(
            gens, target="P00918", n=4,
            admet_fn=lambda s, n="": len(s) <= 3,
        )
        by = {r["name"]: r for r in report.rows}
        self.assertEqual(by["reinvent4"]["produced"], 4)
        self.assertGreater(by["reinvent4"]["diversity_score"], 0)
        self.assertIsNotNone(by["reinvent4"]["admet_pass_rate"])
        self.assertEqual(by["reinvent4"]["error"], "")
        # failed generator recorded, not raised
        self.assertTrue(by["molmim"]["error"])
        self.assertEqual(by["molmim"]["produced"], 0)

    def test_winner_prefers_diversity(self):
        gens = {
            "a": FakeGen("a", ["CCO", "CCO", "CCO", "CCO"]),   # low diversity
            "b": FakeGen("b", ["CCO", "CCN", "CCC", "c1ccccc1CN"]),  # high diversity
        }
        report = benchmark.run_generator_benchmark(gens, n=4, admet_fn=lambda s, n="": True)
        self.assertEqual(report.winner(), "b")


class TestPoseBenchmark(unittest.TestCase):
    def _scores(self, affs):
        return [PoseScore(ligand=f"m{i}", affinity_kcal_mol=a, success=a is not None,
                          source="x") for i, a in enumerate(affs)]

    def test_quality_and_success_rate(self):
        preds = {
            "gnina": FakePose("gnina", self._scores([-9.0, -7.0, None])),
            "diffdock": FakePose("diffdock", [
                PoseScore(ligand="m0", confidence=0.8, success=True, source="diffdock"),
                PoseScore(ligand="m1", confidence=0.2, success=True, source="diffdock"),
            ]),
        }
        report = benchmark.run_pose_benchmark(preds, [("m0", "CCO")])
        by = {r["name"]: r for r in report.rows}
        self.assertEqual(by["gnina"]["best_affinity"], -9.0)
        self.assertAlmostEqual(by["gnina"]["success_rate"], 2 / 3, places=3)
        self.assertEqual(by["diffdock"]["mean_confidence"], 0.5)

    def test_failed_engine_recorded(self):
        preds = {"diffdock": FakePose("diffdock", [], fail=True)}
        report = benchmark.run_pose_benchmark(preds, [("m0", "CCO")])
        self.assertTrue(report.rows[0]["error"])

    def test_winner_prefers_best_affinity(self):
        preds = {
            "gnina": FakePose("gnina", self._scores([-9.5, -8.0])),
            "weak": FakePose("weak", self._scores([-5.0, -4.0])),
        }
        report = benchmark.run_pose_benchmark(preds, [("m0", "CCO")])
        self.assertEqual(report.winner(), "gnina")


class TestExport(unittest.TestCase):
    def test_export_writes_csv_json_md(self):
        gens = {"a": FakeGen("a", ["CCO", "CCN"])}
        report = benchmark.run_generator_benchmark(gens, n=2, admet_fn=lambda s, n="": True)
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.export(tmp)
            for key in ("csv", "json", "markdown"):
                self.assertTrue(Path(paths[key]).exists())
            data = json.loads(Path(paths["json"]).read_text())
            self.assertEqual(data["kind"], "generators")
            self.assertIn("Benchmark", Path(paths["markdown"]).read_text())


if __name__ == "__main__":
    unittest.main()
