# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the pose engine abstraction (Phase 5). No gnina/rdkit/network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose import (  # noqa: E402
    BasePosePredictor,
    DiffDockPredictor,
    DiffDockUnavailable,
    GninaPredictor,
    HybridValidationPredictor,
    build_pose_predictor,
)
from pose.base import PoseResult  # noqa: E402


MOLS = [("mol_001", "CCO"), ("mol_002", "CCN"), ("mol_003", "CCC"), ("mol_004", "CCF")]


def fake_gnina_rows(names, affinities):
    rows = []
    for n, a in zip(names, affinities):
        rows.append({
            "ligand": n, "affinity_kcal_mol": a, "skor_kaynagi": "gnina_accurate",
            "docking_success": a is not None, "docking_error": None,
            "fast_affinity_kcal_mol": a, "accurate_affinity_kcal_mol": a,
        })
    return rows


class TestGninaPredictor(unittest.TestCase):
    def test_two_stage_passthrough_rows_and_scores(self):
        captured = {}

        def two_stage(molecules, top_fraction=0.1, **kw):
            captured["top_fraction"] = top_fraction
            captured["profile"] = kw.get("profile")
            names = [n for n, _ in molecules]
            return fake_gnina_rows(names, [-8.0, -7.0, None, -6.5]), {"gnina_processes": 2}

        pred = GninaPredictor(two_stage_fn=two_stage, single_mode_fn=lambda *a, **k: ([], {}),
                              profile="final", top_fraction=0.2, log_fn=lambda *_: None)
        result = pred.predict_pose(MOLS, receptor="r.pdb", center=(0, 0, 0), size=(20, 20, 20))
        self.assertIsInstance(result, PoseResult)
        self.assertEqual(result.engine, "gnina")
        self.assertEqual(len(result.rows), 4)
        # Rows are the exact gnina rows (behavior-neutral).
        self.assertEqual(result.rows[0]["affinity_kcal_mol"], -8.0)
        self.assertEqual(captured["top_fraction"], 0.2)
        self.assertEqual(captured["profile"], "final")
        best = result.best()
        self.assertEqual(best.ligand, "mol_001")

    def test_single_mode_routing(self):
        seen = {}

        def single(molecules, mode="fast", **kw):
            seen["mode"] = mode
            return fake_gnina_rows([n for n, _ in molecules], [-5.0] * len(molecules)), {}

        pred = GninaPredictor(single_mode_fn=single, two_stage_fn=lambda *a, **k: ([], {}),
                              docking_mode="sadece_fast", log_fn=lambda *_: None)
        pred.predict_pose(MOLS, receptor="r", center=(0, 0, 0), size=(1, 1, 1))
        self.assertEqual(seen["mode"], "fast")


class TestDiffDockPredictor(unittest.TestCase):
    def test_runner_confidences(self):
        def runner(molecules, **kw):
            return {"mol_001": 0.8, "mol_002": -0.2}  # mol_003/004 missing

        pred = DiffDockPredictor(runner=runner, log_fn=lambda *_: None)
        result = pred.predict_pose(MOLS, receptor="r")
        self.assertEqual(result.engine, "diffdock")
        by = {s.ligand: s for s in result.scores}
        self.assertEqual(by["mol_001"].confidence, 0.8)
        self.assertTrue(by["mol_001"].success)
        self.assertFalse(by["mol_003"].success)  # no score
        self.assertIsNone(by["mol_001"].affinity_kcal_mol)

    def test_loads_from_csv_via_loader(self):
        pred = DiffDockPredictor(
            results_csv="/some/diffdock_results.csv",
            loader=lambda path: {"mol_001": 0.5},
            log_fn=lambda *_: None,
        )
        # patch existence check by pointing find at an injected loader path
        pred._find_results_csv = lambda out_dir: Path("/some/diffdock_results.csv")
        result = pred.predict_pose(MOLS)
        self.assertEqual({s.ligand: s.confidence for s in result.scores}["mol_001"], 0.5)

    def test_unavailable_raises(self):
        pred = DiffDockPredictor(log_fn=lambda *_: None)
        with self.assertRaises(DiffDockUnavailable):
            pred.predict_pose(MOLS, out_dir="/nonexistent/dir")


class TestHybridValidation(unittest.TestCase):
    def test_diffdock_then_gnina_confirmation_and_merge(self):
        # DiffDock likes mol_001 (0.9) and mol_002 (0.5); low for others.
        def dd_runner(molecules, **kw):
            return {"mol_001": 0.9, "mol_002": 0.5, "mol_003": -0.8, "mol_004": -0.9}

        confirmed = {}

        def two_stage(molecules, top_fraction=0.1, **kw):
            names = [n for n, _ in molecules]
            confirmed["names"] = names
            # GNINA confirms strong affinity for the top subset.
            return fake_gnina_rows(names, [-8.0] * len(names)), {"gnina_processes": 2}

        diffdock = DiffDockPredictor(runner=dd_runner, log_fn=lambda *_: None)
        gnina = GninaPredictor(two_stage_fn=two_stage, single_mode_fn=lambda *a, **k: ([], {}),
                               log_fn=lambda *_: None)
        hy = HybridValidationPredictor(diffdock=diffdock, gnina=gnina,
                                       top_fraction=0.5, log_fn=lambda *_: None)
        result = hy.predict_pose(MOLS, receptor="r", center=(0, 0, 0), size=(20, 20, 20))
        self.assertEqual(result.engine, "hybrid_validation")
        # Top 50% by confidence = mol_001, mol_002 → those get GNINA.
        self.assertEqual(set(confirmed["names"]), {"mol_001", "mol_002"})
        by = {s.ligand: s for s in result.scores}
        # mol_001: affinity -8 (<=-7) AND confidence 0.9 (>=0) → strong.
        self.assertEqual(by["mol_001"].extra["genel_guven_durumu"], "GÜÇLÜ ADAY")
        self.assertTrue(by["mol_001"].extra["confirmed_by_gnina"])
        # mol_003 not confirmed by GNINA, weak confidence → weak.
        self.assertFalse(by["mol_003"].extra["confirmed_by_gnina"])

    def test_propagates_diffdock_unavailable(self):
        hy = HybridValidationPredictor(
            diffdock=DiffDockPredictor(log_fn=lambda *_: None),
            gnina=GninaPredictor(two_stage_fn=lambda *a, **k: ([], {}),
                                 single_mode_fn=lambda *a, **k: ([], {})),
            log_fn=lambda *_: None,
        )
        with self.assertRaises(DiffDockUnavailable):
            hy.predict_pose(MOLS, out_dir="/nonexistent")


class TestFactory(unittest.TestCase):
    def test_default_is_gnina(self):
        self.assertIsInstance(build_pose_predictor(), GninaPredictor)
        self.assertIsInstance(build_pose_predictor("gnina"), GninaPredictor)

    def test_diffdock_and_hybrid(self):
        self.assertIsInstance(build_pose_predictor("diffdock"), DiffDockPredictor)
        self.assertIsInstance(build_pose_predictor("hybrid"), HybridValidationPredictor)
        self.assertIsInstance(build_pose_predictor("hybrid_validation"), HybridValidationPredictor)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_pose_predictor("nope")

    def test_all_are_base(self):
        for cls in (GninaPredictor, DiffDockPredictor, HybridValidationPredictor):
            self.assertTrue(issubclass(cls, BasePosePredictor))


if __name__ == "__main__":
    unittest.main()
