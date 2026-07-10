# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
test_gnina_engine.py
gnina_engine.py için birim testleri. GNINA binary'si ve GPU GEREKTİRMEZ —
subprocess.run mock'lanır, böylece komut inşası ve iki-aşamalı pipeline
mantığı GPU'suz bir CI/sandbox ortamında da doğrulanabilir.

Gerçek GNINA ile ölçülen süre/skor karşılaştırması için (GPU gerektirir):
    notebooks/remedia_pipeline.ipynb Hücre 5'teki "mod karşılaştırma" hücresini
    Colab'da çalıştır, ya da:
    python src/gnina_engine.py --mode compare ...

Çalıştırma:
    python -m unittest tests.test_gnina_engine -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gnina_engine as ge  # noqa: E402


def fake_prepare(smiles, name, out_dir):
    """rdkit gerektirmeyen sahte SMILES->SDF hazırlama: sadece bir dosya yolu üretir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.sdf"
    p.write_text("fake sdf")
    return p


class FakeProc:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr


class TestBuildGninaCommand(unittest.TestCase):
    def test_fast_mode_flags(self):
        cmd = ge.build_gnina_command(
            "/usr/local/bin/gnina", "r.pdb", "l.sdf", (1.0, 2.0, 3.0), (20, 20, 20),
            mode="fast", out_path="out.sdf",
        )
        self.assertIn("--cnn", cmd)
        self.assertEqual(cmd[cmd.index("--cnn") + 1], "fast")
        self.assertEqual(cmd[cmd.index("--cnn_scoring") + 1], "rescore")
        self.assertEqual(cmd[cmd.index("--exhaustiveness") + 1], "4")
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "1")

    def test_accurate_mode_flags(self):
        cmd = ge.build_gnina_command(
            "/usr/local/bin/gnina", "r.pdb", "l.sdf", (1.0, 2.0, 3.0), (20, 20, 20),
            mode="accurate", out_path="out.sdf",
        )
        # accurate modda --cnn HİÇ verilmemeli -> varsayılan ensemble kullanılsın
        self.assertNotIn("--cnn", cmd)
        self.assertEqual(cmd[cmd.index("--cnn_scoring") + 1], "rescore")
        self.assertEqual(cmd[cmd.index("--exhaustiveness") + 1], "16")
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "9")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ge.build_gnina_command("gnina", "r.pdb", "l.sdf", (0, 0, 0), (20, 20, 20), mode="turbo")


class TestParseAffinity(unittest.TestCase):
    def test_parses_from_stdout_table(self):
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -7.234          0          0\n"
            "   2       -6.100          1          2\n"
        )
        aff = ge.parse_affinity(Path("does_not_exist.sdf"), stdout)
        self.assertAlmostEqual(aff, -7.234)

    def test_returns_none_when_unparsable(self):
        self.assertIsNone(ge.parse_affinity(Path("does_not_exist.sdf"), "no useful output"))


class TestDockWithGnina(unittest.TestCase):
    def test_success_uses_correct_mode_and_returns_affinity(self):
        with patch("gnina_engine.subprocess.run") as mock_run:
            mock_run.return_value = FakeProc(stdout="   1       -8.456          0          0\n")
            result = ge.dock_with_gnina(
                "r.pdb", "l.sdf", (1, 2, 3), (20, 20, 20), mode="accurate",
                ligand_name="mol_1", out_dir="/tmp/gnina_test_out",
            )
        self.assertTrue(result.success)
        self.assertEqual(result.mode, "accurate")
        self.assertAlmostEqual(result.affinity_kcal_mol, -8.456)
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--cnn", cmd)  # accurate

    def test_failure_when_gnina_missing(self):
        with patch("gnina_engine.subprocess.run", side_effect=FileNotFoundError("no gnina")):
            result = ge.dock_with_gnina(
                "r.pdb", "l.sdf", (1, 2, 3), (20, 20, 20), mode="fast",
                ligand_name="mol_1", out_dir="/tmp/gnina_test_out",
            )
        self.assertFalse(result.success)
        self.assertIsNone(result.affinity_kcal_mol)
        self.assertIn("GNINA çalıştırılamadı", result.error)

    def test_unparsable_output_is_reported_as_failure(self):
        with patch("gnina_engine.subprocess.run") as mock_run:
            mock_run.return_value = FakeProc(stdout="", stderr="segfault")
            result = ge.dock_with_gnina(
                "r.pdb", "l.sdf", (1, 2, 3), (20, 20, 20), mode="fast",
                ligand_name="mol_1", out_dir="/tmp/gnina_test_out",
            )
        self.assertFalse(result.success)
        self.assertIn("segfault", result.error)


class TestSelectTopCandidates(unittest.TestCase):
    def _results(self, pairs):
        return [ge.DockResult(name, "fast", aff, 1.0, aff is not None) for name, aff in pairs]

    def test_top_fraction_rounds_up(self):
        # 7 basarili sonuc, top_fraction=0.2 -> ceil(7*0.2)=2
        results = self._results([(f"m{i}", -float(i)) for i in range(7)])
        top = ge.select_top_candidates(results, top_fraction=0.2)
        self.assertEqual(len(top), 2)
        self.assertEqual([r.ligand for r in top], ["m6", "m5"])  # en negatif = en iyi

    def test_top_n_overrides_fraction(self):
        results = self._results([(f"m{i}", -float(i)) for i in range(10)])
        top = ge.select_top_candidates(results, top_n=3, top_fraction=0.9)
        self.assertEqual(len(top), 3)

    def test_failed_results_excluded(self):
        results = self._results([("good", -5.0), ("bad", None)])
        top = ge.select_top_candidates(results, top_n=5)
        self.assertEqual([r.ligand for r in top], ["good"])

    def test_empty_input(self):
        self.assertEqual(ge.select_top_candidates([], top_n=3), [])


class TestRunTwoStageScreening(unittest.TestCase):
    def test_two_stage_merges_accurate_over_fast_for_top_only(self):
        molecules = [(f"m{i}", "C") for i in range(5)]
        # fast skorlari: m0..m4 -> -1,-2,-3,-4,-5 (daha negatif = daha iyi -> m4 en iyi)
        fast_scores = {"m0": -1.0, "m1": -2.0, "m2": -3.0, "m3": -4.0, "m4": -5.0}
        accurate_scores = {"m4": -5.5, "m3": -3.8}  # top_n=2 varsayimiyla sadece bunlar accurate'e girer

        def fake_dock(receptor, ligand_file, center, size, mode, ligand_name=None, **kw):
            if mode == "fast":
                aff = fast_scores[ligand_name]
            else:
                aff = accurate_scores[ligand_name]
            return ge.DockResult(ligand_name, mode, aff, 2.0 if mode == "fast" else 8.0, True)

        with patch("gnina_engine.dock_with_gnina", side_effect=fake_dock):
            rows, stage_info = ge.run_two_stage_screening(
                molecules, "r.pdb", (0, 0, 0), (20, 20, 20),
                top_n=2, out_dir="/tmp/gnina_test_two_stage",
                log_fn=lambda *a, **k: None, prepare_fn=fake_prepare,
            )

        by_ligand = {r["ligand"]: r for r in rows}

        # Top-2 (m4, m3) accurate skoru almali ve skor_kaynagi=gnina_accurate olmali
        self.assertEqual(by_ligand["m4"]["affinity_kcal_mol"], -5.5)
        self.assertEqual(by_ligand["m4"]["skor_kaynagi"], ge.KAYNAK_ACCURATE)
        self.assertEqual(by_ligand["m3"]["affinity_kcal_mol"], -3.8)
        self.assertEqual(by_ligand["m3"]["skor_kaynagi"], ge.KAYNAK_ACCURATE)

        # Digerleri (m0,m1,m2) SADECE fast skoru almali, accurate ile dokunulmamis
        for name in ("m0", "m1", "m2"):
            self.assertEqual(by_ligand[name]["affinity_kcal_mol"], fast_scores[name])
            self.assertEqual(by_ligand[name]["skor_kaynagi"], ge.KAYNAK_FAST)
            self.assertIsNone(by_ligand[name]["accurate_affinity_kcal_mol"])

        self.assertEqual(set(stage_info["top_ligands"]), {"m3", "m4"})
        self.assertEqual(len(stage_info["fast"]), 5)
        self.assertEqual(len(stage_info["accurate"]), 2)

    def test_prep_failure_marked_and_skipped(self):
        molecules = [("bad", "not_a_smiles")]

        def fake_prep_fail(smiles, name, out_dir):
            return None

        with patch("gnina_engine.dock_with_gnina") as mock_dock:
            rows, stage_info = ge.run_two_stage_screening(
                molecules, "r.pdb", (0, 0, 0), (20, 20, 20),
                top_n=1, out_dir="/tmp/gnina_test_two_stage_fail",
                log_fn=lambda *a, **k: None, prepare_fn=fake_prep_fail,
            )
        mock_dock.assert_not_called()
        self.assertEqual(rows[0]["ligand"], "bad")
        self.assertIsNone(rows[0]["affinity_kcal_mol"])
        self.assertEqual(rows[0]["skor_kaynagi"], ge.KAYNAK_FAST)


class TestRunSingleModeScreening(unittest.TestCase):
    def test_fast_only_fills_fast_columns(self):
        molecules = [("m1", "C"), ("m2", "CC")]

        def fake_dock(receptor, ligand_file, center, size, mode, ligand_name=None, **kw):
            self.assertEqual(mode, "fast")
            return ge.DockResult(ligand_name, mode, -3.0, 1.5, True)

        with patch("gnina_engine.dock_with_gnina", side_effect=fake_dock):
            rows, results_by_mode = ge.run_single_mode_screening(
                molecules, "r.pdb", (0, 0, 0), (20, 20, 20), mode="fast",
                out_dir="/tmp/gnina_test_single", log_fn=lambda *a, **k: None,
                prepare_fn=fake_prepare,
            )

        for row in rows:
            self.assertEqual(row["skor_kaynagi"], ge.KAYNAK_FAST)
            self.assertEqual(row["fast_affinity_kcal_mol"], -3.0)
            self.assertIsNone(row["accurate_affinity_kcal_mol"])
            self.assertIsNone(row["accurate_seconds"])
        self.assertEqual(len(results_by_mode["fast"]), 2)

    def test_accurate_only_fills_accurate_columns(self):
        with patch("gnina_engine.dock_with_gnina") as mock_dock:
            mock_dock.return_value = ge.DockResult("m1", "accurate", -9.1, 12.0, True)
            rows, _ = ge.run_single_mode_screening(
                [("m1", "C")], "r.pdb", (0, 0, 0), (20, 20, 20), mode="accurate",
                out_dir="/tmp/gnina_test_single2", log_fn=lambda *a, **k: None,
                prepare_fn=fake_prepare,
            )
        self.assertEqual(rows[0]["skor_kaynagi"], ge.KAYNAK_ACCURATE)
        self.assertEqual(rows[0]["accurate_affinity_kcal_mol"], -9.1)
        self.assertIsNone(rows[0]["fast_affinity_kcal_mol"])


class TestBenchmarkFastVsAccurate(unittest.TestCase):
    def test_measures_time_and_score_gap(self):
        molecules = [("a", "C"), ("b", "CC")]

        def fake_dock(receptor, ligand_file, center, size, mode, ligand_name=None, **kw):
            if mode == "fast":
                elapsed = 1.0
                aff = -5.0 if ligand_name == "a" else -4.0
            else:
                elapsed = 9.0
                aff = -6.2 if ligand_name == "a" else -4.1
            return ge.DockResult(ligand_name, mode, aff, elapsed, True)

        with patch("gnina_engine.dock_with_gnina", side_effect=fake_dock):
            rows, summary = ge.benchmark_fast_vs_accurate(
                molecules, "r.pdb", (0, 0, 0), (20, 20, 20),
                out_dir="/tmp/gnina_test_bench",
                log_fn=lambda *a, **k: None, prepare_fn=fake_prepare,
            )

        self.assertEqual(len(rows), 2)
        row_a = next(r for r in rows if r["ligand"] == "a")
        self.assertAlmostEqual(row_a["skor_farki"], 1.2)
        self.assertAlmostEqual(row_a["hiz_orani"], 9.0)

        self.assertEqual(summary["n_karsilastirilabilir"], 2)
        self.assertAlmostEqual(summary["fast_ortalama_sn"], 1.0)
        self.assertAlmostEqual(summary["accurate_ortalama_sn"], 9.0)
        self.assertAlmostEqual(summary["hiz_orani_ortalama"], 9.0)
        # (1.2 + 0.1) / 2 = 0.65
        self.assertAlmostEqual(summary["skor_farki_ortalama"], 0.65)


if __name__ == "__main__":
    unittest.main()
