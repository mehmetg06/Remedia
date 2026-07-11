"""Regression tests for receptor preparation and NaN docking failures."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gnina_engine as ge  # noqa: E402
import receptor_prep as rp  # noqa: E402


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def fake_prepare(smiles, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.sdf"
    path.write_text(f"fake:{smiles}")
    return path


class TestReceptorPreparation(unittest.TestCase):
    def test_existing_valid_pdbqt_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.pdbqt"
            path.write_text(
                "ATOM      1  C   ALA A   1      10.000  10.000  10.000  0.00  0.00     0.123 C\n"
                "ATOM      2  N   ALA A   1      11.000  10.000  10.000  0.00  0.00    -0.123 N\n"
            )
            self.assertEqual(rp.prepare_receptor_pdbqt(path), path.resolve())

    def test_missing_obabel_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "r.pdb"
            pdb.write_text("ATOM      1  C   ALA A   1      10.000  10.000  10.000\n")
            with patch("receptor_prep.shutil.which", return_value=None):
                with self.assertRaises(rp.ReceptorPreparationError) as ctx:
                    rp.prepare_receptor_pdbqt(pdb)
            self.assertIn("obabel", str(ctx.exception))


class TestFailFast(unittest.TestCase):
    def test_all_failed_scores_raise_instead_of_returning_nan_table(self):
        def failed_batch(receptor, prepared_ligands, center, size, mode, **kwargs):
            return [
                ge.DockResult(name, mode, None, 0.1, False, "CUDA initialization failed")
                for name in prepared_ligands
            ]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ge.GninaScreeningError) as ctx:
                ge.run_single_mode_screening(
                    [("a", "C"), ("b", "CC")],
                    "missing-test-receptor.pdb",
                    (0, 0, 0),
                    (20, 20, 20),
                    out_dir=tmp,
                    prepare_fn=fake_prepare,
                    batch_dock_fn=failed_batch,
                    log_fn=lambda *_args, **_kwargs: None,
                )
        self.assertIn("CUDA initialization failed", str(ctx.exception))

    def test_partial_failure_rows_expose_status_and_error(self):
        results = [
            ge.DockResult("ok", "fast", -7.0, 0.2, True),
            ge.DockResult("bad", "fast", None, 0.2, False, "ligand parse failed"),
        ]
        rows = ge._rows(results, ge.MODE_FAST)
        by_name = {row["ligand"]: row for row in rows}
        self.assertTrue(by_name["ok"]["docking_success"])
        self.assertFalse(by_name["bad"]["docking_success"])
        self.assertEqual(by_name["bad"]["docking_error"], "ligand parse failed")
        self.assertIsNone(by_name["bad"]["affinity_kcal_mol"])


if __name__ == "__main__":
    unittest.main()
