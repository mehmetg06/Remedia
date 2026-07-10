"""Unit tests for the batched GNINA engine; no GPU or GNINA binary required."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gnina_engine as ge  # noqa: E402


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


class TestProfiles(unittest.TestCase):
    def test_balanced_accurate_is_lightweight_default(self):
        cmd = ge.build_gnina_command(
            "gnina", "r.pdb", "batch.sdf", (1, 2, 3), (20, 20, 20),
            mode="accurate",
        )
        self.assertNotIn("--cnn", cmd)
        self.assertEqual(cmd[cmd.index("--exhaustiveness") + 1], "8")
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "3")

    def test_final_profile_keeps_heavy_settings(self):
        cmd = ge.build_gnina_command(
            "gnina", "r.pdb", "batch.sdf", (1, 2, 3), (20, 20, 20),
            mode="accurate", profile="final",
        )
        self.assertEqual(cmd[cmd.index("--exhaustiveness") + 1], "16")
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "9")

    def test_fast_profile_uses_single_fast_cnn(self):
        cmd = ge.build_gnina_command(
            "gnina", "r.pdb", "batch.sdf", (1, 2, 3), (20, 20, 20),
            mode="fast",
        )
        self.assertEqual(cmd[cmd.index("--cnn") + 1], "fast")
        self.assertEqual(cmd[cmd.index("--exhaustiveness") + 1], "4")
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "1")


class TestSingleDockSafety(unittest.TestCase):
    def test_nonzero_return_code_is_failure_and_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "mol_fast_docked.sdf"
            stale.write_text("old result")
            with patch("gnina_engine.subprocess.run") as run:
                run.return_value = FakeProc(stderr="CUDA error", returncode=1)
                result = ge.dock_with_gnina(
                    "r.pdb", Path(tmp) / "mol.sdf", (0, 0, 0), (20, 20, 20),
                    mode="fast", ligand_name="mol", out_dir=tmp,
                )
            self.assertFalse(result.success)
            self.assertIn("CUDA error", result.error)
            self.assertFalse(stale.exists())

    def test_stdout_affinity_still_supported_for_single_ligand(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("gnina_engine.subprocess.run") as run:
                run.return_value = FakeProc(stdout="   1       -7.125  0  0\n")
                result = ge.dock_with_gnina(
                    "r.pdb", Path(tmp) / "mol.sdf", (0, 0, 0), (20, 20, 20),
                    mode="fast", ligand_name="mol", out_dir=tmp,
                )
            self.assertTrue(result.success)
            self.assertAlmostEqual(result.affinity_kcal_mol, -7.125)


class TestSelection(unittest.TestCase):
    def test_default_top_fraction_is_ten_percent(self):
        results = [ge.DockResult(f"m{i}", "fast", -float(i), 1, True) for i in range(20)]
        top = ge.select_top_candidates(results)
        self.assertEqual(len(top), 2)
        self.assertEqual([r.ligand for r in top], ["m19", "m18"])


class TestBatchedPipeline(unittest.TestCase):
    def test_two_stage_uses_two_batch_calls_and_reuses_prepared_sdfs(self):
        molecules = [(f"m{i}", "C") for i in range(10)]
        calls = []

        def fake_batch(receptor, prepared_ligands, center, size, mode, **kwargs):
            calls.append((mode, dict(prepared_ligands), kwargs.get("profile")))
            if mode == "fast":
                return [
                    ge.DockResult(name, mode, -float(index + 1), 0.2, True)
                    for index, name in enumerate(prepared_ligands)
                ]
            return [
                ge.DockResult(name, mode, -20.0 - index, 0.8, True)
                for index, name in enumerate(prepared_ligands)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            rows, info = ge.run_two_stage_screening(
                molecules,
                "r.pdb",
                (0, 0, 0),
                (20, 20, 20),
                out_dir=tmp,
                top_fraction=0.10,
                prepare_fn=fake_prepare,
                batch_dock_fn=fake_batch,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual([call[0] for call in calls], ["fast", "accurate"])
        self.assertEqual(info["gnina_processes"], 2)
        self.assertEqual(len(calls[1][1]), 1)
        selected_name = next(iter(calls[1][1]))
        self.assertEqual(calls[1][1][selected_name], calls[0][1][selected_name])
        by_name = {row["ligand"]: row for row in rows}
        self.assertEqual(by_name[selected_name]["skor_kaynagi"], ge.KAYNAK_ACCURATE)

    def test_single_mode_uses_one_batch_call(self):
        calls = []

        def fake_batch(receptor, prepared_ligands, center, size, mode, **kwargs):
            calls.append((mode, list(prepared_ligands)))
            return [ge.DockResult(name, mode, -5.0, 0.1, True) for name in prepared_ligands]

        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = ge.run_single_mode_screening(
                [("a", "C"), ("b", "CC")],
                "r.pdb",
                (0, 0, 0),
                (20, 20, 20),
                out_dir=tmp,
                prepare_fn=fake_prepare,
                batch_dock_fn=fake_batch,
                log_fn=lambda *_args, **_kwargs: None,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
