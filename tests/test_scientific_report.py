# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the scientific/research report layer (Phases 7 & 7.5).

No rdkit/matplotlib present here, so this also verifies graceful degradation:
the core report is produced without structures/figures/PDF.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scientific_report as sr  # noqa: E402


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_run_dir(tmp):
    root = Path(tmp)
    report_dir = root / sr.REPORT_DIR_NAME
    # Phase 6 output
    _write_csv(root / "remedia_ranking.csv", [
        {"rank": 1, "ligand": "mol_001", "molecule": "mol_001", "smiles": "CCOc1ccccc1",
         "remedia_score": 0.91, "pose_score": 0.95, "admet_score": 1.0,
         "druglikeness_score": 0.88, "diversity_score": 1.0,
         "affinity_kcal_mol": -9.1, "pose_confidence": "", "admet_pass": "True",
         "violations": "-", "scaffold": "c1ccccc1"},
        {"rank": 2, "ligand": "mol_002", "molecule": "mol_002", "smiles": "CCN",
         "remedia_score": 0.55, "pose_score": 0.4, "admet_score": 1.0,
         "druglikeness_score": 0.7, "diversity_score": 1.0,
         "affinity_kcal_mol": -6.0, "pose_confidence": "", "admet_pass": "True",
         "violations": "-", "scaffold": "CCN"},
    ], ["rank", "ligand", "molecule", "smiles", "remedia_score", "pose_score",
        "admet_score", "druglikeness_score", "diversity_score", "affinity_kcal_mol",
        "pose_confidence", "admet_pass", "violations", "scaffold"])
    # props from the base report
    _write_csv(report_dir / "candidate_overview.csv", [
        {"molecule": "mol_001", "smiles": "CCOc1ccccc1", "MW": 122.0, "LogP": 2.1,
         "TPSA": 9.2, "HBD": 0, "HBA": 1, "pass": "True", "violations": "-"},
        {"molecule": "mol_002", "smiles": "CCN", "MW": 45.0, "LogP": -0.1,
         "TPSA": 26.0, "HBD": 1, "HBA": 1, "pass": "True", "violations": "-"},
    ], ["molecule", "smiles", "MW", "LogP", "TPSA", "HBD", "HBA", "pass", "violations"])
    return root


class TestLoad(unittest.TestCase):
    def test_merges_scores_and_props(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_run_dir(tmp)
            cands = sr.load_candidates(root)
            self.assertEqual(len(cands), 2)
            top = cands[0]
            self.assertEqual(top["molecule"], "mol_001")
            self.assertEqual(top["remedia_score"], 0.91)
            self.assertEqual(top["mw"], 122.0)  # merged from overview
            self.assertEqual(top["affinity_kcal_mol"], -9.1)


class TestNarrative(unittest.TestCase):
    def test_ranking_explanation_mentions_binding_and_admet(self):
        cand = {"rank": 1, "affinity_kcal_mol": -9.1, "admet_status": "True",
                "violations": "-", "druglikeness_score": 0.88, "diversity_score": 1.0,
                "remedia_score": 0.91}
        text = sr.ranking_explanation(cand)
        self.assertIn("#1", text)
        self.assertIn("kcal/mol", text)
        self.assertIn("ADMET", text)

    def test_binding_analysis_uses_pocket_and_affinity(self):
        text = sr.binding_analysis({"affinity_kcal_mol": -8.5}, (12.3, 45.1, -3.2))
        self.assertIn("kcal/mol", text)
        self.assertIn("12.3", text)

    def test_similarity_fallback_without_rdkit(self):
        cands = [{"molecule": "m1", "smiles": "CCOc1ccccc1"}]
        known = [{"name": "aspirin", "smiles": "CCOc1ccccc1OC"}]
        sim = sr.similarity_analysis(cands, known)
        self.assertIn("m1", sim)
        self.assertEqual(sim["m1"]["nearest_known"], "aspirin")
        self.assertEqual(sim["m1"]["method"], "string-proxy")

    def test_executive_summary(self):
        cands = [{"molecule": "m1", "affinity_kcal_mol": -9.0, "admet_status": "True",
                  "remedia_score": 0.9}]
        div = {"molecules": 1, "unique_scaffolds": 1, "diversity_score": 1.0}
        text = sr.executive_summary(cands, div, "P00918")
        self.assertIn("P00918", text)
        self.assertIn("m1", text)


class TestBuild(unittest.TestCase):
    def test_produces_all_core_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_run_dir(tmp)
            info = sr.build_scientific_report(
                root, target_uniprot="P00918", requested_molecules=2,
                settings={"generator": "reinvent4", "pose_engine": "gnina"},
                pipeline_log="line1\nline2\n", job_id="abc",
                known_ligands=[{"name": "aspirin", "smiles": "CCOc1ccccc1OC"}],
                pocket_center=(12.3, 45.1, -3.2), seeds=["CCO"],
            )
            report_dir = root / sr.REPORT_DIR_NAME
            for fname in ("report.html", "README_FIRST.txt", "run_manifest.json",
                          "candidate_overview.csv", "pipeline_log.txt"):
                self.assertTrue((report_dir / fname).exists(), fname)

            # HTML has candidate cards + sections
            html = (report_dir / "report.html").read_text()
            self.assertIn("mol_001", html)
            self.assertIn("Neden yüksek sıralandı", html)
            self.assertIn("Yönetici özeti", html)
            self.assertIn("Çeşitlilik analizi", html)

            # manifest carries provenance
            manifest = json.loads((report_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["target_uniprot"], "P00918")
            self.assertEqual(manifest["generator"], "reinvent4")
            self.assertEqual(manifest["pose_engine"], "gnina")
            self.assertEqual(manifest["seeds"], ["CCO"])
            self.assertIn("environment", manifest)
            self.assertIn("python", manifest["environment"])
            self.assertEqual(len(manifest["top_candidates"]), 2)

            # log preserved verbatim
            self.assertIn("line1", (report_dir / "pipeline_log.txt").read_text())

            # returned info
            self.assertEqual(info["candidate_count"], 2)
            self.assertTrue(info["report_path"].endswith("report.html"))

    def test_readme_has_guidance_and_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_run_dir(tmp)
            sr.build_scientific_report(root, target_uniprot="P00918")
            readme = (root / sr.REPORT_DIR_NAME / "README_FIRST.txt").read_text()
            self.assertIn("ÖNCE BUNU OKU", readme)
            self.assertIn("Remedia Score", readme)
            self.assertIn("UYARI", readme)

    def test_empty_run_still_produces_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = sr.build_scientific_report(tmp, target_uniprot="P99999")
            self.assertTrue(Path(info["report_path"]).exists())
            self.assertEqual(info["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
