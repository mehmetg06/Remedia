# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the composite Remedia Score (Phase 6). No rdkit required."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import remedia_score as rs  # noqa: E402


def candidate(name, aff=None, conf=None, passed=None, viol="-", mw=None, logp=None,
              tpsa=None, hbd=None, hba=None, smiles=""):
    return {
        "ligand": name, "affinity_kcal_mol": aff, "pose_confidence": conf,
        "admet_pass": passed, "violations": viol, "MW": mw, "LogP": logp,
        "TPSA": tpsa, "HBD": hbd, "HBA": hba, "smiles": smiles,
    }


class TestComponents(unittest.TestCase):
    def test_druglikeness_good_beats_bad(self):
        good = rs.druglikeness_desirability({"MW": 320, "LogP": 2.1, "TPSA": 70, "HBD": 1, "HBA": 4})
        bad = rs.druglikeness_desirability({"MW": 900, "LogP": 9.0, "TPSA": 250, "HBD": 9, "HBA": 15})
        self.assertGreater(good, bad)
        self.assertGreaterEqual(good, 0.9)

    def test_druglikeness_none_without_props(self):
        self.assertIsNone(rs.druglikeness_desirability({"ligand": "x"}))

    def test_admet_pass_fail_and_violations(self):
        self.assertEqual(rs.admet_score({"admet_pass": True, "violations": "-"}), 1.0)
        self.assertLess(rs.admet_score({"admet_pass": True, "violations": "MW>500;LogP>5"}), 1.0)
        self.assertLess(rs.admet_score({"admet_pass": False}), 1.0)
        self.assertIsNone(rs.admet_score({"ligand": "x"}))

    def test_scaffold_fallback_without_rdkit(self):
        s = rs.scaffold_of("CCO")
        self.assertTrue(s)  # non-empty proxy or Murcko scaffold

    def test_violation_count(self):
        self.assertEqual(rs._violation_count("-"), 0)
        self.assertEqual(rs._violation_count("MW>500;LogP>5"), 2)


class TestComposite(unittest.TestCase):
    def _set(self):
        return [
            candidate("m1", aff=-9.0, passed=True, viol="-", mw=320, logp=2.0, tpsa=70, hbd=1, hba=4, smiles="CCO"),
            candidate("m2", aff=-6.0, passed=True, viol="-", mw=350, logp=3.0, tpsa=80, hbd=2, hba=5, smiles="CCN"),
            candidate("m3", aff=-4.0, passed=False, viol="MW>500;LogP>5", mw=900, logp=9, tpsa=250, hbd=9, hba=15, smiles="CCC"),
        ]

    def test_scores_in_range_and_ranked(self):
        scored = rs.compute_scores(self._set())
        for c in scored:
            self.assertIsNotNone(c["remedia_score"])
            self.assertGreaterEqual(c["remedia_score"], 0.0)
            self.assertLessEqual(c["remedia_score"], 1.0)
        self.assertEqual([c["rank"] for c in scored], [1, 2, 3])
        # best affinity + best props should top the list
        self.assertEqual(scored[0]["ligand"], "m1")

    def test_pose_score_prefers_more_negative_affinity(self):
        scored = {c["ligand"]: c for c in rs.compute_scores(self._set())}
        self.assertGreater(scored["m1"]["pose_score"], scored["m3"]["pose_score"])

    def test_diversity_rewards_unique_scaffolds(self):
        # two identical SMILES share a scaffold; a third is distinct
        cands = [candidate("a", aff=-7, smiles="CCO"),
                 candidate("b", aff=-7, smiles="CCO"),
                 candidate("c", aff=-7, smiles="c1ccccc1CCN")]
        scored = {c["ligand"]: c for c in rs.compute_scores(cands)}
        self.assertAlmostEqual(scored["a"]["diversity_score"], 0.5, places=3)
        self.assertAlmostEqual(scored["c"]["diversity_score"], 1.0, places=3)

    def test_confidence_only_pose(self):
        cands = [candidate("a", conf=0.9, passed=True, smiles="CCO"),
                 candidate("b", conf=-0.5, passed=True, smiles="CCN")]
        scored = {c["ligand"]: c for c in rs.compute_scores(cands)}
        self.assertGreater(scored["a"]["pose_score"], scored["b"]["pose_score"])

    def test_diversity_report(self):
        rep = rs.diversity_report([candidate("a", smiles="CCO"),
                                   candidate("b", smiles="CCO"),
                                   candidate("c", smiles="c1ccccc1CCN")])
        self.assertEqual(rep["molecules"], 3)
        self.assertGreaterEqual(rep["unique_scaffolds"], 2)
        self.assertGreater(rep["diversity_score"], 0)


class TestFallback(unittest.TestCase):
    def test_docking_only_when_no_extra_components(self):
        # only affinity, no admet/props/smiles -> legacy docking-only ranking
        cands = [{"ligand": "a", "affinity_kcal_mol": -6.0},
                 {"ligand": "b", "affinity_kcal_mol": -8.0}]
        ranked = rs.rank_candidates(cands)
        self.assertEqual(ranked[0]["ligand"], "b")  # more negative first
        self.assertNotIn("remedia_score", ranked[0])  # fell back to legacy

    def test_composite_when_admet_present(self):
        cands = [candidate("a", aff=-6, passed=True, smiles="CCO"),
                 candidate("b", aff=-8, passed=False, viol="MW>500;LogP>5;HBD>5", smiles="CCN")]
        ranked = rs.rank_candidates(cands)
        self.assertIn("remedia_score", ranked[0])


class TestCSV(unittest.TestCase):
    def test_write_ranking_csv(self):
        scored = rs.compute_scores([candidate("a", aff=-7, passed=True, smiles="CCO")])
        with tempfile.TemporaryDirectory() as tmp:
            out = rs.write_ranking_csv(scored, Path(tmp) / "remedia_ranking.csv")
            self.assertTrue(out.exists())
            text = out.read_text()
            self.assertIn("remedia_score", text)
            self.assertIn("a", text)


if __name__ == "__main__":
    unittest.main()
