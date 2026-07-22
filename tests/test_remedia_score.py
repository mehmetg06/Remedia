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


class TestDockingFailurePenalty(unittest.TestCase):
    def test_failed_dock_scores_below_equivalent_scored(self):
        # Same molecule twice: one docked, one whose docking produced no score.
        # The failed one must rank strictly below despite identical ADMET/props.
        cands = [
            candidate("m_ok", aff=-8.0, passed=True, viol="-", mw=320, logp=2.0,
                      tpsa=70, hbd=1, hba=4, smiles="CCO"),
            candidate("m_fail", aff=None, passed=True, viol="-", mw=320, logp=2.0,
                      tpsa=70, hbd=1, hba=4, smiles="CCO"),
        ]
        scored = {c["ligand"]: c for c in rs.compute_scores(cands)}
        self.assertEqual(scored["m_ok"]["docking_status"], "scored")
        self.assertEqual(scored["m_fail"]["docking_status"], "docking_failed")
        self.assertLess(scored["m_fail"]["remedia_score"], scored["m_ok"]["remedia_score"])
        # The failed candidate is still scored (not None) but capped by losing the
        # pose weight from the numerator.
        self.assertIsNotNone(scored["m_fail"]["remedia_score"])

    def test_explicit_docking_success_false_is_penalised(self):
        ok = candidate("ok", aff=-7.0, passed=True, smiles="CCO")
        bad = candidate("bad", aff=-7.0, passed=True, smiles="CCO")
        bad["docking_success"] = False
        scored = {c["ligand"]: c for c in rs.compute_scores([ok, bad])}
        self.assertEqual(scored["bad"]["docking_status"], "docking_failed")
        self.assertLess(scored["bad"]["remedia_score"], scored["ok"]["remedia_score"])

    def test_failed_dock_score_capped_below_pose_weight(self):
        # A docking failure loses the pose weight from the numerator but keeps it
        # in the denominator, so its score can never exceed 1 - pose_weight (0.6),
        # however strong its ADMET/drug-likeness/diversity are.  A well-docked
        # candidate can exceed that ceiling — so a failure cannot be rescued to
        # the top.
        cands = [
            candidate("winner", aff=-9.0, passed=True, viol="-", mw=320, logp=2.0,
                      tpsa=70, hbd=1, hba=4, smiles="CCO"),
            candidate("fail", aff=None, passed=True, viol="-", mw=320, logp=2.0,
                      tpsa=70, hbd=1, hba=4, smiles="c1ccccc1CCN"),
        ]
        scored = {c["ligand"]: c for c in rs.compute_scores(cands)}
        self.assertEqual(scored["fail"]["docking_status"], "docking_failed")
        self.assertLessEqual(scored["fail"]["remedia_score"], 0.6 + 1e-9)
        self.assertGreater(scored["winner"]["remedia_score"], scored["fail"]["remedia_score"])

    def test_pose_free_run_not_penalised(self):
        # No candidate has any affinity/confidence: this is a pose-free run, not a
        # failure — the pose weight is renormalised out (graceful degradation),
        # docking_status is "no_pose", and scores stay non-null.
        cands = [
            {"ligand": "a", "admet_pass": True, "violations": "-", "smiles": "CCO"},
            {"ligand": "b", "admet_pass": True, "violations": "-", "smiles": "c1ccccc1CCN"},
        ]
        scored = rs.compute_scores(cands)
        for c in scored:
            self.assertEqual(c["docking_status"], "no_pose")
            self.assertIsNotNone(c["remedia_score"])
            self.assertIsNone(c["pose_score"])

    def test_docking_status_scored_when_all_dock(self):
        scored = rs.compute_scores([
            candidate("a", aff=-9.0, passed=True, smiles="CCO"),
            candidate("b", aff=-6.0, passed=True, smiles="CCN"),
        ])
        for c in scored:
            self.assertEqual(c["docking_status"], "scored")


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
