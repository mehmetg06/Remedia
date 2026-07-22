# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the live scoring/streaming seam (Faz 1). Stdlib only — no rdkit."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import live_ranking  # noqa: E402
import progress  # noqa: E402
import remedia_score as rs  # noqa: E402


def candidate(name, aff=None, passed=None, viol="-", smiles="", **extra):
    row = {
        "ligand": name, "molecule": name, "affinity_kcal_mol": aff,
        "admet_pass": passed, "violations": viol, "smiles": smiles,
    }
    row.update(extra)
    return row


def collector():
    seen = []
    reporter = progress.ProgressReporter(emit_stdout=False, sink=seen.append)
    return reporter, seen


def events_of(seen, event_type):
    return [e for e in seen if e.get("event") == event_type]


class TestScoreAndStream(unittest.TestCase):
    def _set(self):
        return [
            candidate("mol_001", aff=-9.0, passed=True, smiles="CCO"),
            candidate("mol_002", aff=-6.0, passed=True, smiles="CCN"),
            candidate("mol_003", aff=-4.0, passed=True, smiles="c1ccccc1CCN"),
        ]

    def test_returns_same_ranking_as_rank_candidates(self):
        cands = self._set()
        expected = rs.rank_candidates([dict(c) for c in cands])
        got = live_ranking.score_and_stream([dict(c) for c in cands])
        self.assertEqual([c["ligand"] for c in got], [c["ligand"] for c in expected])

    def test_emits_one_scored_event_per_candidate(self):
        reporter, seen = collector()
        live_ranking.score_and_stream(self._set(), reporter=reporter)
        scored_events = events_of(seen, "candidate_scored")
        self.assertEqual(len(scored_events), 3)
        names = {e["candidate"] for e in scored_events}
        self.assertEqual(names, {"mol_001", "mol_002", "mol_003"})
        for e in scored_events:
            self.assertIn("remedia_score", e)
            self.assertIn("docking_status", e)
            self.assertIn("accepted", e)

    def test_leader_changed_emitted_in_input_order(self):
        reporter, seen = collector()
        # mol_001 is scored first and is strongest -> it becomes leader; no later
        # candidate should overtake it, so exactly one leader_changed fires.
        live_ranking.score_and_stream(self._set(), reporter=reporter)
        leaders = events_of(seen, "leader_changed")
        self.assertGreaterEqual(len(leaders), 1)
        self.assertEqual(leaders[0]["candidate"], "mol_001")
        self.assertIsNone(leaders[0]["previous"])

    def test_docking_failure_marked_not_accepted(self):
        reporter, seen = collector()
        cands = [
            candidate("mol_001", aff=-8.0, passed=True, smiles="CCO"),
            candidate("mol_002", aff=None, passed=True, smiles="CCN",
                      docking_success=False),
        ]
        live_ranking.score_and_stream(cands, reporter=reporter)
        by_name = {e["candidate"]: e for e in events_of(seen, "candidate_scored")}
        self.assertTrue(by_name["mol_001"]["accepted"])
        self.assertFalse(by_name["mol_002"]["accepted"])
        self.assertEqual(by_name["mol_002"]["docking_status"], "docking_failed")

    def test_reporter_none_is_noop(self):
        got = live_ranking.score_and_stream(self._set(), reporter=None)
        self.assertEqual(len(got), 3)

    def test_round_index_included(self):
        reporter, seen = collector()
        live_ranking.score_and_stream(self._set(), reporter=reporter, round_index=2)
        for e in events_of(seen, "candidate_scored"):
            self.assertEqual(e["round"], 2)


class TestEmitGenerated(unittest.TestCase):
    def test_emits_one_event_per_molecule(self):
        reporter, seen = collector()
        molecules = [("mol_001", "CCO"), ("mol_002", "CCN")]
        live_ranking.emit_generated(reporter, molecules, round_index=1)
        gen = events_of(seen, "candidate_generated")
        self.assertEqual(len(gen), 2)
        self.assertEqual(gen[0]["candidate"], "mol_001")
        self.assertEqual(gen[0]["smiles"], "CCO")
        self.assertEqual(gen[0]["round"], 1)

    def test_reporter_none_is_noop(self):
        live_ranking.emit_generated(None, [("mol_001", "CCO")])  # must not raise


class TestSplitByDocking(unittest.TestCase):
    def test_partitions_by_status(self):
        rows = [
            {"ligand": "a", "docking_status": "scored", "remedia_score": 0.8},
            {"ligand": "b", "docking_status": "docking_failed", "remedia_score": 0.4},
            {"ligand": "c", "docking_status": "no_pose", "remedia_score": 0.5},
        ]
        buckets = live_ranking.split_by_docking(rows)
        self.assertEqual([c["ligand"] for c in buckets["scored"]], ["a"])
        self.assertEqual([c["ligand"] for c in buckets["docking_failed"]], ["b"])
        self.assertEqual([c["ligand"] for c in buckets["no_pose"]], ["c"])


if __name__ == "__main__":
    unittest.main()
