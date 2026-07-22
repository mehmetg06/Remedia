# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the structured progress layer (Phase 2). Stdlib only — no rdkit."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import progress  # noqa: E402


class TestSchema(unittest.TestCase):
    def test_stage_bounds_are_monotonic_and_bounded(self):
        bounds = progress._cumulative_bounds()
        last_end = 0.0
        for stage in progress.STAGES:
            start, end = bounds[stage.key]
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(start, last_end - 1e-9)
            self.assertLessEqual(end, 1.0 + 1e-9)
            last_end = end

    def test_unknown_stage_does_not_raise(self):
        stage = progress.stage_for("brand_new_stage")
        self.assertEqual(stage.key, "brand_new_stage")
        self.assertTrue(stage.label)


class TestReporter(unittest.TestCase):
    def test_events_carry_item_counts_and_percent(self):
        r = progress.ProgressReporter(emit_stdout=False)
        r.stage("dock_fast", task="GNINA FAST docking", total=50)
        r.update(18)
        event = r.snapshot()
        self.assertEqual(event["stage"], "dock_fast")
        self.assertEqual(event["items_done"], 18)
        self.assertEqual(event["items_total"], 50)
        self.assertEqual(event["message"], "GNINA FAST docking (18/50)")
        self.assertGreater(event["percent"], 0.0)
        self.assertLessEqual(event["percent"], 100.0)

    def test_percent_never_decreases(self):
        r = progress.ProgressReporter(emit_stdout=False)
        r.stage("generate", total=20)
        r.update(20)  # finish generation -> high-ish percent
        high = r.snapshot()["percent"]
        r.stage("admet", total=20)
        r.update(1)  # early admet
        self.assertGreaterEqual(r.snapshot()["percent"], high)

    def test_persistent_logs_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = progress.ProgressReporter(tmp, emit_stdout=False)
            r.stage("receptor")
            r.stage("generate", total=10)
            r.update(5)
            jsonl = Path(tmp) / "progress.jsonl"
            state = Path(tmp) / "progress_state.json"
            self.assertTrue(jsonl.exists())
            self.assertTrue(state.exists())
            lines = jsonl.read_text().strip().splitlines()
            self.assertGreaterEqual(len(lines), 3)
            for line in lines:
                json.loads(line)  # each line is valid JSON
            last = json.loads(state.read_text())
            self.assertEqual(last["items_done"], 5)

    def test_exception_records_full_traceback(self):
        r = progress.ProgressReporter(emit_stdout=False)
        try:
            raise ValueError("boom")
        except ValueError as exc:
            event = r.exception(exc, context="docking failed")
        self.assertEqual(event["level"], "error")
        self.assertEqual(event["error_type"], "ValueError")
        self.assertIn("Traceback", event["traceback"])
        self.assertIn("boom", event["traceback"])

    def test_sink_receives_events(self):
        seen = []
        r = progress.ProgressReporter(emit_stdout=False, sink=seen.append)
        r.stage("pocket")
        self.assertTrue(seen)
        self.assertEqual(seen[-1]["stage"], "pocket")

    def test_sink_failure_does_not_propagate(self):
        def bad_sink(_):
            raise RuntimeError("sink down")

        r = progress.ProgressReporter(emit_stdout=False, sink=bad_sink)
        r.stage("pocket")  # must not raise


class TestLiveEvents(unittest.TestCase):
    def test_event_carries_type_and_payload(self):
        seen = []
        r = progress.ProgressReporter(emit_stdout=False, sink=seen.append)
        r.stage("dock_fast", total=20)
        r.update(10)
        emitted = r.event(
            "candidate_scored", candidate="mol_003", predicted_pic50=7.2,
            accepted=True,
        )
        self.assertEqual(emitted["event"], "candidate_scored")
        self.assertEqual(emitted["candidate"], "mol_003")
        self.assertEqual(emitted["predicted_pic50"], 7.2)
        self.assertTrue(emitted["accepted"])
        self.assertEqual(seen[-1]["event"], "candidate_scored")

    def test_event_does_not_change_counters(self):
        r = progress.ProgressReporter(emit_stdout=False)
        r.stage("dock_fast", total=20)
        r.update(10)
        before = r.snapshot()
        r.event("leader_changed", candidate="mol_001")
        after = r.snapshot()
        self.assertEqual(after["items_done"], before["items_done"])
        self.assertEqual(after["percent"], before["percent"])

    def test_event_persisted_and_sentinel_roundtrips(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            r = progress.ProgressReporter(tmp, emit_stdout=True, stream=buf)
            r.stage("generate", total=5)
            r.event("candidate_generated", candidate="mol_001", smiles="CCO")
            jsonl = Path(tmp) / "progress.jsonl"
            payloads = [json.loads(ln) for ln in jsonl.read_text().splitlines()]
            self.assertTrue(any(p.get("event") == "candidate_generated" for p in payloads))
            sentinel_lines = [
                ln for ln in buf.getvalue().splitlines() if progress.SENTINEL in ln
            ]
            parsed = progress.parse_sentinel(sentinel_lines[-1])
            self.assertEqual(parsed["event"], "candidate_generated")
            self.assertEqual(parsed["candidate"], "mol_001")


class TestStdoutSentinel(unittest.TestCase):
    def test_echo_emits_human_line_and_sentinel(self):
        buf = io.StringIO()
        r = progress.ProgressReporter(emit_stdout=True, stream=buf)
        r.stage("dock_accurate", task="GNINA ACCURATE docking", total=5)
        r.update(3)
        text = buf.getvalue()
        self.assertIn("GNINA ACCURATE docking (3/5)", text)
        self.assertIn(progress.SENTINEL, text)

    def test_parse_sentinel_roundtrip(self):
        buf = io.StringIO()
        r = progress.ProgressReporter(emit_stdout=True, stream=buf)
        r.stage("admet", total=42)
        r.update(42)
        sentinel_lines = [
            ln for ln in buf.getvalue().splitlines() if progress.SENTINEL in ln
        ]
        self.assertTrue(sentinel_lines)
        parsed = progress.parse_sentinel(sentinel_lines[-1])
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["stage"], "admet")
        self.assertEqual(parsed["items_total"], 42)

    def test_parse_sentinel_ignores_plain_lines(self):
        self.assertIsNone(progress.parse_sentinel("just a normal log line"))
        self.assertIsNone(progress.parse_sentinel("GNINA docking yapıyor"))


if __name__ == "__main__":
    unittest.main()
