# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Guard the Modal web worker wiring for Phases 2 (progress) and 4 (generator)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "modal" / "remedia_web_v2.py"


class WebWorkerAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WEB.read_text()

    def test_parses(self):
        ast.parse(self.source)

    def test_structured_progress_parsing(self):
        # Phase 2: worker consumes the structured progress sentinel.
        self.assertIn("PROGRESS_SENTINEL", self.source)
        self.assertIn("remedia.progress/1", self.source)
        self.assertIn("def _structured", self.source)

    def test_ui_shows_item_counts(self):
        # Phase 2: UI renders items_done/items_total and ETA.
        self.assertIn("items_total", self.source)
        self.assertIn("eta_seconds", self.source)

    def test_generator_selector_present(self):
        # Phase 4: generator radio group + all three options.
        self.assertIn('name="gen"', self.source)
        for value in ("reinvent4", "molmim", "hybrid"):
            self.assertIn(f'value="{value}"', self.source)

    def test_pose_selector_present(self):
        # Phase 4 scaffolding / Phase 5: pose engine radio group.
        self.assertIn('name="pose"', self.source)
        for value in ("gnina", "diffdock"):
            self.assertIn(f'value="{value}"', self.source)

    def test_start_validates_generator(self):
        self.assertIn('payload.get("generator"', self.source)
        self.assertIn('"reinvent4", "molmim", "hybrid"', self.source)

    def test_run_job_accepts_generator(self):
        self.assertIn("generator: str", self.source)
        self.assertIn('"generator": generator', self.source)

    def test_start_validates_pose_engine(self):
        # Phase 5: pose engine is validated and passed through to run_job.
        self.assertIn('payload.get("pose_engine"', self.source)
        self.assertIn('"gnina", "diffdock", "boltz2", "hybrid"', self.source)
        self.assertIn('"pose_engine": pose_engine', self.source)

    def test_scientific_report_wired(self):
        # Phase 7/7.5: rich report layered additively, image src rewritten.
        self.assertIn("build_scientific_report", self.source)
        self.assertIn("report-asset", self.source)

    def test_git_auto_sync(self):
        # Auto-updating deploy: code is pulled from GitHub at run time instead of
        # being frozen into the image, so no redeploy is needed after a push.
        self.assertIn("def _git_sync", self.source)
        self.assertIn("REMEDIA_GIT_BRANCH", self.source)
        self.assertIn('"reset", "--hard"', self.source)
        # The GPU worker must sync before loading the pipeline.
        self.assertIn("_git_sync(force=True)", self.source)
        # The page load triggers a throttled sync so a refresh picks up new code.
        self.assertIn("def home", self.source)

    def test_speed_selector_present(self):
        # Speed control exposed in the UI and validated server-side.
        self.assertIn('name="speed"', self.source)
        for value in ("hizli", "dengeli"):
            self.assertIn(f'value="{value}"', self.source)

    def test_start_validates_speed(self):
        self.assertIn('payload.get("speed"', self.source)
        self.assertIn('"hizli", "dengeli"', self.source)

    def test_run_job_accepts_speed(self):
        self.assertIn("speed: str", self.source)
        # Speed maps onto the GNINA staging mode.
        self.assertIn('"sadece_fast"', self.source)
        self.assertIn('"iki_asamali"', self.source)

    def test_live_event_stream_wired(self):
        # Faz 1: per-job events.jsonl + /events polling route.
        self.assertIn("def _events_file", self.source)
        self.assertIn(".events.jsonl", self.source)
        self.assertIn("_append_event", self.source)
        self.assertIn('@api.get("/events/{job_id}")', self.source)
        self.assertIn("since: int", self.source)
        self.assertIn("last_seq", self.source)

    def test_heartbeat_thread_wired(self):
        # Faz 1: a background heartbeat proves liveness during blocking GNINA.
        self.assertIn("def heartbeat", self.source)
        self.assertIn("heartbeat_loop", self.source)
        self.assertIn("threading.Thread", self.source)
        self.assertIn('"event": "heartbeat"', self.source)

    def test_live_console_ui_present(self):
        # Faz 1 §6.4 panels + event-stream polling in the frontend.
        self.assertIn("Canlı lider tablosu", self.source)
        self.assertIn("Canlı molekül akışı", self.source)
        self.assertIn("/events/", self.source)      # UI polls the stream
        self.assertIn("candidate_scored", self.source)
        self.assertIn("leader_changed", self.source)
        self.assertIn("Ham teknik log", self.source)

    def test_speed_labels_are_explicit(self):
        # A3: fast-only vs two-stage named explicitly (values unchanged).
        self.assertIn("Sadece hızlı", self.source)
        self.assertIn("İki aşamalı", self.source)

    def test_score_labelled_as_heuristic(self):
        # A2: the UI frames the score as a temporary heuristic, not a certainty.
        self.assertIn("heuristik", self.source.lower())


if __name__ == "__main__":
    unittest.main()
