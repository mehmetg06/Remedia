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
        self.assertIn('"gnina", "diffdock", "hybrid"', self.source)
        self.assertIn('"pose_engine": pose_engine', self.source)

    def test_scientific_report_wired(self):
        # Phase 7/7.5: rich report layered additively, image src rewritten.
        self.assertIn("build_scientific_report", self.source)
        self.assertIn("report-asset", self.source)


if __name__ == "__main__":
    unittest.main()
