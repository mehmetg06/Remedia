# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for reproducibility capture + publication docs (Phase 9)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reproducibility as repro  # noqa: E402


class TestReproducibility(unittest.TestCase):
    def test_software_versions_shape(self):
        sv = repro.collect_software_versions()
        self.assertIn("python", sv)
        self.assertIn("packages", sv)
        self.assertIn("rdkit", sv["packages"])  # key present even if None

    def test_git_commit_in_repo(self):
        info = repro.git_commit(ROOT)
        # This project is a git repo, so a commit hash should be found.
        self.assertIsNotNone(info["commit"])
        self.assertEqual(len(info["commit"]), 40)

    def test_gnina_version_absent_is_none(self):
        self.assertIsNone(repro.gnina_version("/nonexistent/gnina-binary"))

    def test_capture_run_metadata_keys(self):
        meta = repro.capture_run_metadata(
            settings={"generator": "reinvent4", "pose_engine": "gnina", "profile": "balanced"},
            seeds=["CCO", "CCN"],
            random_seed=42,
        )
        for key in ("captured_at_utc", "git", "random_seed", "seed_molecules",
                    "generator", "pose_engine", "parameters", "software", "tools"):
            self.assertIn(key, meta)
        self.assertEqual(meta["random_seed"], 42)
        self.assertEqual(meta["generator"], "reinvent4")
        self.assertEqual(meta["seed_molecules"], ["CCO", "CCN"])

    def test_write_manifest(self):
        meta = repro.capture_run_metadata(settings={"generator": "molmim"})
        with tempfile.TemporaryDirectory() as tmp:
            path = repro.write_manifest(Path(tmp) / "repro.json", meta)
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["generator"], "molmim")


class TestPublicationDocs(unittest.TestCase):
    def test_docs_exist_and_have_content(self):
        for name in ("benchmark_protocol.md", "reproducibility.md"):
            path = ROOT / "docs" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(len(path.read_text()), 500, name)

    def test_reproducibility_doc_mentions_tracked_items(self):
        text = (ROOT / "docs" / "reproducibility.md").read_text().lower()
        for term in ("seed", "commit", "version", "run_manifest.json"):
            self.assertIn(term, text)

    def test_benchmark_doc_mentions_metrics(self):
        text = (ROOT / "docs" / "benchmark_protocol.md").read_text().lower()
        for term in ("runtime", "diversity", "admet", "reinvent4", "gnina", "diffdock"):
            self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
