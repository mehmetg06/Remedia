import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LocalAssetsTest(unittest.TestCase):
    def test_local_notebook_code_cells_parse(self):
        notebook = json.loads(
            (ROOT / "notebooks" / "remedia_local.ipynb").read_text()
        )
        self.assertEqual(notebook["nbformat"], 4)
        sources = [
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertGreaterEqual(len(sources), 5)
        for index, source in enumerate(sources, start=1):
            with self.subTest(cell=index):
                ast.parse(source)

    def test_local_notebook_has_safe_defaults(self):
        notebook = json.loads(
            (ROOT / "notebooks" / "remedia_local.ipynb").read_text()
        )
        source = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('ACCURACY_PROFILE = "balanced"', source)
        self.assertIn("RUN_BENCHMARK = False", source)
        self.assertIn('REPO_DIR / "local_workspace"', source)
        self.assertIn("nvidia-smi", source)

    def test_local_support_files_exist(self):
        for relative_path in (
            "scripts/setup_local.sh",
            "Dockerfile.local",
            "environment.yml",
        ):
            self.assertTrue((ROOT / relative_path).is_file())


if __name__ == "__main__":
    unittest.main()
