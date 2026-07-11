import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModalAssetsTest(unittest.TestCase):
    def test_modal_launcher_parses(self):
        source = (ROOT / "modal" / "remedia_modal.py").read_text()
        ast.parse(source)
        self.assertIn("gpu=GPU", source)
        self.assertIn("MAX_SESSION_MINUTES = 240", source)
        self.assertIn("modal.forward", source)

    def test_modal_notebook_code_cells_parse(self):
        notebook_path = ROOT / "notebooks" / "remedia_modal.ipynb"
        notebook = json.loads(notebook_path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertGreaterEqual(len(code_cells), 5)
        for index, cell in enumerate(code_cells, start=1):
            with self.subTest(cell=index):
                ast.parse(cell["source"])

    def test_modal_notebook_has_cost_guards(self):
        notebook_path = ROOT / "notebooks" / "remedia_modal.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('ACCURACY_PROFILE = "balanced"', source)
        self.assertIn("RUN_BENCHMARK = False", source)
        self.assertIn('PERSISTENT_ROOT = "/mnt/remedia-data"', source)


if __name__ == "__main__":
    unittest.main()
