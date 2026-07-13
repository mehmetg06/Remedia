"""Load the latest Remedia Modal setup and form into the current notebook.

This file is intentionally fetched on every launcher run. The launcher notebook
therefore stays unchanged while setup/runtime fixes can be shipped here or in
``notebooks/remedia_modal_import.ipynb``.
"""
from __future__ import annotations

import json
import time
import urllib.request

IMPORT_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/mehmetg06/Remedia/"
    "main/notebooks/remedia_modal_import.ipynb"
)


def _source_text(source):
    return "".join(source) if isinstance(source, list) else str(source or "")


def run(namespace=None):
    """Fetch and execute the latest import notebook in one shared namespace."""
    target = namespace if namespace is not None else globals()
    cache_busted_url = f"{IMPORT_NOTEBOOK_URL}?t={time.time_ns()}"
    request = urllib.request.Request(
        cache_busted_url,
        headers={"Cache-Control": "no-cache", "User-Agent": "Remedia-Modal-Launcher"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        notebook = json.load(response)

    code_cells = [
        _source_text(cell.get("source"))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if not code_cells:
        raise RuntimeError("Güncel Remedia import kodu bulunamadı.")

    print("🔄 Güncel Remedia kurulumu GitHub'dan alındı.")
    for index, source in enumerate(code_cells, 1):
        exec(compile(source, f"{IMPORT_NOTEBOOK_URL}#cell-{index}", "exec"), target)


if __name__ == "__main__":
    run(globals())
