# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Reproducibility metadata capture (Phase 9).

Every Remedia run should be reproducible.  This module records the exact inputs
and environment that produced a result so a run can be repeated and audited:

* **git commit** (and whether the tree was dirty),
* **random seeds** (pipeline seed + any generator/pose seeds),
* **software versions** (Python, RDKit, torch, numpy, REINVENT, …),
* **tool versions** (the GNINA binary),
* **parameters** (the full ``settings`` dict).

The result is embedded in ``run_manifest.json`` (via ``scientific_report``) and
can also be written standalone with :func:`write_manifest`.

Stdlib only; every probe is best-effort and never raises.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Default random seed used across the pipeline (GNINA docking seed).
DEFAULT_RANDOM_SEED = 42

_TRACKED_PACKAGES = (
    "rdkit", "torch", "numpy", "pandas", "scipy", "matplotlib",
    "reinvent", "requests", "biopython", "meeko",
)


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip() or None


def git_commit(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Return {commit, dirty, branch} for the repo, best-effort."""
    cwd = str(repo_dir) if repo_dir else str(Path(__file__).resolve().parent.parent)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def collect_software_versions() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in _TRACKED_PACKAGES},
    }


def gnina_version(gnina_path: str | None = None) -> str | None:
    """Return the GNINA binary version string, or None if unavailable."""
    import os

    path = gnina_path or os.environ.get("GNINA_PATH", "/usr/local/bin/gnina")
    out = _run([path, "--version"])
    if not out:
        return None
    return out.splitlines()[0].strip()


def capture_run_metadata(
    *,
    settings: dict[str, Any] | None = None,
    seeds: list[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    gnina_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a full reproducibility record for a run."""
    settings = settings or {}
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_commit(),
        "random_seed": random_seed,
        "seed_molecules": list(seeds or []),
        "generator": settings.get("generator"),
        "pose_engine": settings.get("pose_engine"),
        "parameters": settings,
        "software": collect_software_versions(),
        "tools": {"gnina": gnina_version(gnina_path)},
        "extra": extra or {},
    }


def write_manifest(path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
