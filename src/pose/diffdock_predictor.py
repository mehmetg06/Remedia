# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""DiffDock pose predictor behind the :class:`BasePosePredictor` interface (Phase 5).

This **reuses the DiffDock groundwork already in the repository** rather than
rebuilding it: ``merge_diffdock_results.load_diffdock`` parses a
``diffdock_results.csv`` (``ligand, diffdock_confidence, ...``) that DiffDock
produces (e.g. the existing Colab flow) and this predictor turns those
confidences into engine-agnostic :class:`PoseScore` records.

Three ways to obtain confidences, in priority order:

1. an injected ``runner`` callable (used in tests / for a live DiffDock backend),
2. an existing ``diffdock_results.csv`` (the repo's established workflow),
3. the NVIDIA DiffDock NIM endpoint (credential-gated; documented, non-blocking).

If none is available it raises :class:`DiffDockUnavailable`, which Hybrid
Validation and the factory handle gracefully so a run is never blocked.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .base import BasePosePredictor, PoseResult, PoseScore

# NVIDIA DiffDock NIM (hosted). Override with DIFFDOCK_BASE_URL for self-hosted.
DEFAULT_DIFFDOCK_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
API_KEY_ENV_VARS = ("DIFFDOCK_API_KEY", "NVIDIA_API_KEY", "NGC_API_KEY", "NVCF_RUN_KEY")


class DiffDockUnavailable(RuntimeError):
    """Raised when no DiffDock source (runner / CSV / credentials) is available."""


def diffdock_api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class DiffDockPredictor(BasePosePredictor):
    """Deep-learning pose prediction (confidence scores) via DiffDock."""

    name = "diffdock"

    def __init__(
        self,
        *,
        results_csv: str | Path | None = None,
        runner: Callable[..., Any] | None = None,
        loader: Callable[[Path], dict[str, float | None]] | None = None,
        base_url: str | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.results_csv = Path(results_csv) if results_csv else None
        self._runner = runner
        self._loader = loader
        self.base_url = base_url or os.environ.get("DIFFDOCK_BASE_URL", DEFAULT_DIFFDOCK_URL)
        self._log = log_fn

    def _confidences(
        self,
        molecules: list[tuple[str, str]],
        receptor: str | None,
        center: Any,
        size: Any,
        out_dir: Any,
        reporter: Any,
        **kwargs: Any,
    ) -> dict[str, float | None]:
        # 1) injected runner (tests / live backend)
        if self._runner is not None:
            result = self._runner(molecules=molecules, receptor=receptor, center=center,
                                  size=size, out_dir=out_dir, **kwargs)
            if isinstance(result, (str, Path)):
                return self._load_csv(Path(result))
            return dict(result or {})
        # 2) existing diffdock_results.csv (repo groundwork)
        csv_path = self._find_results_csv(out_dir)
        if csv_path is not None:
            if reporter is not None:
                reporter.log(f"DiffDock sonuçları okunuyor: {csv_path}")
            return self._load_csv(csv_path)
        # 3) hosted NIM (needs credentials) — documented, not run by default
        if diffdock_api_key():
            raise DiffDockUnavailable(
                "DiffDock NIM istemcisi bu ortamda etkin değil. "
                "diffdock_results.csv sağla ya da bir runner enjekte et "
                "(bkz. docs/diffdock_setup.md)."
            )
        raise DiffDockUnavailable(
            "DiffDock için kaynak bulunamadı: runner, diffdock_results.csv veya "
            "DIFFDOCK_API_KEY gerekli. Kurulum: docs/diffdock_setup.md"
        )

    def _find_results_csv(self, out_dir: Any) -> Path | None:
        candidates = []
        if self.results_csv is not None:
            candidates.append(self.results_csv)
        if out_dir is not None:
            candidates.append(Path(out_dir) / "diffdock_results.csv")
            candidates.append(Path(out_dir).parent / "diffdock_results.csv")
        for path in candidates:
            if path and Path(path).exists():
                return Path(path)
        return None

    def _load_csv(self, path: Path) -> dict[str, float | None]:
        if self._loader is not None:
            return self._loader(path)
        from merge_diffdock_results import load_diffdock

        return load_diffdock(path)

    def predict_pose(
        self,
        molecules: list[tuple[str, str]],
        *,
        receptor: str | None = None,
        center: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        out_dir: Any | None = None,
        reporter: Any | None = None,
        **kwargs: Any,
    ) -> PoseResult:
        if reporter is not None:
            reporter.log("DiffDock poz tahmini")
        confidences = self._confidences(molecules, receptor, center, size, out_dir, reporter, **kwargs)

        scores: list[PoseScore] = []
        for name, _smiles in molecules:
            conf = confidences.get(name)
            scores.append(PoseScore(
                ligand=name,
                affinity_kcal_mol=None,
                confidence=conf,
                success=conf is not None,
                source="diffdock",
                error=None if conf is not None else "DiffDock skoru yok",
            ))
        rows = [s.to_row() for s in scores]
        if reporter is not None:
            ok = sum(1 for s in scores if s.success)
            reporter.update(ok, total=len(scores), message=f"DiffDock: {ok}/{len(scores)} poz")
        return PoseResult(
            engine=self.name,
            scores=scores,
            rows=rows,
            stage_info={"diffdock_scored": sum(1 for s in scores if s.success)},
            metadata={"endpoint": self.base_url, "source_csv": str(self._find_results_csv(out_dir) or "")},
        )
