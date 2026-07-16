# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""GNINA pose predictor behind the :class:`BasePosePredictor` interface (Phase 5).

A faithful wrapper around ``gnina_engine``: it makes the same
``run_two_stage_screening`` / ``run_single_mode_screening`` calls the pipeline
made before and returns those exact rows, so GNINA behavior is unchanged.  GNINA
remains fully operational and is never removed — it is the physics-based
confirmation step in Hybrid Validation.

The screening functions are injectable so this predictor is unit-tested without
a GNINA binary or GPU.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import BasePosePredictor, PoseResult, PoseScore

DEFAULT_GNINA_PATH = "/usr/local/bin/gnina"


class GninaPredictor(BasePosePredictor):
    """Physics-based docking via GNINA (fast → accurate two-stage by default)."""

    name = "gnina"

    def __init__(
        self,
        *,
        profile: str = "balanced",
        docking_mode: str = "iki_asamali",
        top_fraction: float = 0.10,
        gnina_path: str = DEFAULT_GNINA_PATH,
        two_stage_fn: Callable[..., Any] | None = None,
        single_mode_fn: Callable[..., Any] | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.profile = profile
        self.docking_mode = docking_mode
        self.top_fraction = top_fraction
        self.gnina_path = gnina_path
        self._two_stage = two_stage_fn
        self._single = single_mode_fn
        self._log = log_fn

    def _resolve(self) -> None:
        if self._two_stage is not None and self._single is not None:
            return
        import gnina_engine

        if self._two_stage is None:
            self._two_stage = gnina_engine.run_two_stage_screening
        if self._single is None:
            self._single = gnina_engine.run_single_mode_screening

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
        self._resolve()
        if reporter is not None:
            reporter.log(f"GNINA docking ({self.docking_mode}, profile={self.profile})")

        common = dict(
            receptor=receptor,
            center=center,
            size=size,
            gnina_path=self.gnina_path,
            out_dir=out_dir,
            log_fn=self._log,
            profile=self.profile,
        )
        if self.docking_mode == "iki_asamali":
            rows, stage_info = self._two_stage(
                molecules, top_fraction=self.top_fraction, **common
            )
        elif self.docking_mode == "sadece_fast":
            rows, stage_info = self._single(molecules, mode="fast", **common)
        else:
            rows, stage_info = self._single(molecules, mode="accurate", **common)

        scores = [self._row_to_score(row) for row in rows]
        if reporter is not None:
            reporter.update(len(scores), total=len(scores), message="GNINA docking tamamlandı")
        return PoseResult(
            engine=self.name,
            scores=scores,
            rows=rows,  # exact legacy rows — downstream unchanged
            stage_info=stage_info,
            metadata={"profile": self.profile, "docking_mode": self.docking_mode,
                      "gnina_processes": stage_info.get("gnina_processes")},
        )

    @staticmethod
    def _row_to_score(row: dict[str, Any]) -> PoseScore:
        return PoseScore(
            ligand=str(row.get("ligand")),
            affinity_kcal_mol=row.get("affinity_kcal_mol"),
            confidence=None,
            success=bool(row.get("docking_success")),
            source="gnina",
            error=row.get("docking_error"),
            extra={
                "fast_affinity_kcal_mol": row.get("fast_affinity_kcal_mol"),
                "accurate_affinity_kcal_mol": row.get("accurate_affinity_kcal_mol"),
                "skor_kaynagi": row.get("skor_kaynagi"),
            },
        )
