# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Pose predictor abstraction for Remedia (Phase 5).

GNINA (physics-based docking) and DiffDock (deep-learning pose prediction), plus
a Hybrid Validation strategy, all implement the same :class:`BasePosePredictor`
interface and return a :class:`PoseResult`.  Downstream ranking/reporting consume
``result.rows`` (docking-score CSV rows) and, in Phase 6, the per-ligand
:class:`PoseScore` (affinity and/or confidence).

GNINA stays fully operational: :class:`~pose.gnina_predictor.GninaPredictor`
returns the exact rows ``gnina_engine`` produced today, so wiring the pipeline
through this interface is behavior-neutral for the default engine.

Stdlib only; concrete predictors import heavy deps lazily.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseScore:
    """A single ligand's pose outcome, engine-agnostic.

    ``affinity_kcal_mol`` is the physics score (GNINA/Vina; more negative =
    stronger).  ``confidence`` is DiffDock's pose confidence (higher = better).
    A record may carry one or both (Hybrid Validation carries both).
    """

    ligand: str
    affinity_kcal_mol: float | None = None
    confidence: float | None = None
    success: bool = False
    source: str = ""
    error: str | None = None
    pose_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """A docking_scores.csv-compatible row (superset of the legacy schema)."""
        row = {
            "ligand": self.ligand,
            "affinity_kcal_mol": self.affinity_kcal_mol,
            "pose_confidence": self.confidence,
            "skor_kaynagi": self.source if self.success else None,
            "pose_engine": self.source,
            "docking_success": self.success,
            "docking_error": self.error,
        }
        row.update(self.extra)
        return row


@dataclass
class PoseResult:
    """Uniform output of every pose predictor."""

    engine: str
    scores: list[PoseScore] = field(default_factory=list)
    #: docking_scores.csv rows. For GNINA these are the exact legacy rows.
    rows: list[dict[str, Any]] = field(default_factory=list)
    stage_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def successful(self) -> list[PoseScore]:
        return [s for s in self.scores if s.success]

    def best(self) -> PoseScore | None:
        scored = [s for s in self.scores if s.success and s.affinity_kcal_mol is not None]
        if scored:
            return min(scored, key=lambda s: s.affinity_kcal_mol)
        conf = [s for s in self.scores if s.success and s.confidence is not None]
        if conf:
            return max(conf, key=lambda s: s.confidence)
        return None


class BasePosePredictor(abc.ABC):
    """Interface every pose engine implements."""

    #: Stable identifier recorded in provenance and used by the factory.
    name: str = "base"

    @abc.abstractmethod
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
        """Predict binding poses/scores for ``molecules`` = ``[(name, smiles), ...]``."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
