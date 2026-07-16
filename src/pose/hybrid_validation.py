# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Hybrid Validation pose strategy (Phase 5).

Pipeline::

    DiffDock (all molecules) -> top candidates by confidence -> GNINA confirmation

The two independent methods are combined with the repository's existing
"genel güven" (overall-confidence) logic from ``merge_diffdock_results``: a
molecule that both methods rank strongly is a stronger candidate than one only a
single method likes.

If DiffDock is unavailable, :class:`DiffDockUnavailable` propagates so the caller
can fall back to plain GNINA — GNINA stays fully operational.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .base import BasePosePredictor, PoseResult, PoseScore
from .diffdock_predictor import DiffDockPredictor
from .gnina_predictor import GninaPredictor


class HybridValidationPredictor(BasePosePredictor):
    """DiffDock screen → GNINA confirmation on the top candidates."""

    name = "hybrid_validation"

    def __init__(
        self,
        *,
        diffdock: DiffDockPredictor | None = None,
        gnina: GninaPredictor | None = None,
        top_fraction: float = 0.25,
        top_n: int | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self._diffdock = diffdock or DiffDockPredictor(log_fn=log_fn)
        self._gnina = gnina or GninaPredictor(log_fn=log_fn)
        self.top_fraction = top_fraction
        self.top_n = top_n
        self._log = log_fn

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
        common = dict(receptor=receptor, center=center, size=size, out_dir=out_dir, reporter=reporter)

        # 1) DiffDock over the whole library (may raise DiffDockUnavailable).
        if reporter is not None:
            reporter.log("Hybrid Validation: DiffDock taraması")
        dd = self._diffdock.predict_pose(molecules, **common)
        dd_by = {s.ligand: s for s in dd.scores}

        # 2) Select the top candidates by DiffDock confidence.
        ranked = sorted(
            [s for s in dd.scores if s.confidence is not None],
            key=lambda s: s.confidence, reverse=True,
        )
        count = self.top_n if self.top_n is not None else max(1, math.ceil(len(ranked) * self.top_fraction))
        top_names = {s.ligand for s in ranked[:count]}
        top_molecules = [(n, smi) for n, smi in molecules if n in top_names]
        if reporter is not None:
            reporter.log(f"Hybrid Validation: {len(top_molecules)} aday GNINA ile doğrulanıyor")

        # 3) GNINA confirmation on the top subset.
        gn = self._gnina.predict_pose(top_molecules, **common) if top_molecules else None
        gn_by = {s.ligand: s for s in (gn.scores if gn else [])}

        # 4) Merge with the repository's overall-confidence rule.
        from merge_diffdock_results import genel_guven

        scores: list[PoseScore] = []
        for name, _smiles in molecules:
            dscore = dd_by.get(name)
            gscore = gn_by.get(name)
            confidence = dscore.confidence if dscore else None
            affinity = gscore.affinity_kcal_mol if gscore else None
            confirmed = bool(gscore and gscore.success)
            status = genel_guven(affinity, confidence)
            scores.append(PoseScore(
                ligand=name,
                affinity_kcal_mol=affinity,
                confidence=confidence,
                success=confirmed or (dscore.success if dscore else False),
                source="hybrid_validation",
                extra={
                    "genel_guven_durumu": status,
                    "confirmed_by_gnina": confirmed,
                    "in_top_subset": name in top_names,
                    "skor_kaynagi": (gscore.extra.get("skor_kaynagi") if gscore else None) or "diffdock",
                },
            ))

        rows = [s.to_row() for s in scores]
        return PoseResult(
            engine=self.name,
            scores=scores,
            rows=rows,
            stage_info={
                "diffdock_scored": sum(1 for s in dd.scores if s.success),
                "gnina_confirmed": len(top_molecules),
                "gnina_processes": (gn.stage_info.get("gnina_processes") if gn else 0),
            },
            metadata={
                "top_fraction": self.top_fraction,
                "top_confirmed": len(top_molecules),
                "strong_candidates": sum(
                    1 for s in scores if s.extra.get("genel_guven_durumu") == "GÜÇLÜ ADAY"
                ),
            },
        )
