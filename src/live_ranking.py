# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Score candidates and **stream** the result as live events (Faz 1).

The composite scoring in :mod:`remedia_score` runs as a single batch after
docking, but the live experiment console wants to reveal candidates one by one —
a molecule feed, a leaderboard that re-sorts as scores arrive, and a running
"leader changed" signal.  This module is the thin, importable, unit-testable seam
between the batch scorer and the :class:`progress.ProgressReporter` event stream,
so that logic lives in ``src/`` rather than in the notebook cell.

It emits the roadmap §6.2 event types ``candidate_generated``,
``candidate_scored`` and ``leader_changed`` (see the §6.3 example payload) and
returns the same ranked list :func:`remedia_score.rank_candidates` would, so the
existing pipeline keeps writing ``remedia_ranking.csv`` unchanged.

Stdlib only; no rdkit/pandas required (delegates chemistry to ``remedia_score``).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from remedia_score import DEFAULT_WEIGHTS, ScoreWeights, rank_candidates


def _name(row: dict[str, Any]) -> str:
    for key in ("molecule", "ligand", "name"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _smiles(row: dict[str, Any]) -> str:
    return str(row.get("smiles") or row.get("canonical_smiles") or "")


def _reason(status: str, score: float | None) -> str:
    if status == "docking_failed":
        return "Docking skoru üretilemedi — pose cezası uygulandı"
    if status == "no_pose":
        return "Docking çalıştırılmadı (pose motoru yok)"
    if score is None:
        return "Skor hesaplanamadı"
    return f"Heuristik skor {score}"


def emit_generated(
    reporter: Any,
    molecules: Sequence[tuple[str, str]],
    *,
    round_index: int | None = None,
) -> None:
    """Emit one ``candidate_generated`` event per freshly generated molecule.

    ``molecules`` is the pipeline's canonical ``[(name, smiles)]`` list.  A
    ``None`` reporter is a no-op so callers need no guard.
    """
    if reporter is None:
        return
    total = len(molecules)
    for index, (name, smiles) in enumerate(molecules, 1):
        payload: dict[str, Any] = {
            "candidate": name,
            "smiles": smiles,
            "index": index,
            "total": total,
        }
        if round_index is not None:
            payload["round"] = round_index
        reporter.event(
            "candidate_generated",
            message=f"Üretildi: {name} ({index}/{total})",
            **payload,
        )


def score_and_stream(
    candidates: list[dict[str, Any]],
    *,
    reporter: Any = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    use_qed: bool = True,
    round_index: int | None = None,
) -> list[dict[str, Any]]:
    """Rank ``candidates`` and stream per-candidate live events.

    Returns exactly what :func:`remedia_score.rank_candidates` returns (so
    downstream CSV writing is unchanged).  When ``reporter`` is provided, emits a
    ``candidate_scored`` event for every candidate — revealed in input
    (generation) order so the console's leaderboard animates — and a
    ``leader_changed`` event whenever the running best improves.
    """
    scored = rank_candidates(candidates, weights=weights, use_qed=use_qed)
    if reporter is None:
        return scored

    by_name: dict[str, dict[str, Any]] = {}
    for row in scored:
        by_name.setdefault(_name(row), row)

    best_score: float | None = None
    best_name: str | None = None

    for cand in candidates:
        name = _name(cand)
        row = by_name.get(name, cand)
        score = row.get("remedia_score")
        status = row.get("docking_status")
        if status is None:
            status = "scored" if score is not None else "no_pose"
        accepted = status != "docking_failed"
        reason = _reason(status, score)

        payload: dict[str, Any] = {
            "candidate": name,
            "smiles": _smiles(row),
            "rank": row.get("rank"),
            "remedia_score": score,
            "pose_score": row.get("pose_score"),
            "admet_score": row.get("admet_score"),
            "druglikeness_score": row.get("druglikeness_score"),
            "diversity_score": row.get("diversity_score"),
            "docking_status": status,
            "affinity_kcal_mol": row.get("affinity_kcal_mol"),
            "accepted": accepted,
            "reason": reason,
        }
        if round_index is not None:
            payload["round"] = round_index
        reporter.event("candidate_scored", message=f"{name}: skor {score}", **payload)

        if score is not None and (best_score is None or score > best_score):
            previous = best_name
            best_score, best_name = score, name
            leader_payload: dict[str, Any] = {
                "candidate": name,
                "remedia_score": score,
                "previous": previous,
            }
            if round_index is not None:
                leader_payload["round"] = round_index
            reporter.event(
                "leader_changed",
                message=f"Yeni lider: {name} ({score})",
                **leader_payload,
            )

    return scored


def split_by_docking(candidates: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition scored candidates into scored / docking_failed / no_pose buckets.

    Used by the report to show docking-less candidates separately (roadmap §12.7).
    """
    buckets: dict[str, list[dict[str, Any]]] = {"scored": [], "docking_failed": [], "no_pose": []}
    for cand in candidates:
        status = cand.get("docking_status")
        if status not in buckets:
            status = "scored" if cand.get("remedia_score") is not None else "no_pose"
        buckets[status].append(cand)
    return buckets
