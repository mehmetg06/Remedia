# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Composite **Remedia Score** (Phase 6).

Ranking used to rely on docking affinity alone.  The Remedia Score combines four
independent, interpretable components into a single 0–1 value (higher is better):

* **Pose** — GNINA affinity (more negative = better) and/or DiffDock confidence.
* **ADMET** — the Lipinski/Veber drug-likeness filter outcome.
* **Drug-likeness** — desirability of physicochemical properties (QED if RDKit is
  available, otherwise a rule-based desirability from MW/LogP/TPSA/HBD/HBA).
* **Diversity** — rewards under-represented scaffolds so the top list is not all
  near-duplicates.

Design:

* **Docking-only ranking stays available as a fallback** (:func:`docking_only_rank`)
  and is used automatically when no component beyond affinity is present.
* Components that are missing for a candidate are dropped and the remaining
  weights re-normalised, so a partial run still yields a sensible score.
* Stdlib only; RDKit (for QED and Murcko scaffolds) is imported lazily and the
  module degrades gracefully without it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ScoreWeights:
    pose: float = 0.40
    admet: float = 0.20
    druglikeness: float = 0.20
    diversity: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {"pose": self.pose, "admet": self.admet,
                "druglikeness": self.druglikeness, "diversity": self.diversity}


DEFAULT_WEIGHTS = ScoreWeights()


# -- small helpers ---------------------------------------------------------
def _get(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        # case-insensitive fallback
        for existing, value in row.items():
            if existing.lower() == key.lower() and value not in (None, ""):
                return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _truthy(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed", "geçti", "gecti", "ok"}


def _violation_count(value: Any) -> int:
    if value in (None, "", "-"):
        return 0
    tokens = [t for t in re.split(r"[;,]", str(value)) if t.strip() and t.strip() != "-"]
    return len(tokens)


# -- individual component scores ------------------------------------------
def _minmax(values: list[float], invert: bool) -> dict[int, float]:
    """Map values to 0–1 by min-max. ``invert`` True → smaller value scores higher."""
    finite = [v for v in values if v is not None]
    if not finite:
        return {}
    lo, hi = min(finite), max(finite)
    out: dict[int, float] = {}
    for i, v in enumerate(values):
        if v is None:
            continue
        if hi == lo:
            out[i] = 1.0
        else:
            frac = (v - lo) / (hi - lo)
            out[i] = (1.0 - frac) if invert else frac
    return out


def druglikeness_desirability(row: dict[str, Any]) -> float | None:
    """Rule-based 0–1 desirability from physicochemical properties (no RDKit)."""
    mw = _to_float(_get(row, "mw", "molecular_weight", "molwt", "MW"))
    logp = _to_float(_get(row, "logp", "mol_logp", "LogP"))
    tpsa = _to_float(_get(row, "tpsa", "psa", "TPSA"))
    hbd = _to_float(_get(row, "hbd", "num_h_donors", "HBD"))
    hba = _to_float(_get(row, "hba", "num_h_acceptors", "HBA"))
    rotb = _to_float(_get(row, "rotb", "rotatable_bonds", "RotB"))

    parts: list[float] = []

    def window(x: float, lo: float, hi: float, slack: float) -> float:
        if lo <= x <= hi:
            return 1.0
        dist = (lo - x) if x < lo else (x - hi)
        return max(0.0, 1.0 - dist / slack)

    if mw is not None:
        parts.append(window(mw, 150, 500, 250))
    if logp is not None:
        parts.append(window(logp, -0.4, 5.0, 3.0))
    if tpsa is not None:
        parts.append(window(tpsa, 0, 140, 60))
    if hbd is not None:
        parts.append(1.0 if hbd <= 5 else max(0.0, 1 - (hbd - 5) / 5))
    if hba is not None:
        parts.append(1.0 if hba <= 10 else max(0.0, 1 - (hba - 10) / 5))
    if rotb is not None:
        parts.append(1.0 if rotb <= 10 else max(0.0, 1 - (rotb - 10) / 5))

    if not parts:
        return None
    return round(sum(parts) / len(parts), 4)


def _qed(smiles: str) -> float | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(QED.qed(mol))
    except Exception:
        return None


def scaffold_of(smiles: str) -> str:
    """Murcko scaffold (RDKit) or a coarse fallback signature without RDKit."""
    if not smiles:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
    except Exception:
        pass
    # Fallback: ring-signature proxy so diversity still works without RDKit.
    rings = smiles.count("1") + smiles.count("2") + smiles.count("3")
    aromatic = sum(smiles.count(c) for c in "cnos")
    length_bucket = len(smiles) // 5
    return f"proxy:r{rings}:a{aromatic}:l{length_bucket}"


def admet_score(row: dict[str, Any]) -> float | None:
    passed = _truthy(_get(row, "admet_pass", "pass", "passed", "admet_status"))
    violations = _get(row, "violations", "ihlal", "fails")
    if passed is None and violations is None:
        return None
    v = _violation_count(violations)
    base = 1.0 if (passed or passed is None) else 0.4
    score = base - 0.2 * v
    return round(max(0.0, min(1.0, score)), 4)


# -- composite -------------------------------------------------------------
def compute_scores(
    candidates: list[dict[str, Any]],
    *,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    use_qed: bool = True,
) -> list[dict[str, Any]]:
    """Return candidates enriched with subscores + ``remedia_score``, ranked.

    Each returned dict includes: ``pose_score``, ``admet_score``,
    ``druglikeness_score``, ``diversity_score``, ``remedia_score`` (0–1), a
    ``score_components`` breakdown, ``scaffold`` and ``rank``.
    """
    n = len(candidates)
    if n == 0:
        return []

    affinities = [_to_float(_get(c, "affinity_kcal_mol", "accurate_affinity_kcal_mol",
                                 "fast_affinity_kcal_mol")) for c in candidates]
    confidences = [_to_float(_get(c, "pose_confidence", "confidence", "diffdock_confidence"))
                   for c in candidates]
    aff_norm = _minmax(affinities, invert=True)   # more negative → higher
    conf_norm = _minmax(confidences, invert=False)  # higher → higher

    # scaffolds + counts for the diversity component
    scaffolds = [scaffold_of(str(_get(c, "smiles", "canonical_smiles") or "")) for c in candidates]
    counts: dict[str, int] = {}
    for s in scaffolds:
        if s:
            counts[s] = counts.get(s, 0) + 1

    # "Pose expected" = at least one molecule in this batch actually produced an
    # affinity or a confidence.  Used to tell a *docking failure* (some molecules
    # docked, this one did not) apart from a *pose-free run* (no pose engine at
    # all), so only genuine failures are penalised.
    pose_expected = bool(aff_norm) or bool(conf_norm)

    enriched: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        out = dict(cand)
        out["scaffold"] = scaffolds[i]

        # pose — the measured value; ``None`` when this molecule produced no
        # affinity/confidence of its own.
        pose_parts = []
        if i in aff_norm:
            pose_parts.append(aff_norm[i])
        if i in conf_norm:
            pose_parts.append(conf_norm[i])
        pose = round(sum(pose_parts) / len(pose_parts), 4) if pose_parts else None

        # Docking status + penalty.  A molecule that fails docking must NOT be
        # rescued to the top by ADMET/drug-likeness/diversity alone.  When the
        # batch produced poses at all (``pose_expected``) but this molecule has
        # none — or its row is explicitly flagged ``docking_success=False`` — the
        # pose component is fed to the weighted mean as ``0.0`` rather than
        # dropped, so it stays in the denominator and the score ceiling falls by
        # the pose weight.  When *no* molecule docked (pose engine absent) the
        # pose weight is renormalised out as before, so a pose-free run still
        # ranks sensibly.
        explicit_fail = _truthy(_get(cand, "docking_success")) is False
        if pose_expected and (pose is None or explicit_fail):
            docking_status = "docking_failed"
            pose_component = 0.0
        elif pose is None:
            docking_status = "no_pose"
            pose_component = None
        else:
            docking_status = "scored"
            pose_component = pose

        # admet
        adm = admet_score(cand)

        # drug-likeness (QED preferred when available)
        smi = str(_get(cand, "smiles", "canonical_smiles") or "")
        dl = _qed(smi) if (use_qed and smi) else None
        if dl is None:
            dl = druglikeness_desirability(cand)
        dl = round(dl, 4) if dl is not None else None

        # diversity: rarer scaffold → higher
        div = round(1.0 / counts[scaffolds[i]], 4) if scaffolds[i] and counts.get(scaffolds[i]) else None

        components = {"pose": pose_component, "admet": adm, "druglikeness": dl, "diversity": div}
        out.update({
            "pose_score": pose,
            "admet_score": adm,
            "druglikeness_score": dl,
            "diversity_score": div,
            "docking_status": docking_status,
            "score_components": components,
            "remedia_score": _weighted(components, weights),
        })
        enriched.append(out)

    # Sort by score (higher first); a docking failure sinks below an otherwise
    # equal scored candidate, and ``None`` scores sink last.
    enriched.sort(key=lambda c: (
        c["remedia_score"] is None,
        -(c["remedia_score"] or 0.0),
        c.get("docking_status") == "docking_failed",
    ))
    for rank, item in enumerate(enriched, 1):
        item["rank"] = rank
    return enriched


def _weighted(components: dict[str, float | None], weights: ScoreWeights) -> float | None:
    w = weights.as_dict()
    num = 0.0
    den = 0.0
    for key, value in components.items():
        if value is None:
            continue
        num += w[key] * value
        den += w[key]
    if den == 0:
        return None
    return round(num / den, 4)


def docking_only_rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy fallback: rank by affinity (ADMET pass first), unchanged behavior."""
    def key(c: dict[str, Any]):
        passed = _truthy(_get(c, "admet_pass", "pass", "admet_status"))
        aff = _to_float(_get(c, "affinity_kcal_mol"))
        return (passed is False, aff if aff is not None else 999.0)

    ranked = sorted((dict(c) for c in candidates), key=key)
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return ranked


def rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    use_qed: bool = True,
) -> list[dict[str, Any]]:
    """Rank by Remedia Score; fall back to docking-only when no component beyond
    affinity is present."""
    scored = compute_scores(candidates, weights=weights, use_qed=use_qed)
    # Use the composite only when a component beyond pose/affinity exists;
    # otherwise fall back to the legacy docking-only ranking.
    has_composite = any(
        c.get("admet_score") is not None or c.get("druglikeness_score") is not None
        or c.get("diversity_score") is not None
        for c in scored
    )
    if not has_composite:
        return docking_only_rank(candidates)
    return scored


def diversity_report(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Set-level scaffold diversity (also used by the Phase 7.5 report)."""
    scaffolds = [scaffold_of(str(_get(c, "smiles", "canonical_smiles") or "")) for c in candidates]
    scaffolds = [s for s in scaffolds if s]
    total = len(scaffolds)
    clusters: dict[str, int] = {}
    for s in scaffolds:
        clusters[s] = clusters.get(s, 0) + 1
    unique = len(clusters)
    return {
        "molecules": total,
        "unique_scaffolds": unique,
        "diversity_score": round(unique / total, 4) if total else 0.0,
        "largest_cluster": max(clusters.values()) if clusters else 0,
        "scaffold_clusters": clusters,
    }


RANKING_FIELDS = [
    "rank", "ligand", "molecule", "smiles", "remedia_score",
    "pose_score", "admet_score", "druglikeness_score", "diversity_score",
    "docking_status", "affinity_kcal_mol", "pose_confidence", "admet_pass",
    "violations", "scaffold",
]


def write_ranking_csv(scored: list[dict[str, Any]], path: str | Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in scored:
            merged = {"molecule": _get(row, "molecule", "ligand"),
                      "ligand": _get(row, "ligand", "molecule")}
            merged.update({k: row.get(k) for k in RANKING_FIELDS if k in row})
            writer.writerow(merged)
    return path
