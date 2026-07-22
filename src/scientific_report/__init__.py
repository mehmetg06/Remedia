"""Compatibility wrapper that fixes scientific report ADMET data wiring.

The project historically ships ``src/scientific_report.py`` as a module.  This
package intentionally takes import precedence and loads that implementation,
then patches two functions without duplicating the full report generator.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

_LEGACY_PATH = Path(__file__).resolve().parents[1] / "scientific_report.py"
_SPEC = importlib.util.spec_from_file_location("_remedia_scientific_report_impl", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Scientific report implementation could not be loaded: {_LEGACY_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)

_original_load_candidates = _impl.load_candidates
_original_ranking_explanation = _impl.ranking_explanation


def _read_admet(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("ligand") or row.get("molecule") or row.get("name") or "")
        if name:
            result[name] = row
    return result


def _float(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_candidates(root: Path):
    """Load candidates and make ADMET CSV authoritative for property fields."""
    candidates = _original_load_candidates(root)
    admet_by_name = _read_admet(Path(root) / "admet_results.csv")
    for candidate in candidates:
        row = admet_by_name.get(str(candidate.get("molecule") or ""))
        if not row:
            continue
        candidate["mw"] = _float(row.get("MW") or row.get("mw"))
        candidate["logp"] = _float(row.get("LogP") or row.get("logp"))
        candidate["tpsa"] = _float(row.get("TPSA") or row.get("tpsa"))
        candidate["hbd"] = _float(row.get("HBD") or row.get("hbd"))
        candidate["hba"] = _float(row.get("HBA") or row.get("hba"))
        status = row.get("pass") or row.get("admet_status")
        if status not in (None, ""):
            candidate["admet_status"] = status
        violations = row.get("violations") or row.get("reason")
        if violations not in (None, ""):
            candidate["violations"] = violations
    return candidates


def ranking_explanation(candidate: dict[str, Any]) -> str:
    """Explain allowed single Lipinski/Veber violations without contradiction."""
    text = _original_ranking_explanation(candidate)
    status = str(candidate.get("admet_status") or "").strip().lower()
    violations = str(candidate.get("violations") or "").strip()
    passed = status in {"true", "pass", "passed", "geçti", "gecti", "ok"}
    failed = status in {"false", "fail", "failed", "geçmedi", "gecmedi"}

    if passed and violations not in {"", "-"}:
        text = text.replace(
            f"ADMET filtresini geçiyor ({violations} ihlal notu)",
            "ADMET (Lipinski/Veber) ön filtresini geçiyor "
            f"(kuralların izin verdiği tek ihlal: {violations})",
        )
    elif passed:
        text = text.replace(
            "ADMET filtresini geçiyor",
            "ADMET (Lipinski/Veber) ön filtresini ihlalsiz geçiyor",
        )
    elif failed and "ADMET (Lipinski/Veber) ön filtresini geçemiyor" not in text:
        suffix = f" ({violations})" if violations not in {"", "-"} else ""
        text = text.rstrip(".") + f", ADMET (Lipinski/Veber) ön filtresini geçemiyor{suffix}."
    return text


# Patch the implementation module globals used by its own report builders.
_impl.load_candidates = load_candidates
_impl.ranking_explanation = ranking_explanation

# Re-export the implementation's public surface.
for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals().setdefault(_name, getattr(_impl, _name))

__all__ = [name for name in globals() if not name.startswith("_")]
