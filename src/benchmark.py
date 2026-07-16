# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Benchmark framework (Phase 8).

Compares interchangeable components on the same inputs and exports a report:

* **Generators** — REINVENT4 vs MolMIM (vs Hybrid / heuristic) on runtime,
  molecules produced, uniqueness, scaffold **diversity**, and **ADMET pass rate**.
* **Pose engines** — GNINA vs DiffDock (vs Hybrid Validation) on runtime,
  success rate, and **docking quality** (best/mean affinity, mean confidence).

Everything runs against the Phase 3/5 abstractions (:class:`BaseGenerator`,
:class:`BasePosePredictor`), so a component that needs a GPU or credentials
(MolMIM, DiffDock) is compared exactly like the others — and if it fails
(e.g. no API key) the benchmark records the error and continues rather than
aborting.  The framework is stdlib-only and unit-tested with fakes.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def default_admet_pass_fn() -> Callable[[str, str], bool] | None:
    """ADMET pass predicate using the repo's Lipinski/Veber filter (needs RDKit)."""
    try:
        from admet_filter import lipinski_veber_filter
    except Exception:
        return None

    def _fn(smiles: str, name: str = "mol") -> bool:
        try:
            return bool(lipinski_veber_filter(smiles, name).get("pass"))
        except Exception:
            return False

    return _fn


def _diversity(smiles: list[str]) -> dict[str, Any]:
    try:
        import remedia_score

        return remedia_score.diversity_report([{"smiles": s} for s in smiles])
    except Exception:
        uniq = len(set(smiles))
        return {"molecules": len(smiles), "unique_scaffolds": uniq,
                "diversity_score": round(uniq / len(smiles), 4) if smiles else 0.0,
                "largest_cluster": 0, "scaffold_clusters": {}}


def _admet_pass_rate(smiles: list[str], admet_fn: Callable[[str, str], bool] | None) -> float | None:
    if admet_fn is None or not smiles:
        return None
    passed = 0
    for i, s in enumerate(smiles):
        try:
            if admet_fn(s, f"bench_{i}"):
                passed += 1
        except Exception:
            pass
    return round(passed / len(smiles), 4)


# ----------------------------------------------------------------------
# Report container
# ----------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    kind: str  # "generators" | "pose_engines"
    rows: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    context: dict[str, Any] = field(default_factory=dict)

    def _fieldnames(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        return seen

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = self._fieldnames()
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": self.kind, "created_at": self.created_at,
            "context": self.context, "rows": self.rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        if not self.rows:
            return f"# Benchmark ({self.kind})\n\n_No results._\n"
        fields = self._fieldnames()
        head = "| " + " | ".join(fields) + " |"
        sep = "| " + " | ".join("---" for _ in fields) + " |"
        body = []
        for row in self.rows:
            body.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
        winner = self.winner()
        title = f"# Benchmark — {self.kind}\n\n_Generated {self.created_at}_\n"
        win = f"\n**Öne çıkan:** {winner}\n" if winner else ""
        return title + win + "\n" + "\n".join([head, sep, *body]) + "\n"

    def winner(self) -> str | None:
        ok = [r for r in self.rows if not r.get("error")]
        if not ok:
            return None
        if self.kind == "generators":
            # Prefer more distinct scaffolds (absolute), then ADMET pass rate,
            # then molecules produced. Using the absolute count avoids rewarding
            # a tiny set whose unique/total ratio is trivially 1.0.
            best = max(ok, key=lambda r: (r.get("unique_scaffolds") or 0,
                                          r.get("admet_pass_rate") or 0,
                                          r.get("produced") or 0))
            return best.get("name")
        # pose engines: prefer most-negative best affinity, then success rate
        def key(r):
            aff = r.get("best_affinity")
            aff_rank = -aff if aff is not None else -999
            return (aff_rank, r.get("success_rate") or 0)
        return max(ok, key=key).get("name")

    def export(self, out_dir: str | Path, stem: str = "benchmark") -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.to_csv(out_dir / f"{stem}_{self.kind}.csv")
        json_path = self.to_json(out_dir / f"{stem}_{self.kind}.json")
        md_path = out_dir / f"{stem}_{self.kind}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


# ----------------------------------------------------------------------
# Generator benchmark
# ----------------------------------------------------------------------
def _as_items(components: Any) -> list[tuple[str, Any]]:
    if isinstance(components, dict):
        return list(components.items())
    items = []
    for c in components:
        if isinstance(c, tuple):
            items.append(c)
        else:
            items.append((getattr(c, "name", str(c)), c))
    return items


def run_generator_benchmark(
    generators: Any,
    *,
    target: str | None = None,
    n: int = 20,
    seeds: list[str] | None = None,
    admet_fn: Callable[[str, str], bool] | None = None,
    reporter: Any | None = None,
) -> BenchmarkReport:
    """Benchmark generators (REINVENT4 vs MolMIM vs …) on the same request."""
    if admet_fn is None:
        admet_fn = default_admet_pass_fn()
    report = BenchmarkReport(kind="generators", context={"target": target, "n": n})

    for name, gen in _as_items(generators):
        if reporter is not None:
            reporter.log(f"Benchmark üretici: {name}")
        started = time.monotonic()
        try:
            result = gen.generate(target=target, n=n, seeds=seeds)
            elapsed = round(time.monotonic() - started, 3)
            smiles = list(result.smiles)
            div = _diversity(smiles)
            report.rows.append({
                "name": name,
                "runtime_seconds": elapsed,
                "produced": len(smiles),
                "unique": len(set(smiles)),
                "uniqueness_ratio": round(len(set(smiles)) / len(smiles), 4) if smiles else 0.0,
                "unique_scaffolds": div.get("unique_scaffolds"),
                "diversity_score": div.get("diversity_score"),
                "admet_pass_rate": _admet_pass_rate(smiles, admet_fn),
                "error": "",
            })
        except Exception as exc:
            report.rows.append({
                "name": name,
                "runtime_seconds": round(time.monotonic() - started, 3),
                "produced": 0, "unique": 0, "uniqueness_ratio": 0.0,
                "unique_scaffolds": None, "diversity_score": None,
                "admet_pass_rate": None, "error": str(exc),
            })
            if reporter is not None:
                reporter.warning(f"Benchmark üretici {name} başarısız: {exc}")
    return report


# ----------------------------------------------------------------------
# Pose engine benchmark
# ----------------------------------------------------------------------
def _pose_quality(scores: Iterable[Any]) -> dict[str, Any]:
    affinities = [s.affinity_kcal_mol for s in scores
                  if getattr(s, "success", False) and s.affinity_kcal_mol is not None]
    confidences = [s.confidence for s in scores
                   if getattr(s, "success", False) and s.confidence is not None]
    out: dict[str, Any] = {
        "best_affinity": min(affinities) if affinities else None,
        "mean_affinity": round(sum(affinities) / len(affinities), 3) if affinities else None,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
    }
    return out


def run_pose_benchmark(
    predictors: Any,
    molecules: list[tuple[str, str]],
    *,
    receptor: str | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] | None = None,
    out_dir: Any | None = None,
    reporter: Any | None = None,
) -> BenchmarkReport:
    """Benchmark pose engines (GNINA vs DiffDock vs …) on the same molecules."""
    report = BenchmarkReport(kind="pose_engines",
                             context={"molecules": len(molecules)})
    for name, pred in _as_items(predictors):
        if reporter is not None:
            reporter.log(f"Benchmark poz motoru: {name}")
        started = time.monotonic()
        try:
            result = pred.predict_pose(molecules, receptor=receptor, center=center,
                                       size=size, out_dir=out_dir)
            elapsed = round(time.monotonic() - started, 3)
            scores = result.scores
            success = sum(1 for s in scores if getattr(s, "success", False))
            quality = _pose_quality(scores)
            report.rows.append({
                "name": name,
                "runtime_seconds": elapsed,
                "scored": success,
                "total": len(scores),
                "success_rate": round(success / len(scores), 4) if scores else 0.0,
                **quality,
                "error": "",
            })
        except Exception as exc:
            report.rows.append({
                "name": name,
                "runtime_seconds": round(time.monotonic() - started, 3),
                "scored": 0, "total": len(molecules), "success_rate": 0.0,
                "best_affinity": None, "mean_affinity": None, "mean_confidence": None,
                "error": str(exc),
            })
            if reporter is not None:
                reporter.warning(f"Benchmark poz motoru {name} başarısız: {exc}")
    return report
