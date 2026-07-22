# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
generate_dashboard.py
results/*.csv sonuçlarını, config.yaml'ı ve opsiyonel progress.jsonl zamanlama
kaydını okuyup dashboard.html'i güncel verilerle yeniden yazar.

Script yalnızca stdlib + PyYAML kullanır; veri parçalarını HTML şablonundaki
basit token'lara yerleştirir. Dashboard tarafı büyük sonuç setlerinde algılanan
hızı artırmak için ilk açılışta en iyi adayları gösterir, arama/sıralama ile
kalan satırları istemci tarafında filtreler.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

TEMPLATE_DEFAULT = Path(__file__).resolve().parent / "dashboard_template.html"
MAX_INITIAL_ROWS = 25
MAX_BARS = 16


_CSV_CACHE: dict[tuple[str, int, int], list[dict[str, str]]] = {}


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """CSV oku; aynı dosya aynı process içinde tekrar istenirse cache kullan."""
    p = Path(path)
    if not p.exists():
        return []
    stat = p.stat()
    key = (str(p.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _CSV_CACHE.get(key)
    if cached is not None:
        return [dict(row) for row in cached]
    with p.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    _CSV_CACHE.clear()
    _CSV_CACHE[key] = rows
    return [dict(row) for row in rows]


def parse_affinity(value: Any) -> float | None:
    """CSV'deki affinity hücresini float'a çevir; boş/None ise None döndür."""
    if value in (None, "", "None", "—"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any, fallback: str = "—") -> str:
    if value in (None, "", "None"):
        return fallback
    return html.escape(str(value))


def ranked_rows(ranking_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Satırları en güçlü affinity önce gelecek şekilde normalize et."""
    normalized = []
    for index, row in enumerate(ranking_rows):
        aff = parse_affinity(row.get("affinity_kcal_mol"))
        normalized.append({
            **row,
            "_index": index,
            "_affinity": aff,
            "_passed": str(row.get("admet_pass")).strip().lower() == "true",
        })
    return sorted(normalized, key=lambda r: (r["_affinity"] is None, r["_affinity"] or 9999, r["_index"]))


def build_affinity_bars(ranking_rows: list[dict[str, str]]) -> str:
    """final_ranking.csv'den affinity bar satırlarını (HTML) üretir."""
    scored = [
        (r.get("ligand", "ligand"), parse_affinity(r.get("affinity_kcal_mol")), r.get("skor_kaynagi", "real_docking"))
        for r in ranking_rows
    ]
    scored = [(name, aff, mode) for name, aff, mode in scored if aff is not None]
    if not scored:
        return '      <div class="empty-state">Affinity verisi bulunamadı.</div>'

    scored.sort(key=lambda t: t[1])
    best = abs(scored[0][1]) or 1.0
    hidden = max(0, len(scored) - MAX_BARS)

    rows = []
    for i, (name, aff, mode) in enumerate(scored[:MAX_BARS]):
        width = round(abs(aff) / best * 100)
        cls = "" if width >= 85 else (" mid" if width >= 60 else " low")
        if mode == "qed_fallback":
            cls += " qed-fallback"
        tag = '<span class="tag">en güçlü</span>' if i == 0 else ""
        label = html.escape(str(name).replace("_", " ").title())
        rows.append(
            '      <div class="bar-row">\n'
            f'        <div class="bar-label">{label}{tag}</div>\n'
            f'        <div class="bar-track"><div class="bar-fill{cls}" data-w="{width}"></div></div>\n'
            f'        <div class="bar-value">{aff:g}<span class="unit">kcal/mol</span></div>\n'
            '      </div>'
        )
    if hidden:
        rows.append(f'      <div class="chart-note">+{hidden} aday tablo görünümünde listeleniyor.</div>')
    return "\n".join(rows)


def build_admet_rows(ranking_rows: list[dict[str, str]]) -> str:
    """final_ranking.csv'den filtrelenebilir ADMET tablo satırlarını üretir."""
    rows = ranked_rows(ranking_rows)
    if not rows:
        return '        <tr><td colspan="4">Veri yok</td></tr>'

    html_rows = []
    for rank, r in enumerate(rows, start=1):
        name = _safe_text(r.get("ligand"), "ligand")
        aff = r.get("_affinity")
        aff_disp = f"{aff:g}" if aff is not None else "-"
        passed = bool(r.get("_passed"))
        status = "pass" if passed else "fail"
        pill = '<span class="pill pass">Geçti</span>' if passed else '<span class="pill fail">Kaldı</span>'
        initially_hidden = ' data-hidden-initial="true" hidden' if rank > MAX_INITIAL_ROWS else ""
        html_rows.append(
            f'        <tr data-name="{name.lower()}" data-affinity="{aff if aff is not None else 9999}" '
            f'data-status="{status}"{initially_hidden}>'
            f'<td>{rank}</td><td>{name}</td><td>{html.escape(aff_disp)}</td><td>{pill}</td></tr>'
        )
    return "\n".join(html_rows)


def load_timing_summary(path: str | Path | None) -> tuple[str, str]:
    """progress.jsonl'den aşama sürelerini çıkar; yoksa boş durum döndür."""
    if not path:
        return "—", '<div class="empty-state">Zamanlama kaydı bağlanmadı.</div>'
    p = Path(path)
    if not p.exists():
        return "—", '<div class="empty-state">Zamanlama dosyası bulunamadı.</div>'

    first_seen: OrderedDict[str, tuple[str, float]] = OrderedDict()
    last_seen: dict[str, float] = {}
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            stage = str(event.get("stage", ""))
            if not stage:
                continue
            elapsed = float(event.get("elapsed_seconds") or 0)
            label = str(event.get("stage_label") or stage.replace("_", " ").title())
            first_seen.setdefault(stage, (label, elapsed))
            last_seen[stage] = elapsed

    durations = []
    for stage, (label, started) in first_seen.items():
        if stage in {"done", "error"}:
            continue
        duration = max(0.0, last_seen.get(stage, started) - started)
        durations.append((label, duration))
    total = max(last_seen.values()) if last_seen else 0.0
    if not durations and total <= 0:
        return "—", '<div class="empty-state">Zamanlama kaydında ölçülebilir aşama yok.</div>'

    cards = [
        f'<div class="timing-card"><span>{html.escape(label)}</span><b>{duration:.1f}s</b></div>'
        for label, duration in durations
    ]
    return f"{total:.1f}s", "\n".join(cards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard HTML üretici")
    parser.add_argument("--ranking", default="results/final_ranking.csv")
    parser.add_argument("--admet", default="results/admet_results.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timing", default="results/progress.jsonl", help="Opsiyonel progress.jsonl zamanlama kaydı")
    parser.add_argument("--template", default=str(TEMPLATE_DEFAULT))
    parser.add_argument("--output", default="dashboard.html")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    ranking_rows = load_csv(args.ranking)
    ordered = ranked_rows(ranking_rows)

    affinities = [r["_affinity"] for r in ordered if r["_affinity"] is not None]
    best_affinity = f"{min(affinities):g}" if affinities else "—"
    mean_affinity = f"{mean(affinities):.2f}" if affinities else "—"

    n_ligands = len(ranking_rows)
    n_pass = sum(1 for r in ordered if r["_passed"])
    admet_ratio = f"{n_pass}/{n_ligands}" if n_ligands else "0/0"
    pass_pct = f"{round(n_pass / n_ligands * 100)}%" if n_ligands else "0%"
    top_ligand = _safe_text(ordered[0].get("ligand"), "—") if ordered else "—"

    dash = config.get("dashboard", {}) or {}
    pocket = dash.get("pocket", {}) or {}
    center = config.get("pocket_center", [0, 0, 0])
    center_str = ", ".join(f"{float(c):.2f}" for c in center)
    total_runtime, timing_cards = load_timing_summary(args.timing)

    replacements = {
        "{{UNIPROT}}": _safe_text(config.get("uniprot_id", ""), ""),
        "{{AVG_PLDDT}}": _safe_text(dash.get("avg_plddt")),
        "{{ADMET_RATIO}}": admet_ratio,
        "{{ADMET_PASS_PCT}}": pass_pct,
        "{{BEST_AFFINITY}}": best_affinity,
        "{{MEAN_AFFINITY}}": mean_affinity,
        "{{TOP_LIGAND}}": top_ligand,
        "{{TOTAL_RUNTIME}}": total_runtime,
        "{{TIMING_CARDS}}": timing_cards,
        "{{POCKET_NAME}}": _safe_text(dash.get("pocket_name", "Pocket")),
        "{{N_LIGANDS}}": str(n_ligands),
        "{{INITIAL_ROWS}}": str(min(n_ligands, MAX_INITIAL_ROWS)),
        "{{EXHAUSTIVENESS}}": _safe_text(config.get("exhaustiveness")),
        "{{POCKET_CENTER}}": center_str,
        "{{DRUGGABILITY}}": _safe_text(pocket.get("druggability")),
        "{{VOLUME}}": _safe_text(pocket.get("volume")),
        "{{APOLAR_SASA}}": _safe_text(pocket.get("apolar_sasa")),
        "{{ALPHA_SPHERES}}": _safe_text(pocket.get("alpha_spheres")),
        "{{FLEXIBILITY}}": _safe_text(pocket.get("flexibility")),
        "{{AFFINITY_BARS}}": build_affinity_bars(ranking_rows),
        "{{ADMET_ROWS}}": build_admet_rows(ranking_rows),
    }

    template = Path(args.template).read_text(encoding="utf-8")
    for token, value in replacements.items():
        template = template.replace(token, value)

    Path(args.output).write_text(template, encoding="utf-8")
    print(f"[OK] Dashboard güncellendi: {args.output}")
    print(f"     {n_ligands} ligand · en iyi affinity {best_affinity} kcal/mol · ADMET {admet_ratio}")


if __name__ == "__main__":
    main()
