# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
generate_dashboard.py
results/*.csv sonuçlarını ve config.yaml'ı okuyup dashboard.html'i güncel
verilerle YENİDEN yazar.

Tasarım (dark theme, teal accent, Space Grotesk / IBM Plex Mono) sabit bir
şablonda tutulur (src/dashboard_template.html); bu script yalnızca veri
kısımlarını (affinity bar chart, ADMET tablosu, pocket bilgisi, hero
istatistikleri) doldurur. Bağımlılık eklememek için basit token değişimi
(str.replace) kullanılır — CSS'teki tekli süslü parantezlerle çakışmaz.

Kullanım:
    python generate_dashboard.py \
        --ranking results/final_ranking.csv \
        --admet results/admet_results.csv \
        --config config.yaml \
        --output dashboard.html
"""
import argparse
import csv
import html
from pathlib import Path

import yaml

TEMPLATE_DEFAULT = Path(__file__).resolve().parent / "dashboard_template.html"


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def parse_affinity(value):
    """CSV'deki affinity hücresini float'a çevir; boş/None ise None döndür."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_affinity_bars(ranking_rows):
    """final_ranking.csv'den affinity bar satırlarını (HTML) üretir.

    Bar genişliği en güçlü (en negatif) affinity'ye göre orantılanır;
    renk sınıfı güce göre teal/amber/coral olarak atanır.
    """
    scored = [
        (r["ligand"], parse_affinity(r.get("affinity_kcal_mol")), r.get("skor_kaynagi", "real_docking"))
        for r in ranking_rows
    ]
    scored = [(name, aff, mode) for name, aff, mode in scored if aff is not None]
    if not scored:
        return '      <div class="bar-row"><div class="bar-label">Veri yok</div></div>'

    # En güçlü = en negatif affinity, en üstte.
    scored.sort(key=lambda t: t[1])
    best = abs(scored[0][1]) or 1.0

    rows = []
    for i, (name, aff, mode) in enumerate(scored):
        width = round(abs(aff) / best * 100)
        cls = "" if width >= 85 else (" mid" if width >= 60 else " low")
        if mode == "qed_fallback":
            cls += " qed-fallback"
        tag = '<span class="tag">en güçlü</span>' if i == 0 else ""
        label = html.escape(name.replace("_", " ").title())
        rows.append(
            '      <div class="bar-row">\n'
            f'        <div class="bar-label">{label}{tag}</div>\n'
            f'        <div class="bar-track"><div class="bar-fill{cls}" data-w="{width}"></div></div>\n'
            f'        <div class="bar-value">{aff:g}<span class="unit">kcal/mol</span></div>\n'
            '      </div>'
        )
    return "\n".join(rows)


def build_admet_rows(ranking_rows):
    """final_ranking.csv'den ADMET tablo satırlarını (HTML) üretir."""
    if not ranking_rows:
        return '        <tr><td colspan="3">Veri yok</td></tr>'

    rows = []
    for r in ranking_rows:
        name = html.escape(r["ligand"])
        aff = r.get("affinity_kcal_mol")
        aff_disp = aff if aff not in (None, "", "None") else "-"
        passed = str(r.get("admet_pass")).strip().lower() == "true"
        if passed:
            pill = '<span class="pill pass">Geçti</span>'
        else:
            pill = '<span class="pill fail">Kaldı</span>'
        rows.append(
            f'        <tr><td>{name}</td><td>{html.escape(str(aff_disp))}</td>'
            f'<td>{pill}</td></tr>'
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Dashboard HTML üretici")
    parser.add_argument("--ranking", default="results/final_ranking.csv")
    parser.add_argument("--admet", default="results/admet_results.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--template", default=str(TEMPLATE_DEFAULT))
    parser.add_argument("--output", default="dashboard.html")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ranking_rows = load_csv(args.ranking)

    # --- Türetilen istatistikler ------------------------------------------
    affinities = [
        aff for aff in (parse_affinity(r.get("affinity_kcal_mol")) for r in ranking_rows)
        if aff is not None
    ]
    best_affinity = f"{min(affinities):g}" if affinities else "—"

    n_ligands = len(ranking_rows)
    n_pass = sum(
        1 for r in ranking_rows
        if str(r.get("admet_pass")).strip().lower() == "true"
    )
    admet_ratio = f"{n_pass}/{n_ligands}" if n_ligands else "0/0"

    # --- config.yaml görsel meta verileri ---------------------------------
    dash = config.get("dashboard", {}) or {}
    pocket = dash.get("pocket", {}) or {}
    center = config.get("pocket_center", [0, 0, 0])
    center_str = ", ".join(f"{float(c):.2f}" for c in center)

    replacements = {
        "{{UNIPROT}}": html.escape(str(config.get("uniprot_id", ""))),
        "{{AVG_PLDDT}}": str(dash.get("avg_plddt", "—")),
        "{{ADMET_RATIO}}": admet_ratio,
        "{{BEST_AFFINITY}}": best_affinity,
        "{{POCKET_NAME}}": html.escape(str(dash.get("pocket_name", "Pocket"))),
        "{{N_LIGANDS}}": str(n_ligands),
        "{{EXHAUSTIVENESS}}": str(config.get("exhaustiveness", "—")),
        "{{POCKET_CENTER}}": center_str,
        "{{DRUGGABILITY}}": str(pocket.get("druggability", "—")),
        "{{VOLUME}}": str(pocket.get("volume", "—")),
        "{{APOLAR_SASA}}": str(pocket.get("apolar_sasa", "—")),
        "{{ALPHA_SPHERES}}": str(pocket.get("alpha_spheres", "—")),
        "{{FLEXIBILITY}}": str(pocket.get("flexibility", "—")),
        "{{AFFINITY_BARS}}": build_affinity_bars(ranking_rows),
        "{{ADMET_ROWS}}": build_admet_rows(ranking_rows),
    }

    template = Path(args.template).read_text()
    for token, value in replacements.items():
        template = template.replace(token, value)

    Path(args.output).write_text(template)
    print(f"[OK] Dashboard güncellendi: {args.output}")
    print(f"     {n_ligands} ligand · en iyi affinity {best_affinity} kcal/mol · ADMET {admet_ratio}")


if __name__ == "__main__":
    main()
