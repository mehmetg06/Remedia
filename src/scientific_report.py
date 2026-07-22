# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Scientist-friendly result report (Phases 7 and 7.5).

Layered **additively** on top of ``result_report.build_result_package`` — it
never deletes or rewrites raw pipeline files, only adds explanatory artefacts
into the ``00_REMEDIA_REPORT`` folder:

    README_FIRST.txt      plain-language guide to the whole package
    report.html           scientist report: candidate cards + analysis + figures
    report.pdf            (optional) rendered when matplotlib is available
    candidate_overview.csv rich per-candidate table
    pipeline_log.txt      full run log (nothing hidden)
    run_manifest.json     provenance: seeds, versions, parameters, scores

For every top candidate the report shows a **Candidate Card** (docking score,
MW/LogP/TPSA/HBA/HBD, SMILES, 2D structure), a **binding analysis**, an
automatic **"why this molecule ranked highly"** explanation, a **similarity**
comparison against known ligands, a set-level **diversity analysis**, an
**executive summary**, and **publication figures**.

Heavy dependencies (RDKit for structures/QED/fingerprints, matplotlib for
figures/PDF) are imported lazily; every one degrades gracefully so the core
report is always produced.  The module itself imports with the stdlib only.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPORT_DIR_NAME = "00_REMEDIA_REPORT"
TOP_N = 12

#: User-facing name for the composite score.  It is NOT a trained model and NOT a
#: precise measurement — a temporary, fixed-weight heuristic ranking component
#: (roadmap §2.5/§12.6), so it is labelled and shown with its sub-scores rather
#: than as one artificially precise number.
SCORE_LABEL = "Geçici Heuristik Skor (v0)"
#: Mandatory scientific disclaimer (roadmap §8), shown verbatim in every report.
DISCLAIMER = ("Bu sonuçlar hesaplamalı tahmindir; deneysel aktivite, toksisite "
              "veya klinik uygunluk kanıtı değildir.")
#: docking_status values that mean "no independent docking score for this molecule".
UNDOCKED_STATUSES = ("docking_failed", "no_pose")


def _score_band(score: Any) -> str:
    """Coarse qualitative band, so the score is not read as false precision."""
    if score is None:
        return "—"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "—"
    if s >= 0.75:
        return "yüksek"
    if s >= 0.5:
        return "orta"
    return "düşük"


def _is_undocked(cand: dict[str, Any]) -> bool:
    status = str(cand.get("docking_status") or "")
    if status:
        return status in UNDOCKED_STATUSES
    # Older runs without docking_status: treat a missing affinity as undocked.
    return cand.get("affinity_kcal_mol") is None and cand.get("pose_confidence") is None


def _docking_reason(cand: dict[str, Any]) -> str:
    status = str(cand.get("docking_status") or "")
    if status == "docking_failed":
        return "Docking skoru üretilemedi (pose cezası uygulandı)"
    if status == "no_pose":
        return "Docking çalıştırılmadı"
    return "Bağımsız docking skoru yok"


# ======================================================================
# Loading + enrichment
# ======================================================================
def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_candidates(root: Path) -> list[dict[str, Any]]:
    """Merge the composite ranking with physicochemical props into one table.

    Prefers ``remedia_ranking.csv`` (Phase 6 scores) joined with the report's
    ``candidate_overview.csv`` / ``admet_results.csv`` for MW/LogP/etc.  Falls
    back to whatever candidate table exists so a report is always produced.
    """
    report_dir = root / REPORT_DIR_NAME
    ranking = _read_csv(root / "remedia_ranking.csv")
    overview = _read_csv(report_dir / "candidate_overview.csv")
    admet = _read_csv(root / "admet_results.csv")

    props: dict[str, dict[str, Any]] = {}
    for row in overview:
        name = str(_first(row, "molecule", "ligand", "name") or "")
        if name:
            props[name] = row
    for row in admet:
        name = str(_first(row, "ligand", "molecule", "name") or "")
        if name and name not in props:
            props[name] = row

    base = ranking or overview
    if not base:
        # Last resort: reuse result_report discovery.
        try:
            import result_report

            tables = result_report._discover_tables(root, report_dir)
            primary = max(tables, key=lambda t: t["score"], default=None)
            base = result_report._normalise_candidates(primary["rows"]) if primary else []
        except Exception:
            base = []

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(base, 1):
        name = str(_first(row, "molecule", "ligand", "name") or f"mol_{index:03d}")
        p = props.get(name, {})
        candidates.append({
            "rank": int(_f(_first(row, "rank")) or index),
            "molecule": name,
            "smiles": str(_first(row, "smiles", "canonical_smiles") or _first(p, "smiles") or ""),
            "remedia_score": _f(_first(row, "remedia_score")),
            "pose_score": _f(_first(row, "pose_score")),
            "admet_score": _f(_first(row, "admet_score")),
            "druglikeness_score": _f(_first(row, "druglikeness_score")),
            "diversity_score": _f(_first(row, "diversity_score")),
            "affinity_kcal_mol": _f(_first(row, "affinity_kcal_mol", "accurate_affinity_kcal_mol",
                                            "fast_affinity_kcal_mol")),
            "pose_confidence": _f(_first(row, "pose_confidence", "confidence", "diffdock_confidence")),
            "admet_status": _first(row, "admet_pass", "admet_status") or _first(p, "pass", "admet_status") or "",
            "docking_status": _first(row, "docking_status") or "",
            "violations": _first(row, "violations") or _first(p, "violations") or "",
            "mw": _f(_first(row, "mw", "MW") or _first(p, "MW", "mw")),
            "logp": _f(_first(row, "logp", "LogP") or _first(p, "LogP", "logp")),
            "tpsa": _f(_first(row, "tpsa", "TPSA") or _first(p, "TPSA", "tpsa")),
            "hbd": _f(_first(row, "hbd", "HBD") or _first(p, "HBD", "hbd")),
            "hba": _f(_first(row, "hba", "HBA") or _first(p, "HBA", "hba")),
            "scaffold": _first(row, "scaffold") or "",
        })

    # Ensure ranked order (by remedia_score if present, else keep existing rank).
    if any(c["remedia_score"] is not None for c in candidates):
        candidates.sort(key=lambda c: (c["remedia_score"] is None, -(c["remedia_score"] or 0)))
        for i, c in enumerate(candidates, 1):
            c["rank"] = i
    return candidates


# ======================================================================
# Narrative generation
# ======================================================================
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ranking_explanation(cand: dict[str, Any]) -> str:
    """Automatic 'why this molecule ranked highly' sentence."""
    bits: list[str] = []
    aff = cand.get("affinity_kcal_mol")
    conf = cand.get("pose_confidence")
    if aff is not None:
        strength = "güçlü" if aff <= -8 else "orta" if aff <= -6 else "zayıf"
        bits.append(f"{strength} öngörülen bağlanma (GNINA {aff:.2f} kcal/mol)")
    if conf is not None:
        bits.append(f"DiffDock güven skoru {conf:.2f}")
    admet = str(cand.get("admet_status") or "").lower()
    viol = str(cand.get("violations") or "").strip()
    if admet in {"true", "pass", "geçti", "gecti", "ok", "passed"}:
        bits.append("ADMET filtresini geçiyor" + (f" ({viol} ihlal notu)" if viol not in ("", "-") else ""))
    dl = cand.get("druglikeness_score")
    if dl is not None:
        quality = "iyi" if dl >= 0.7 else "orta" if dl >= 0.5 else "zayıf"
        bits.append(f"{quality} ilaç-benzerliği ({dl:.2f})")
    div = cand.get("diversity_score")
    if div is not None and div >= 0.9:
        bits.append("özgün bir kimyasal iskelet")
    if _is_undocked(cand):
        bits.append(_docking_reason(cand).lower())
    score = cand.get("remedia_score")
    if score is not None:
        head = f"#{cand.get('rank')} sırada, {SCORE_LABEL} {score:.2f} ({_score_band(score)})"
    else:
        head = f"#{cand.get('rank')} sırada"
    if not bits:
        return f"{head}."
    return f"{head}: " + ", ".join(bits) + "."


def binding_analysis(cand: dict[str, Any], pocket_center: Any) -> str:
    """Plain-language binding interpretation (residue-level if available)."""
    aff = cand.get("affinity_kcal_mol")
    residues = cand.get("interacting_residues")  # populated only if a pose parser provides it
    if residues:
        return (f"Öngörülen etkileşen rezidüler: {residues}. "
                "Bu rezidüler bağlanma cebinde temas noktalarıdır.")
    center_txt = ""
    if pocket_center:
        try:
            cx, cy, cz = (round(float(x), 1) for x in pocket_center)
            center_txt = f" Cep merkezi ≈ ({cx}, {cy}, {cz})."
        except Exception:
            center_txt = ""
    if aff is None:
        return "Bu aday için bağlanma skoru okunamadı; poz analizi yapılamıyor." + center_txt
    tone = "güçlü" if aff <= -8 else "orta düzey" if aff <= -6 else "zayıf"
    return (f"Molekül, hedef cebe {tone} bir yerleşim skoru ({aff:.2f} kcal/mol) gösteriyor."
            f"{center_txt} Rezidü-düzeyi etkileşim çıktısı bu çalışmada üretilmedi; "
            "deneysel doğrulama önerilir.")


def _fallback_similarity(a: str, b: str) -> float:
    import difflib

    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def similarity_analysis(
    candidates: list[dict[str, Any]],
    known_ligands: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Nearest known ligand + similarity per candidate (Tanimoto if RDKit)."""
    known = [k for k in (known_ligands or []) if k.get("smiles")]
    result: dict[str, dict[str, Any]] = {}
    if not known:
        return result

    fps = None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs  # noqa: F401

        def fp(smi):
            mol = Chem.MolFromSmiles(smi)
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None

        known_fps = [(k.get("name") or "known", fp(k["smiles"])) for k in known]
        fps = True
    except Exception:
        fps = None

    for cand in candidates:
        smi = cand.get("smiles") or ""
        best_name, best_sim = None, 0.0
        if fps:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem, DataStructs

                mol = Chem.MolFromSmiles(smi)
                cfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None
                for kname, kfp in known_fps:
                    if cfp is None or kfp is None:
                        continue
                    sim = DataStructs.TanimotoSimilarity(cfp, kfp)
                    if sim > best_sim:
                        best_sim, best_name = sim, kname
            except Exception:
                fps = None
        if not fps:
            for k in known:
                sim = _fallback_similarity(smi, k["smiles"])
                if sim > best_sim:
                    best_sim, best_name = sim, k.get("name") or "known"
        result[cand["molecule"]] = {
            "nearest_known": best_name,
            "similarity": round(best_sim, 3),
            "method": "tanimoto" if fps else "string-proxy",
        }
    return result


def executive_summary(
    candidates: list[dict[str, Any]],
    diversity: dict[str, Any],
    target: str,
) -> str:
    scored = [c for c in candidates if c.get("affinity_kcal_mol") is not None]
    passed = [c for c in candidates if str(c.get("admet_status") or "").lower()
              in {"true", "pass", "passed", "geçti", "gecti", "ok"}]
    undocked = [c for c in candidates if _is_undocked(c)]
    best = candidates[0] if candidates else None
    lines = [
        f"Hedef {target} için {len(candidates)} aday değerlendirildi; "
        f"{len(scored)} tanesi için bağlanma skoru, {len(passed)} tanesi için ADMET geçişi elde edildi.",
    ]
    if best:
        lines.append(
            f"En yüksek sıralı aday {best['molecule']} "
            f"({SCORE_LABEL} {_fmt(best.get('remedia_score'), 2)} · {_score_band(best.get('remedia_score'))}, "
            f"GNINA {_fmt(best.get('affinity_kcal_mol'), 2)} kcal/mol)."
        )
    if undocked:
        lines.append(
            f"{len(undocked)} aday için bağımsız docking skoru üretilemedi; bunlar "
            "cezalandırıldı ve raporda ayrı listelendi."
        )
    lines.append(
        f"Kimyasal çeşitlilik: {diversity.get('unique_scaffolds', 0)} benzersiz iskelet / "
        f"{diversity.get('molecules', 0)} molekül (çeşitlilik skoru {diversity.get('diversity_score', 0)})."
    )
    lines.append(
        "Bu sonuçlar hesaplamalı bir ön elemedir; deneysel yapı, kontroller ve "
        "laboratuvar doğrulaması olmadan bağlanma/etkinlik kanıtı değildir."
    )
    return " ".join(lines)


# ======================================================================
# Figures (matplotlib / rdkit optional)
# ======================================================================
def _figure_structure_grid(candidates: list[dict[str, Any]], out: Path) -> bool:
    usable = [c for c in candidates if c.get("smiles")][:TOP_N]
    if not usable:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mols, legends = [], []
        for c in usable:
            mol = Chem.MolFromSmiles(c["smiles"])
            if mol is None:
                continue
            mols.append(mol)
            legends.append(f"#{c['rank']} {c['molecule']}\nscore {_fmt(c.get('remedia_score'))}")
        if not mols:
            return False
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(340, 260), legends=legends)
        img.save(out)
        return True
    except Exception:
        return False


def _figure_distribution(values: list[float], title: str, xlabel: str, out: Path) -> bool:
    values = [v for v in values if v is not None]
    if not values:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(values, bins=min(20, max(4, len(values))), color="#3b3b3b", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Molekül sayısı")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return True
    except Exception:
        return False


def _figure_scaffold_diversity(diversity: dict[str, Any], out: Path) -> bool:
    clusters = diversity.get("scaffold_clusters") or {}
    if not clusters:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        items = sorted(clusters.items(), key=lambda kv: kv[1], reverse=True)[:12]
        labels = [f"S{i+1}" for i in range(len(items))]
        counts = [c for _, c in items]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(labels, counts, color="#3b3b3b")
        ax.set_title("İskelet kümeleri (en kalabalık 12)")
        ax.set_ylabel("Molekül sayısı")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return True
    except Exception:
        return False


def build_figures(candidates: list[dict[str, Any]], diversity: dict[str, Any], report_dir: Path) -> dict[str, str]:
    figures: dict[str, str] = {}
    if _figure_structure_grid(candidates, report_dir / "fig_candidate_grid.png"):
        figures["candidate_grid"] = "fig_candidate_grid.png"
    if _figure_distribution([c.get("affinity_kcal_mol") for c in candidates],
                            "Docking skoru dağılımı", "GNINA (kcal/mol)",
                            report_dir / "fig_docking_distribution.png"):
        figures["docking_distribution"] = "fig_docking_distribution.png"
    if _figure_distribution([c.get("mw") for c in candidates],
                            "Molekül ağırlığı dağılımı", "MW (g/mol)",
                            report_dir / "fig_mw_distribution.png"):
        figures["mw_distribution"] = "fig_mw_distribution.png"
    if _figure_distribution([c.get("logp") for c in candidates],
                            "LogP dağılımı", "LogP",
                            report_dir / "fig_logp_distribution.png"):
        figures["logp_distribution"] = "fig_logp_distribution.png"
    if _figure_scaffold_diversity(diversity, report_dir / "fig_scaffold_diversity.png"):
        figures["scaffold_diversity"] = "fig_scaffold_diversity.png"
    return figures


def _write_pdf(candidates: list[dict[str, Any]], figures: dict[str, str], report_dir: Path,
               target: str, summary: str) -> str | None:
    """Render a simple multi-page PDF via matplotlib when available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import textwrap
    except Exception:
        return None
    try:
        pdf_path = report_dir / "report.pdf"
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.text(0.08, 0.92, f"Remedia Raporu — {target}", fontsize=20, weight="bold")
            wrapped = "\n".join(textwrap.wrap(summary, 90))
            fig.text(0.08, 0.80, wrapped, fontsize=10, va="top")
            rows = [["#", "Molekül", "Score", "GNINA", "MW", "LogP"]]
            for c in candidates[:15]:
                rows.append([str(c.get("rank")), c.get("molecule", "")[:16],
                             _fmt(c.get("remedia_score")), _fmt(c.get("affinity_kcal_mol"), 2),
                             _fmt(c.get("mw"), 1), _fmt(c.get("logp"), 2)])
            ax = fig.add_axes([0.08, 0.08, 0.84, 0.55])
            ax.axis("off")
            table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            pdf.savefig(fig)
            plt.close(fig)
            for _key, fname in figures.items():
                fpath = report_dir / fname
                if not fpath.exists():
                    continue
                try:
                    import matplotlib.image as mpimg

                    img = mpimg.imread(str(fpath))
                    f2 = plt.figure(figsize=(8.27, 6))
                    a2 = f2.add_axes([0.05, 0.05, 0.9, 0.9])
                    a2.imshow(img)
                    a2.axis("off")
                    pdf.savefig(f2)
                    plt.close(f2)
                except Exception:
                    continue
        return str(pdf_path)
    except Exception:
        return None


# ======================================================================
# HTML
# ======================================================================
def _card_html(cand: dict[str, Any], explanation: str, binding: str, sim: dict[str, Any] | None) -> str:
    def cell(label, value):
        return f"<div class='prop'><span>{label}</span><b>{html.escape(_fmt(value))}</b></div>"

    sim_html = ""
    if sim:
        sim_html = (f"<p class='sim'>En yakın bilinen ligand: "
                    f"<b>{html.escape(str(sim.get('nearest_known') or '—'))}</b> "
                    f"(benzerlik {sim.get('similarity')}, {sim.get('method')})</p>")
    score = cand.get('remedia_score')
    return f"""<div class="card">
<div class="card-head"><span class="rank">#{cand.get('rank')}</span>
<span class="mol">{html.escape(str(cand.get('molecule')))}</span>
<span class="score" title="{html.escape(SCORE_LABEL)} — eğitilmiş model değil">Heuristik v0 {html.escape(_fmt(score, 2))} · {_score_band(score)}</span></div>
<div class="subscores">
{cell('Pose alt-skor', cand.get('pose_score'))}
{cell('ADMET alt-skor', cand.get('admet_score'))}
{cell('İlaç-benzerliği', cand.get('druglikeness_score'))}
{cell('Çeşitlilik', cand.get('diversity_score'))}
</div>
<div class="props">
{cell('GNINA kcal/mol', cand.get('affinity_kcal_mol'))}
{cell('Pose conf.', cand.get('pose_confidence'))}
{cell('MW', cand.get('mw'))}
{cell('LogP', cand.get('logp'))}
{cell('TPSA', cand.get('tpsa'))}
{cell('HBD', cand.get('hbd'))}
{cell('HBA', cand.get('hba'))}
{cell('Docking durumu', cand.get('docking_status') or '—')}
</div>
<p class="smiles">{html.escape(str(cand.get('smiles') or '—'))}</p>
<p class="why"><b>Neden yüksek sıralandı?</b> {html.escape(explanation)}</p>
<p class="binding"><b>Bağlanma analizi:</b> {html.escape(binding)}</p>
{sim_html}
</div>"""


def _undocked_section(undocked: list[dict[str, Any]]) -> str:
    """Separate section for candidates without an independent docking score.

    Roadmap §5/§12.7: docking is not the main ranking motor and its failures must
    be *shown*, not silently absorbed into the score.
    """
    if not undocked:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(c.get('molecule')))}</td>"
        f"<td>{html.escape(_docking_reason(c))}</td>"
        f"<td>{html.escape(_fmt(c.get('remedia_score'), 2))} ({_score_band(c.get('remedia_score'))})</td>"
        f"<td>{html.escape(_fmt(c.get('admet_score'), 2))}</td>"
        f"<td class='smiles'>{html.escape(str(c.get('smiles') or '—'))}</td></tr>"
        for c in undocked
    )
    return (
        f"<section class=\"section\"><h2>Docking sonucu olmayan / doğrulanamayan adaylar "
        f"({len(undocked)})</h2>"
        "<p class=\"muted\">Bu adaylar için bağımsız docking kontrolü bir skor üretmedi. "
        "Ana sıralamada pose bileşeni cezalandırılır (skor tavanı düşer) ve buraya ayrılırlar; "
        "docking/pose araçları bir <b>bağımsız fiziksel kontrol</b>dür, bağlanma veya etkinlik "
        "kanıtı değildir.</p>"
        "<table class=\"undocked\"><thead><tr><th>Molekül</th><th>Durum</th>"
        f"<th>{html.escape(SCORE_LABEL)}</th><th>ADMET alt-skor</th><th>SMILES</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def build_html(
    *,
    target: str,
    created_at: str,
    summary: str,
    candidates: list[dict[str, Any]],
    explanations: dict[str, str],
    bindings: dict[str, str],
    similarity: dict[str, dict[str, Any]],
    diversity: dict[str, Any],
    figures: dict[str, str],
) -> str:
    scored_cands = [c for c in candidates if not _is_undocked(c)]
    undocked_cands = [c for c in candidates if _is_undocked(c)]
    cards = "".join(
        _card_html(c, explanations.get(c["molecule"], ""), bindings.get(c["molecule"], ""),
                   similarity.get(c["molecule"]))
        for c in scored_cands[:TOP_N]
    )
    undocked_html = _undocked_section(undocked_cands)
    fig_html = "".join(
        f"<figure><img src='{fname}' alt='{key}'><figcaption>{html.escape(key)}</figcaption></figure>"
        for key, fname in figures.items()
    )
    div = diversity
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia Bilimsel Rapor – {html.escape(target)}</title>
<style>
:root{{--ink:#171717;--muted:#666;--line:#e2e2dc;--paper:#fff;--bg:#f4f4f1}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,sans-serif;line-height:1.55}}
main{{width:min(1180px,calc(100% - 32px));margin:26px auto 60px}}
header,.section{{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:24px;margin-bottom:18px}}
h1{{font-size:32px;margin:0 0 4px}}h2{{margin:0 0 14px;font-size:22px}}
.summary{{font-size:16px}}.warn{{background:#fff6df;border:1px solid #eed69d;border-radius:12px;padding:12px;margin-top:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
.card{{border:1px solid var(--line);border-radius:16px;padding:16px;background:#fbfbf9}}
.card-head{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}.rank{{background:#171717;color:#fff;border-radius:8px;padding:2px 8px;font-weight:800}}
.mol{{font-weight:800}}.score{{margin-left:auto;color:#333;font-weight:700}}
.props{{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:8px}}
.subscores{{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:8px}}
.subscores .prop{{background:#f0f0ec;border-color:#dcdcd4}}
.prop{{display:flex;justify-content:space-between;background:#fff;border:1px solid #eee;border-radius:8px;padding:5px 9px;font-size:13px}}
table.undocked{{width:100%;border-collapse:collapse;font-size:13px}}
table.undocked th,table.undocked td{{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}}
table.undocked th{{background:#f7f7f4}}table.undocked td.smiles{{font-family:ui-monospace,monospace;font-size:11px;overflow-wrap:anywhere}}
.smiles{{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere;background:#fff;border:1px solid #eee;border-radius:8px;padding:6px}}
.why,.binding,.sim{{font-size:13px;margin:8px 0 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:8px}}.metric{{background:#f7f7f4;border-radius:12px;padding:12px}}.metric b{{display:block;font-size:22px}}
figure{{margin:0}}figure img{{max-width:100%;border:1px solid var(--line);border-radius:12px}}figcaption{{color:var(--muted);font-size:12px;margin-top:4px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
@media print{{body{{background:#fff}}.card,.section,header{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="muted">Remedia hesaplamalı ilaç keşfi — bilimsel rapor</div>
<h1>{html.escape(target)}</h1><p class="muted">Oluşturulma: {html.escape(created_at)}</p></header>

<section class="section"><h2>Yönetici özeti</h2><p class="summary">{html.escape(summary)}</p>
<div class="metrics">
<div class="metric">Aday<b>{len(candidates)}</b></div>
<div class="metric">Benzersiz iskelet<b>{div.get('unique_scaffolds', 0)}</b></div>
<div class="metric">Çeşitlilik skoru<b>{div.get('diversity_score', 0)}</b></div>
<div class="metric">En büyük küme<b>{div.get('largest_cluster', 0)}</b></div>
</div>
<p class="warn"><b>Önemli:</b> {html.escape(DISCLAIMER)} <br>
<b>{html.escape(SCORE_LABEL)}</b> eğitilmiş bir model değil, sabit ağırlıklı geçici bir
sıralama bileşenidir (pose + ADMET + ilaç-benzerliği + çeşitlilik); tek başına kesin
bir değer olarak yorumlanmamalıdır. Bağımsız docking skoru üretilemeyen adaylar
cezalandırılır ve aşağıda ayrı listelenir.</p></section>

<section class="section"><h2>Skorlanan adaylar (en iyi {min(TOP_N, len(scored_cands))} / {len(scored_cands)})</h2>
<div class="grid">{cards}</div></section>

{undocked_html}

<section class="section"><h2>Yayın figürleri</h2>
{('<div class="figs">' + fig_html + '</div>') if fig_html else '<p class="muted">Figürler bu ortamda üretilemedi (matplotlib/RDKit gerekli).</p>'}
</section>

<section class="section"><h2>Çeşitlilik analizi</h2>
<p>{div.get('unique_scaffolds', 0)} benzersiz Murcko iskeleti / {div.get('molecules', 0)} molekül.
En kalabalık iskelet kümesi {div.get('largest_cluster', 0)} molekül içeriyor.
Çeşitlilik skoru (benzersiz/toplam): <b>{div.get('diversity_score', 0)}</b>.</p></section>
</main></body></html>"""


# ======================================================================
# Provenance
# ======================================================================
def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def collect_versions() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("rdkit", "torch", "numpy", "pandas", "matplotlib", "reinvent", "requests")
        },
    }


# ======================================================================
# Orchestration
# ======================================================================
def build_scientific_report(
    result_dir: str | Path,
    *,
    target_uniprot: str,
    requested_molecules: int = 0,
    settings: dict[str, Any] | None = None,
    pipeline_log: str = "",
    job_id: str | None = None,
    known_ligands: list[dict[str, Any]] | None = None,
    pocket_center: Any | None = None,
    generation_manifest: dict[str, Any] | None = None,
    seeds: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the scientist-facing report package.  Never raises on optional
    features; returns paths + summary counts."""
    root = Path(result_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sonuç klasörü bulunamadı: {root}")
    report_dir = root / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    candidates = load_candidates(root)

    # diversity, similarity, narrative
    try:
        import remedia_score

        diversity = remedia_score.diversity_report(candidates)
    except Exception:
        diversity = {"molecules": len(candidates), "unique_scaffolds": 0,
                     "diversity_score": 0.0, "largest_cluster": 0, "scaffold_clusters": {}}

    similarity = similarity_analysis(candidates, known_ligands)
    explanations = {c["molecule"]: ranking_explanation(c) for c in candidates[:TOP_N]}
    bindings = {c["molecule"]: binding_analysis(c, pocket_center) for c in candidates[:TOP_N]}
    summary = executive_summary(candidates, diversity, target_uniprot)

    # figures + pdf (optional)
    figures = build_figures(candidates, diversity, report_dir)
    pdf_path = _write_pdf(candidates, figures, report_dir, target_uniprot, summary)

    # HTML
    report_html = build_html(
        target=target_uniprot, created_at=created_at, summary=summary,
        candidates=candidates, explanations=explanations, bindings=bindings,
        similarity=similarity, diversity=diversity, figures=figures,
    )
    report_path = report_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")

    # candidate_overview.csv (rich)
    overview_fields = ["rank", "molecule", "smiles", "remedia_score", "pose_score",
                       "admet_score", "druglikeness_score", "diversity_score",
                       "affinity_kcal_mol", "pose_confidence", "admet_status",
                       "violations", "mw", "logp", "tpsa", "hbd", "hba", "scaffold"]
    with (report_dir / "candidate_overview.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=overview_fields, extrasaction="ignore")
        writer.writeheader()
        for c in candidates:
            writer.writerow(c)

    # pipeline_log.txt
    (report_dir / "pipeline_log.txt").write_text(pipeline_log or "", encoding="utf-8")

    # run_manifest.json (provenance for reproducibility — Phase 9 consumes this)
    manifest = {
        "report_version": 2,
        "created_at_utc": created_at,
        "job_id": job_id,
        "target_uniprot": target_uniprot,
        "requested_molecules": requested_molecules,
        "generator": (settings or {}).get("generator"),
        "pose_engine": (settings or {}).get("pose_engine"),
        "settings": settings or {},
        "seeds": list(seeds or []),
        "pocket_center": list(pocket_center) if pocket_center else None,
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(1 for c in candidates if c.get("affinity_kcal_mol") is not None),
        "top_candidates": [
            {"rank": c["rank"], "molecule": c["molecule"], "smiles": c["smiles"],
             "remedia_score": c.get("remedia_score"), "affinity_kcal_mol": c.get("affinity_kcal_mol"),
             "explanation": explanations.get(c["molecule"], "")}
            for c in candidates[:TOP_N]
        ],
        "diversity": diversity,
        "generation_manifest": generation_manifest or {},
        "environment": collect_versions(),
        "figures": figures,
        "pdf": bool(pdf_path),
    }
    # Phase 9: full reproducibility record (git SHA, seeds, versions, params).
    try:
        import reproducibility

        manifest["reproducibility"] = reproducibility.capture_run_metadata(
            settings=settings or {},
            seeds=seeds or [],
            gnina_path=(settings or {}).get("gnina_path"),
        )
    except Exception:
        pass
    (report_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # README_FIRST.txt
    _write_readme(report_dir, target_uniprot, summary, candidates, figures, bool(pdf_path))

    return {
        "report_dir": str(report_dir),
        "report_path": str(report_path),
        "pdf_path": pdf_path,
        "candidate_count": len(candidates),
        "scored_candidate_count": manifest["scored_candidate_count"],
        "figures": figures,
        "manifest_path": str(report_dir / "run_manifest.json"),
    }


def _write_readme(report_dir: Path, target: str, summary: str,
                  candidates: list[dict[str, Any]], figures: dict[str, str], has_pdf: bool) -> None:
    best = candidates[0] if candidates else None
    lines = [
        "REMEDIA SONUÇ PAKETİ — ÖNCE BUNU OKU",
        "=" * 44,
        "",
        f"Hedef (UniProt): {target}",
        f"Aday sayısı: {len(candidates)}",
        "",
        "YÖNETİCİ ÖZETİ",
        "-" * 14,
        summary,
        "",
        "BU PAKETTE NE VAR?",
        "-" * 18,
        "  report.html            İnsan-dostu bilimsel rapor (tarayıcıda aç).",
        "  candidate_overview.csv  Tüm adaylar, skorlar ve özellikler tablosu.",
        "  remedia_ranking.csv     Geçici Heuristik Skor (v0) sıralaması (üst klasörde).",
        "  run_manifest.json       Tekrarlanabilirlik: parametreler, tohumlar, sürümler.",
        "  pipeline_log.txt        Çalışmanın tam kaydı (hiçbir hata gizlenmez).",
    ]
    if has_pdf:
        lines.append("  report.pdf              Rapor PDF (yayın/paylaşım için).")
    for _key, fname in figures.items():
        lines.append(f"  {fname:<23} Yayın figürü.")
    lines += [
        "",
        "METRİKLER NASIL YORUMLANIR?",
        "-" * 27,
        "  Heuristik Skor (v0): 0-1 arası SABİT AĞIRLIKLI geçici sıralama bileşeni",
        "                  (eğitilmiş model DEĞİL; kesinlik ifade etmez). Bileşenler:",
        "                  bağlanma pozu, ADMET, ilaç-benzerliği, çeşitlilik. Bağımsız",
        "                  docking skoru üretemeyen adaylar cezalandırılır ve ayrı listelenir.",
        "  GNINA kcal/mol: Daha negatif = daha güçlü öngörülen bağlanma (bağımsız kontrol).",
        "  Pose conf.    : DiffDock güven skoru (varsa; yüksek = daha güvenilir poz).",
        "  ADMET         : Basit Lipinski/Veber ön filtresi.",
        "  Çeşitlilik    : Benzersiz kimyasal iskelet oranı.",
    ]
    if best:
        lines += ["", "EN İYİ ADAY", "-" * 11,
                  f"  {best['molecule']} — {SCORE_LABEL} {_fmt(best.get('remedia_score'), 2)}, "
                  f"GNINA {_fmt(best.get('affinity_kcal_mol'), 2)} kcal/mol",
                  f"  SMILES: {best.get('smiles')}"]
    lines += [
        "",
        "BİLİMSEL UYARI",
        "-" * 14,
        "Bu çıktı bir araştırma ön elemesidir. Docking/pose skorları ve ADMET",
        "filtreleri deneysel bağlanma, etkinlik veya güvenlilik kanıtı DEĞİLDİR.",
        "Adaylar deneysel yapı, kontroller ve laboratuvar doğrulaması ile",
        "değerlendirilmelidir.",
        "",
    ]
    (report_dir / "README_FIRST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
