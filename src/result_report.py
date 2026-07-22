"""Create a human-readable Remedia result package from a pipeline run directory.

The pipeline is allowed to evolve without keeping this module in lockstep.  The
report builder therefore discovers CSV/JSON/SDF files dynamically, chooses the
most informative candidate table, and always emits a useful manifest even when
some expected columns are absent.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


REPORT_DIR_NAME = "00_REMEDIA_REPORT"

COLUMN_EXPLANATIONS = {
    "rank": "Adayın nihai sırası; 1 en yüksek önceliktir.",
    "ligand": "Molekülün işlem içindeki adı veya kimliği.",
    "molecule": "Molekülün işlem içindeki adı veya kimliği.",
    "name": "Molekül adı veya kimliği.",
    "smiles": "Molekülün SMILES kimyasal gösterimi.",
    "affinity_kcal_mol": "Seçilen nihai GNINA bağlanma skoru (kcal/mol). Daha negatif değer genellikle daha güçlü öngörülen bağlanmayı gösterir.",
    "fast_affinity_kcal_mol": "GNINA hızlı tarama aşamasındaki bağlanma skoru (kcal/mol).",
    "accurate_affinity_kcal_mol": "GNINA ayrıntılı doğrulama aşamasındaki bağlanma skoru (kcal/mol).",
    "skor_kaynagi": "Nihai skorun hızlı mı yoksa ayrıntılı GNINA aşamasından mı geldiği.",
    "docking_success": "GNINA'nın bu aday için kullanılabilir bir skor üretip üretmediği.",
    "docking_error": "Docking başarısızsa teknik neden.",
    "admet_status": "Basit ADMET/ilaç-benzerliği filtresinin genel sonucu.",
    "status": "Adayın filtre veya işlem durumu.",
    "violations": "Adayın geçtiği sınırları ihlal eden özellikler.",
    "mw": "Molekül ağırlığı (g/mol).",
    "molecular_weight": "Molekül ağırlığı (g/mol).",
    "logp": "Tahmini lipofilite. Çok yüksek değerler çözünürlük ve geliştirme riski doğurabilir.",
    "hbd": "Hidrojen bağı verici sayısı.",
    "hba": "Hidrojen bağı alıcı sayısı.",
    "tpsa": "Topolojik polar yüzey alanı (Å²).",
    "rotatable_bonds": "Dönebilir bağ sayısı; moleküler esnekliğin kaba göstergesidir.",
    "fast_seconds": "Hızlı docking aşamasında molekül başına yaklaşık süre.",
    "accurate_seconds": "Ayrıntılı docking aşamasında molekül başına yaklaşık süre.",
}

NAME_KEYS = ("ligand", "molecule", "name", "id", "compound", "candidate")
SMILES_KEYS = ("smiles", "SMILES", "canonical_smiles", "canonical")
AFFINITY_KEYS = (
    "affinity_kcal_mol",
    "accurate_affinity_kcal_mol",
    "fast_affinity_kcal_mol",
    "minimizedAffinity",
    "CNNaffinity",
    "affinity",
)
STATUS_KEYS = ("admet_status", "status", "filter_status", "pass", "passed")


def _clean_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    low = str(value).strip().lower()
    if low in {"true", "1", "yes", "evet", "pass", "passed", "başarılı", "basarili"}:
        return True
    if low in {"false", "0", "no", "hayır", "hayir", "fail", "failed", "başarısız", "basarisiz"}:
        return False
    return None


def _first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    lowered = {_clean_key(str(k)): v for k, v in row.items()}
    for key in keys:
        if _clean_key(key) in lowered and lowered[_clean_key(key)] not in (None, ""):
            return lowered[_clean_key(key)]
    return None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                return list(reader.fieldnames or []), rows
        except UnicodeDecodeError:
            continue
        except Exception:
            return [], []
    return [], []


def _candidate_table_score(columns: list[str], row_count: int, path: Path) -> float:
    names = {_clean_key(c) for c in columns}
    score = min(row_count, 1000) / 100.0
    if names.intersection({_clean_key(k) for k in NAME_KEYS}):
        score += 10
    if names.intersection({_clean_key(k) for k in SMILES_KEYS}):
        score += 18
    if names.intersection({_clean_key(k) for k in AFFINITY_KEYS}):
        score += 30
    if names.intersection({_clean_key(k) for k in STATUS_KEYS}):
        score += 8
    low_path = path.as_posix().lower()
    if any(token in low_path for token in ("final", "rank", "result", "candidate", "admet", "dock")):
        score += 12
    if "benchmark" in low_path:
        score -= 10
    return score


def _describe_file(path: Path) -> str:
    low = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if "dock" in low:
            return "GNINA docking skorları ve aşama bilgileri"
        if "admet" in low:
            return "ADMET/ilaç-benzerliği hesapları"
        if "rank" in low or "final" in low:
            return "Nihai sıralanmış aday tablosu"
        return "Tablosal sonuç verisi"
    if suffix in {".sdf", ".mol", ".mol2"}:
        return "3B molekül yapıları ve/veya docking pozları"
    if suffix in {".pdb", ".pdbqt", ".cif", ".mmcif"}:
        return "Reseptör yapısı veya hazırlanmış docking girdisi"
    if suffix == ".smi":
        return "SMILES molekül listesi"
    if suffix == ".json":
        return "Makinece okunabilir ayar veya özet"
    if suffix in {".log", ".txt"}:
        return "Metin kaydı veya açıklama"
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "Görsel çıktı"
    return "Pipeline tarafından üretilen yardımcı dosya"


def _file_inventory(root: Path, report_dir: Path | None = None) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if report_dir and (path == report_dir or report_dir in path.parents):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "description": _describe_file(path),
            }
        )
    return inventory


def _discover_tables(root: Path, report_dir: Path) -> list[dict[str, Any]]:
    tables = []
    for path in sorted(root.rglob("*.csv")):
        if report_dir in path.parents:
            continue
        columns, rows = _read_csv(path)
        tables.append(
            {
                "path": path,
                "relative_path": path.relative_to(root).as_posix(),
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "score": _candidate_table_score(columns, len(rows), path),
            }
        )
    return tables


def _normalise_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for index, row in enumerate(rows, 1):
        name = _first(row, NAME_KEYS) or f"mol_{index:03d}"
        smiles = _first(row, SMILES_KEYS)
        affinity = _safe_float(_first(row, AFFINITY_KEYS))
        fast = _safe_float(_first(row, ("fast_affinity_kcal_mol",)))
        accurate = _safe_float(_first(row, ("accurate_affinity_kcal_mol",)))
        docking_success = _truthy(_first(row, ("docking_success",)))
        status = _first(row, STATUS_KEYS)
        candidates.append(
            {
                "molecule": str(name),
                "smiles": str(smiles) if smiles not in (None, "") else "",
                "affinity_kcal_mol": affinity,
                "fast_affinity_kcal_mol": fast,
                "accurate_affinity_kcal_mol": accurate,
                "score_source": _first(row, ("skor_kaynagi", "score_source")) or "",
                "docking_success": docking_success,
                "docking_error": _first(row, ("docking_error", "error")) or "",
                "admet_status": status or "",
                "violations": _first(row, ("violations", "ihlal", "fails", "reasons")) or "",
                "mw": _safe_float(_first(row, ("mw", "molecular_weight", "molwt"))),
                "logp": _safe_float(_first(row, ("logp", "log_p", "mol_logp"))),
                "hbd": _safe_float(_first(row, ("hbd", "num_h_donors"))),
                "hba": _safe_float(_first(row, ("hba", "num_h_acceptors"))),
                "tpsa": _safe_float(_first(row, ("tpsa", "psa"))),
                "original_row": row,
            }
        )

    def sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
        affinity = item["affinity_kcal_mol"]
        return (affinity is None, affinity if affinity is not None else float("inf"), item["molecule"])

    candidates.sort(key=sort_key)
    for index, item in enumerate(candidates, 1):
        item["rank"] = index
    return candidates


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _status_label(candidate: dict[str, Any]) -> str:
    raw = str(candidate.get("admet_status") or "").strip()
    if raw:
        return raw
    if candidate.get("docking_success") is False:
        return "Docking başarısız"
    if candidate.get("affinity_kcal_mol") is not None:
        return "Skorlandı"
    return "Belirsiz"


def _render_top_molecules(candidates: list[dict[str, Any]], output: Path) -> bool:
    usable = [c for c in candidates if c.get("smiles")][:12]
    if not usable:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mols, legends = [], []
        for candidate in usable:
            mol = Chem.MolFromSmiles(candidate["smiles"])
            if mol is None:
                continue
            mols.append(mol)
            score = _display(candidate.get("affinity_kcal_mol"))
            legends.append(f"#{candidate['rank']} {candidate['molecule']}\nGNINA: {score} kcal/mol")
        if not mols:
            return False
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(360, 280),
            legends=legends,
            useSVG=False,
        )
        image.save(output)
        return True
    except Exception:
        return False


def _html_table(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for c in candidates[:50]:
        status = _status_label(c)
        rows.append(
            "<tr>"
            f"<td>{c['rank']}</td>"
            f"<td><strong>{html.escape(c['molecule'])}</strong></td>"
            f"<td class='num'>{html.escape(_display(c.get('affinity_kcal_mol')))}</td>"
            f"<td>{html.escape(str(c.get('score_source') or '—'))}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(c.get('violations') or '—'))}</td>"
            f"<td class='smiles'>{html.escape(str(c.get('smiles') or '—'))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p class='warning'>Aday tablosu bulunamadı. Aşağıdaki dosya envanterinden ham çıktıları kontrol et.</p>"
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Sıra</th><th>Molekül</th><th>GNINA kcal/mol</th><th>Skor kaynağı</th>"
        "<th>Durum</th><th>İhlaller</th><th>SMILES</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _build_html(
    *,
    target_uniprot: str,
    requested_molecules: int,
    candidates: list[dict[str, Any]],
    primary_table: str | None,
    inventory: list[dict[str, Any]],
    created_at: str,
    has_image: bool,
) -> str:
    scored = [c for c in candidates if c.get("affinity_kcal_mol") is not None]
    passed = [c for c in candidates if str(c.get("admet_status", "")).strip().lower() in {"pass", "passed", "geçti", "gecti", "ok"}]
    best = scored[0] if scored else None
    best_text = (
        f"{html.escape(best['molecule'])} ({best['affinity_kcal_mol']:.3f} kcal/mol)"
        if best else "Kullanılabilir docking skoru yok"
    )
    inventory_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['path'])}</code></td>"
        f"<td>{html.escape(item['description'])}</td>"
        f"<td class='num'>{item['size_bytes'] if item['size_bytes'] is not None else '—'}</td>"
        "</tr>"
        for item in inventory
    )
    image_html = "<img class='molecules' src='top_molecules.png' alt='En yüksek sıralı moleküller'>" if has_image else ""
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia Sonuç Raporu – {html.escape(target_uniprot)}</title>
<style>
:root{{--ink:#171717;--muted:#666;--line:#deded8;--paper:#fff;--bg:#f4f4f1;--soft:#f8f8f5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,sans-serif;line-height:1.5}}
main{{width:min(1180px,calc(100% - 32px));margin:28px auto 60px}}header,.section{{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:26px;margin-bottom:18px}}
h1{{font-size:34px;margin:0 0 4px}}h2{{margin:0 0 16px;font-size:22px}}h3{{margin:20px 0 8px}}p{{margin:8px 0}}.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:22px}}.metric{{background:var(--soft);border-radius:14px;padding:16px}}.metric b{{display:block;font-size:23px;margin-top:5px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 12px;border-bottom:1px solid #ecece7;text-align:left;vertical-align:top}}th{{background:#f5f5f2;position:sticky;top:0}}.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}.smiles{{max-width:420px;overflow-wrap:anywhere;font-family:ui-monospace,monospace;font-size:12px}}
.warning{{background:#fff6df;border:1px solid #eed69d;border-radius:12px;padding:14px}}.note{{background:#eef5ff;border:1px solid #c8d8ef;border-radius:12px;padding:14px}}code{{overflow-wrap:anywhere}}.molecules{{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:14px}}
ul{{padding-left:20px}}
@media print{{body{{background:#fff}}main{{width:100%;margin:0}}header,.section{{break-inside:avoid;border-color:#bbb}}}}
</style></head><body><main>
<header><div class="muted">Remedia hesaplamalı ilaç keşfi raporu</div><h1>{html.escape(target_uniprot)}</h1><p>Oluşturulma: {html.escape(created_at)}</p>
<div class="cards">
<div class="metric">İstenen molekül<b>{requested_molecules}</b></div>
<div class="metric">Tabloda bulunan aday<b>{len(candidates)}</b></div>
<div class="metric">Docking skoru bulunan<b>{len(scored)}</b></div>
<div class="metric">ADMET “pass” görünen<b>{len(passed)}</b></div>
</div></header>
<section class="section"><h2>Yönetici özeti</h2><p><strong>En iyi öngörülen aday:</strong> {best_text}</p>
<p><strong>Ana veri tablosu:</strong> <code>{html.escape(primary_table or 'Bulunamadı')}</code></p>
<p class="warning"><strong>Önemli:</strong> Bu sonuçlar deneysel doğrulama değildir. GNINA skoru bağlanma eğilimi için hesaplamalı bir ön elemedir; toksisite, etkinlik, seçicilik ve klinik uygunluk sonucu olarak yorumlanmamalıdır. Bu sonuçlar hesaplamalı tahmindir; deneysel aktivite, toksisite veya klinik uygunluk kanıtı değildir.</p></section>
<section class="section"><h2>En yüksek sıralı adaylar</h2>{image_html}{_html_table(candidates)}</section>
<section class="section"><h2>Nasıl yorumlanmalı?</h2>
<ul><li><strong>GNINA affinity:</strong> Daha negatif skor genellikle daha iyi öngörülen bağlanmayı ifade eder; farklı hedefler arasında doğrudan kıyas yapılmamalıdır.</li>
<li><strong>FAST → ACCURATE:</strong> Tüm adaylar hızlı taranır; en iyi bölüm daha ayrıntılı ayarla yeniden dock edilir.</li>
<li><strong>ADMET filtresi:</strong> Basit fizikokimyasal kurallardır. “Pass” güvenli veya etkili ilaç anlamına gelmez.</li>
<li><strong>Önceliklendirme:</strong> İyi docking, kabul edilebilir özellikler ve kimyasal çeşitlilik birlikte değerlendirilmelidir.</li></ul></section>
<section class="section"><h2>Çıktı dosyaları</h2><p class="note"><code>candidate_overview.csv</code> hızlı inceleme içindir; ham pipeline dosyaları değiştirilmeden ZIP içinde tutulmuştur.</p>
<div class="table-wrap"><table><thead><tr><th>Dosya</th><th>İçerik</th><th>Bayt</th></tr></thead><tbody>{inventory_rows}</tbody></table></div></section>
</main></body></html>"""


def build_result_package(
    result_dir: str | Path,
    *,
    target_uniprot: str,
    requested_molecules: int,
    settings: dict[str, Any] | None = None,
    pipeline_log: str = "",
    job_id: str | None = None,
) -> dict[str, Any]:
    """Add explanatory, analysis-ready files to a completed pipeline run.

    Existing files are never deleted or overwritten outside ``00_REMEDIA_REPORT``.
    The returned paths are absolute so the web layer can expose the HTML report.
    """
    root = Path(result_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sonuç klasörü bulunamadı: {root}")

    report_dir = root / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = _discover_tables(root, report_dir)
    primary = max(tables, key=lambda item: item["score"], default=None)
    candidates = _normalise_candidates(primary["rows"]) if primary else []
    inventory = _file_inventory(root, report_dir)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    overview_fields = [
        "rank", "molecule", "smiles", "affinity_kcal_mol",
        "fast_affinity_kcal_mol", "accurate_affinity_kcal_mol",
        "score_source", "docking_success", "docking_error",
        "admet_status", "violations", "mw", "logp", "hbd", "hba", "tpsa",
    ]
    overview_rows = [{key: candidate.get(key) for key in overview_fields} for candidate in candidates]
    _write_csv(report_dir / "candidate_overview.csv", overview_rows, overview_fields)

    dictionary_rows = [
        {"column": column, "explanation_tr": explanation}
        for column, explanation in sorted(COLUMN_EXPLANATIONS.items())
    ]
    _write_csv(report_dir / "data_dictionary.csv", dictionary_rows, ["column", "explanation_tr"])

    table_summaries = [
        {
            "path": item["relative_path"],
            "row_count": item["row_count"],
            "columns": item["columns"],
            "candidate_table_score": round(item["score"], 3),
        }
        for item in tables
    ]
    manifest = {
        "report_version": 1,
        "created_at_utc": created_at,
        "job_id": job_id,
        "target_uniprot": target_uniprot,
        "requested_molecules": requested_molecules,
        "settings": settings or {},
        "primary_candidate_table": primary["relative_path"] if primary else None,
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(c["affinity_kcal_mol"] is not None for c in candidates),
        "tables": table_summaries,
        "files": inventory,
        "scientific_limitations": [
            "REINVENT4 sampling genel bir prior modelinden yapılır; hedefe özel eğitim değildir.",
            "GNINA skorları hesaplamalı tahmindir ve deneysel bağlanma ölçümü değildir.",
            "ADMET sonuçları basit ön filtrelerdir; güvenlilik veya klinik etkinlik kanıtı değildir.",
        ],
    }
    (report_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "pipeline_log.txt").write_text(pipeline_log or "Log kaydı yok.\n", encoding="utf-8")

    readme = f"""REMEDIA SONUÇ PAKETİ
=====================

Hedef UniProt: {target_uniprot}
İstenen molekül sayısı: {requested_molecules}
Bulunan aday satırı: {len(candidates)}
Ana aday tablosu: {primary['relative_path'] if primary else 'bulunamadı'}

ÖNCE BUNLARI AÇ
----------------
1. report.html
   Görsel ve açıklamalı sonuç raporu. Tarayıcıda açılır.
2. candidate_overview.csv
   En önemli alanların tek tabloda, sıralanmış özeti.
3. data_dictionary.csv
   Sütunların Türkçe açıklamaları.
4. run_manifest.json
   Ayarlar, tüm dosyalar ve tablo envanteri.
5. pipeline_log.txt
   Pipeline'ın teknik çalışma kaydı.

BİLİMSEL UYARI
--------------
Bu paket deneysel doğrulama değildir. Docking ve ADMET değerleri adayları önceliklendirmek
için hesaplamalı tahminlerdir. Klinik karar, hasta tedavisi veya güvenlilik iddiası için
kullanılmamalıdır.
"""
    (report_dir / "README_FIRST.txt").write_text(readme, encoding="utf-8")

    has_image = _render_top_molecules(candidates, report_dir / "top_molecules.png")
    report_path = report_dir / "report.html"
    report_path.write_text(
        _build_html(
            target_uniprot=target_uniprot,
            requested_molecules=requested_molecules,
            candidates=candidates,
            primary_table=primary["relative_path"] if primary else None,
            inventory=inventory,
            created_at=created_at,
            has_image=has_image,
        ),
        encoding="utf-8",
    )

    return {
        "report_path": str(report_path),
        "report_dir": str(report_dir),
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(c["affinity_kcal_mol"] is not None for c in candidates),
        "primary_candidate_table": primary["relative_path"] if primary else None,
        "has_molecule_image": has_image,
    }
