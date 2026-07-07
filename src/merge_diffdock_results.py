# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
merge_diffdock_results.py
Colab'da (T4 GPU) çalıştırılan DiffDock'un ürettiği `diffdock_results.csv`
dosyasını, Codespaces'teki mevcut `validated_candidates.csv` (AutoDock Vina
skorları) ile birleştirir ve iki yöntemi yan yana koyan bir "genel güven"
tablosu üretir.

Mantık — neden iki yöntem?
    Vina fiziksel bir skorlama fonksiyonu (kcal/mol) kullanır; DiffDock ise
    derin öğrenmeyle pozu tahmin edip bir "güven skoru" verir. İki BAĞIMSIZ
    yöntem de bir molekülü güçlü buluyorsa, o moleküle daha çok güvenebiliriz.
    Sadece biri güçlüyse temkinli oluruz; ikisi de zayıfsa elenir.

Kullanım (Codespaces terminali):
    python src/merge_diffdock_results.py

    # veya dosya yollarını elle vererek:
    python src/merge_diffdock_results.py \\
        --diffdock results/diffdock_results.csv \\
        --validated results/validated_candidates.csv \\
        --output results/final_comparison.csv

Çıktı (final_comparison.csv):
    ligand, vina_affinity, diffdock_confidence, genel_guven_durumu
"""

import argparse
import csv
import sys
from pathlib import Path

# ============================================================================
# EŞİK DEĞERLERİ — "güçlü" sayılmanın sınırları
# ----------------------------------------------------------------------------
# Vina: affinity kcal/mol cinsinden; DAHA NEGATİF = DAHA GÜÇLÜ bağlanma.
#       İlaç benzeri bir molekül için ≤ -7.0 kcal/mol yaygın olarak "iyi" kabul
#       edilir.
# DiffDock: güven skoru; DAHA BÜYÜK = DAHA GÜVENİLİR poz. DiffDock makalesinde
#       confidence > 0 tipik olarak "yüksek güven" eşiği olarak kullanılır.
# ============================================================================
VINA_STRONG_THRESHOLD = -7.0      # vina_affinity <= bu değer ise "güçlü"
DIFFDOCK_STRONG_THRESHOLD = 0.0   # diffdock_confidence >= bu değer ise "güçlü"

# Genel güven etiketleri
DURUM_GUCLU = "GÜÇLÜ ADAY"
DURUM_TEK = "TEK YÖNTEMLE DESTEKLENİYOR"
DURUM_ZAYIF = "ZAYIF ADAY"

FIELDNAMES = ["ligand", "vina_affinity", "diffdock_confidence", "genel_guven_durumu"]


def _to_float(value) -> float | None:
    """Bir hücreyi float'a çevirir; boş/geçersizse None döndürür (asla patlamaz)."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ============================================================================
# CSV OKUMA
# ============================================================================
def load_validated(path: Path) -> dict[str, float | None]:
    """
    validated_candidates.csv'yi okur ve {ligand: vina_affinity} sözlüğü döndürür.

    vina_affinity olarak önce `dogrulanmis_skor` (yüksek exhaustiveness ile
    doğrulanmış skor) kullanılır; yoksa `ilk_skor`a düşülür.
    Dosya yoksa boş sözlük döner (çağıran taraf uyarır).
    """
    result: dict[str, float | None] = {}
    if not path.exists():
        return result
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("ligand", "")).strip()
            if not name or name == "(yok)":
                continue
            vina = _to_float(row.get("dogrulanmis_skor"))
            if vina is None:
                vina = _to_float(row.get("ilk_skor"))
            result[name] = vina
    return result


def load_diffdock(path: Path) -> dict[str, float | None]:
    """
    diffdock_results.csv'yi okur ve {ligand: diffdock_confidence} döndürür.
    Beklenen sütunlar: ligand, diffdock_confidence, diffdock_pose_path.
    """
    result: dict[str, float | None] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("ligand", "")).strip()
            if not name:
                continue
            result[name] = _to_float(row.get("diffdock_confidence"))
    return result


# ============================================================================
# GENEL GÜVEN MANTIĞI
# ============================================================================
def genel_guven(vina: float | None, diffdock: float | None) -> str:
    """
    İki yöntemin güçlü olup olmadığına bakarak genel güven etiketi üretir.

        her ikisi de güçlü  → GÜÇLÜ ADAY
        yalnızca biri güçlü → TEK YÖNTEMLE DESTEKLENİYOR
        ikisi de zayıf/yok  → ZAYIF ADAY
    """
    vina_guclu = vina is not None and vina <= VINA_STRONG_THRESHOLD
    diffdock_guclu = diffdock is not None and diffdock >= DIFFDOCK_STRONG_THRESHOLD

    if vina_guclu and diffdock_guclu:
        return DURUM_GUCLU
    if vina_guclu or diffdock_guclu:
        return DURUM_TEK
    return DURUM_ZAYIF


# ============================================================================
# ANA BİRLEŞTİRME
# ============================================================================
def merge(
    diffdock_csv: Path,
    validated_csv: Path,
    output_csv: Path,
) -> list[dict]:
    """
    DiffDock ve Vina sonuçlarını birleştirir, tabloyu yazar ve satırları döndürür.
    """
    diffdock_scores = load_diffdock(diffdock_csv)
    vina_scores = load_validated(validated_csv)

    if not vina_scores:
        print(
            f"[UYARI] {validated_csv} bulunamadı ya da boş — vina_affinity sütunu "
            f"boş kalacak. Sadece DiffDock skorları raporlanacak."
        )

    # İki kaynaktaki tüm ligandların birleşimi (DiffDock önce gelsin ki sıra
    # test edilen moleküllere göre olsun).
    all_ligands = list(diffdock_scores.keys())
    for name in vina_scores:
        if name not in diffdock_scores:
            all_ligands.append(name)

    rows = []
    for name in all_ligands:
        vina = vina_scores.get(name)
        diffdock = diffdock_scores.get(name)
        rows.append({
            "ligand": name,
            "vina_affinity": "" if vina is None else round(vina, 4),
            "diffdock_confidence": "" if diffdock is None else round(diffdock, 4),
            "genel_guven_durumu": genel_guven(vina, diffdock),
        })

    # En güvenilirden en zayıfa sırala: GÜÇLÜ → TEK → ZAYIF
    _oncelik = {DURUM_GUCLU: 0, DURUM_TEK: 1, DURUM_ZAYIF: 2}
    rows.sort(key=lambda r: _oncelik.get(r["genel_guven_durumu"], 3))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _print_summary(rows: list[dict], output_csv: Path) -> None:
    """Kullanıcının bir bakışta anlayacağı NET bir özet yazdırır."""
    guclu = sum(1 for r in rows if r["genel_guven_durumu"] == DURUM_GUCLU)
    tek = sum(1 for r in rows if r["genel_guven_durumu"] == DURUM_TEK)
    zayif = sum(1 for r in rows if r["genel_guven_durumu"] == DURUM_ZAYIF)

    print(f"[OK] {len(rows)} molekül birleştirildi.")
    print(f"Güçlü aday: {guclu} molekül")
    print(f"Tek yöntemle desteklenen: {tek} molekül")
    print(f"Zayıf: {zayif} molekül")
    print(f"Detaylı tablo: {output_csv}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Colab DiffDock sonuçlarını Vina (validated_candidates.csv) "
                    "ile birleştirir ve genel güven tablosu üretir."
    )
    parser.add_argument(
        "--diffdock", default="results/diffdock_results.csv",
        help="Colab'dan indirilen DiffDock sonuç CSV'si "
             "(varsayılan: results/diffdock_results.csv)"
    )
    parser.add_argument(
        "--validated", default="results/validated_candidates.csv",
        help="Vina doğrulama sonuçları (varsayılan: results/validated_candidates.csv)"
    )
    parser.add_argument(
        "--output", default="results/final_comparison.csv",
        help="Birleştirilmiş çıktı tablosu (varsayılan: results/final_comparison.csv)"
    )
    args = parser.parse_args()

    diffdock_csv = Path(args.diffdock)

    # Dosya yoksa KIRILMA — sade, yol gösterici bir hata ver.
    if not diffdock_csv.exists():
        print(
            f"[HATA] diffdock_results.csv bulunamadı: {diffdock_csv}\n"
            f"       Önce Colab'dan indirdiğin dosyayı results/ klasörüne yükle.\n"
            f"       (notebooks/diffdock_colab.ipynb son hücresi bu dosyayı indirir.)"
        )
        sys.exit(1)

    rows = merge(
        diffdock_csv=diffdock_csv,
        validated_csv=Path(args.validated),
        output_csv=Path(args.output),
    )
    _print_summary(rows, Path(args.output))


if __name__ == "__main__":
    main()
