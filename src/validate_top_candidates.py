# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
validate_top_candidates.py
Düşük exhaustiveness ile bulunan "çok iyi" docking skorlarının artefakt olup
olmadığını tespit etmek için otomatik doğrulama katmanı.

Arka plan:
    Genetik algoritma veya toplu tarama sırasında düşük exhaustiveness (ör. 8)
    kullanıldığında Vina bazen "şanslı" ama gerçekçi olmayan bir konformasyon
    bulur (ör. -11.6 kcal/mol). Aynı molekül exhaustiveness=32 ile yeniden
    docklandığında skor -5.2'ye düşebilir. Bu script bu tutarsızlıkları tespit
    edip kullanıcıyı uyarır.

Kullanım:
    python src/validate_top_candidates.py \\
        --input results/docking_scores.csv \\
        --receptor data/P30405_alphafold.pdbqt \\
        --center 5.00 -1.02 -15.56 --size 20 20 20 \\
        --top-n 5 --exhaustiveness 32 \\
        --output results/validated_candidates.csv

Çıktı (validated_candidates.csv):
    ligand, ilk_skor, dogrulanmis_skor, fark, guven_durumu
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

# --- Proje modüllerini import edilebilir kıl ----------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ============================================================================
# GÜVEN DURUMU MANTIKI
# ============================================================================
GUVEN_GUVENILIR = "GÜVENİLİR"
GUVEN_SUPHE = "ŞÜPHELİ — tekrar kontrol et"
GUVEN_ARTEFAKT = "ARTEFAKT OLASI — güvenme"


def guven_durumu(ilk_skor: float, dogrulanmis_skor: float) -> tuple[str, str]:
    """
    İlk skor ile doğrulama skoru arasındaki farka ve YÖNÜNE göre güven durumu
    belirler.
    
    Returns:
        (guven_durumu, yon)
        yon: 'iyilesti' veya 'kotulesti'
    """
    fark = abs(ilk_skor - dogrulanmis_skor)
    
    if dogrulanmis_skor <= ilk_skor: # Daha negatif olduysa iyileşti (güçlü bağlanma)
        yon = "iyilesti"
        if fark <= 1.0:
            return GUVEN_GUVENILIR, yon
        else:
            return "GÜÇLÜ ADAY — ilk tarama hafife almış", yon
    else: # Skor kötüleşti (daha zayıf bağlanma)
        yon = "kotulesti"
        if fark <= 1.0:
            return GUVEN_GUVENILIR, yon
        elif fark <= 2.5:
            return GUVEN_SUPHE, yon
        else:
            return GUVEN_ARTEFAKT, yon


# ============================================================================
# CSV OKUMA
# ============================================================================
def load_scores(csv_path: Path) -> list[dict]:
    """
    docking_scores.csv (veya validated_candidates.csv) dosyasını okur ve
    {'ligand': str, 'affinity_kcal_mol': float} listesi döndürür.
    Affinity'si None / boş / geçersiz olan satırları atlar.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                aff = float(row["affinity_kcal_mol"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append({"ligand": row["ligand"].strip(), "affinity_kcal_mol": aff})
    return rows


# ============================================================================
# LİGAND KAYNAĞINI BULMA
# ============================================================================
def find_ligand_pdbqt(ligand_name: str, search_dirs: list[Path]) -> Path | None:
    """
    Verilen ligand adını içeren .pdbqt dosyasını arama dizinlerinde arar.
    '_docked' son ekli dosyaları tercih eder, yoksa normal dosyayı döndürür.
    """
    # Önce _docked versiyonunu ara
    for d in search_dirs:
        p = d / f"{ligand_name}_docked.pdbqt"
        if p.exists():
            return p
    # Sonra ham versiyonu ara
    for d in search_dirs:
        p = d / f"{ligand_name}.pdbqt"
        if p.exists():
            return p
    return None


# ============================================================================
# YENİDEN DOCKING (doğrulama)
# ============================================================================
def redock_ligand(
    ligand_pdbqt: Path,
    receptor_pdbqt: Path,
    center: list[float],
    box_size: list[float],
    exhaustiveness: int,
    poses_dir: Path,
) -> float | None:
    """
    Tek bir ligand'ı yüksek exhaustiveness ile yeniden docklar.
    docking.py'yi fonksiyon olarak import eder — CLI bozulmaz.

    Returns:
        En iyi affinity skoru (kcal/mol) ya da hata durumunda None.
    """
    try:
        import docking  # src/docking.py
    except ImportError as e:
        print(f"  [UYARI] docking.py import edilemedi: {e}")
        return None

    # docking.dock_all tek bir ligand için de çalışır:
    # ligand_pdbqt'yi geçici bir dizinde hazır halde bekletiyoruz.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Orijinal ligand dosyasını geçici dizine sembolik bağ ya da kopya olarak koy
        import shutil

        dest = tmp_path / ligand_pdbqt.name
        shutil.copy2(ligand_pdbqt, dest)

        try:
            results = docking.dock_all(
                receptor_pdbqt=receptor_pdbqt,
                ligands_dir=tmp_path,
                center=center,
                box_size=box_size,
                poses_dir=poses_dir,
                exhaustiveness=exhaustiveness,
            )
        except Exception as e:
            print(f"  [UYARI] Vina çalışmadı: {e}")
            return None

    if results:
        return results[0]["affinity_kcal_mol"]
    return None


# ============================================================================
# ANA DOĞRULAMA FONKSİYONU
# ============================================================================
def validate_top_candidates(
    scores_csv: Path,
    receptor_pdbqt: Path,
    center: list[float],
    box_size: list[float],
    top_n: int = 5,
    exhaustiveness: int = 32,
    ligand_dirs: list[Path] | None = None,
    output_csv: Path = Path("results/validated_candidates.csv"),
) -> list[dict]:
    """
    En iyi N molekülü yüksek exhaustiveness ile yeniden docklar ve karşılaştırma
    tablosu oluşturur.

    Args:
        scores_csv:      İlk tarama sonuçlarını içeren CSV (ligand, affinity_kcal_mol).
        receptor_pdbqt:  Reseptör PDBQT dosyası.
        center:          Docking kutusu merkezi [x, y, z].
        box_size:        Docking kutusu boyutu [sx, sy, sz].
        top_n:           Doğrulanacak en iyi N molekül sayısı.
        exhaustiveness:  Yeniden docking için kullanılacak exhaustiveness.
        ligand_dirs:     Hazır .pdbqt dosyalarının aranacağı dizinler.
        output_csv:      Karşılaştırma tablosunun yazılacağı yol.

    Returns:
        Karşılaştırma satırlarının listesi (dict).
    """
    # --- Varsayılan arama dizinleri ---
    if ligand_dirs is None:
        ligand_dirs = [
            Path("data/ligands_prepared"),
            Path("results/docked_poses"),
            Path("results/ga_work/prepared"),
            Path("results/ga_work/poses"),
            Path("results/ui_ga_work/prepared"),
            Path("results/ui_ga_work/poses"),
        ]
    # Varolmayanları filtrele
    ligand_dirs = [d for d in ligand_dirs if d.exists()]

    # --- Skorları yükle ve ilk N tanesini seç ---
    all_scores = load_scores(scores_csv)
    if not all_scores:
        print(f"[HATA] {scores_csv} dosyasında geçerli skor bulunamadı.")
        return []

    # En iyi (en negatif affinity) N molekülü seç
    all_scores.sort(key=lambda r: r["affinity_kcal_mol"])
    top_rows = all_scores[:top_n]

    print(f"\n{'='*60}")
    print(f"DOĞRULAMA KATMANI — Top {top_n} molekül, exhaustiveness={exhaustiveness}")
    print(f"{'='*60}")
    print(f"{'Ligand':<20} {'İlk Skor':>10} {'Doğrulama':>10} {'Fark':>7} {'Durum'}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*7} {'-'*30}")

    # Doğrulama pozları için dizin
    val_poses_dir = output_csv.parent / "validation_poses"
    val_poses_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    for row in top_rows:
        ligand_name = row["ligand"]
        ilk_skor = row["affinity_kcal_mol"]

        # Ligand PDBQT'yi bul
        lig_pdbqt = find_ligand_pdbqt(ligand_name, ligand_dirs)
        if lig_pdbqt is None:
            print(f"  [UYARI] {ligand_name}: PDBQT dosyası bulunamadı — atlanıyor.")
            comparison_rows.append({
                "ligand": ligand_name,
                "ilk_skor": ilk_skor,
                "dogrulanmis_skor": "",
                "fark": "",
                "yon": "",
                "guven_durumu": "DOĞRULANAMADI — PDBQT bulunamadı",
            })
            continue

        print(f"  ↻  {ligand_name}: {ilk_skor:.3f} → yeniden docklama...", end=" ", flush=True)
        dogrulanmis_skor = redock_ligand(
            ligand_pdbqt=lig_pdbqt,
            receptor_pdbqt=receptor_pdbqt,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            poses_dir=val_poses_dir,
        )

        if dogrulanmis_skor is None:
            print("BAŞARISIZ")
            comparison_rows.append({
                "ligand": ligand_name,
                "ilk_skor": ilk_skor,
                "dogrulanmis_skor": "",
                "fark": "",
                "yon": "",
                "guven_durumu": "DOĞRULANAMADI — Vina hatası",
            })
            continue

        fark_val = abs(ilk_skor - dogrulanmis_skor)
        durum, yon = guven_durumu(ilk_skor, dogrulanmis_skor)

        # Konsol çıktısı
        durum_sembol = {"GÜVENİLİR": "✓", "ŞÜPHELİ — tekrar kontrol et": "⚠", "ARTEFAKT OLASI — güvenme": "✗", "GÜÇLÜ ADAY — ilk tarama hafife almış": "⭐"}.get(durum, "?")
        print(f"{dogrulanmis_skor:.3f}  (Δ={fark_val:.2f})  {durum_sembol} {durum}")

        comparison_rows.append({
            "ligand": ligand_name,
            "ilk_skor": round(ilk_skor, 4),
            "dogrulanmis_skor": round(dogrulanmis_skor, 4),
            "fark": round(fark_val, 4),
            "yon": yon,
            "guven_durumu": durum,
        })

    print(f"{'='*60}\n")

    # --- Çıktıyı kaydet ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["ligand", "ilk_skor", "dogrulanmis_skor", "fark", "yon", "guven_durumu"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"[OK] Doğrulama sonuçları kaydedildi: {output_csv}")

    # --- Smina Cross Validation ---
    try:
        import cross_validate_docking as cvd
        smina_output_csv = output_csv.parent / "cross_validated.csv"
        print(f"\n[Smina Çapraz Doğrulama] Smina ile cross validation başlatılıyor...")
        cv_results = cvd.cross_validate(
            smiles_csv=output_csv,
            receptor_pdbqt=receptor_pdbqt,
            center=center,
            box_size=box_size,
            ligand_dirs=ligand_dirs,
            top_n=top_n
        )
        if cv_results:
            with open(smina_output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ligand", "vina_affinity", "smina_affinity", "fark", "tutarlilik_durumu"])
                writer.writeheader()
                writer.writerows(cv_results)
            print(f"[OK] Smina çapraz doğrulama sonuçları kaydedildi: {smina_output_csv}")
    except Exception as e:
        print(f"  [UYARI] Smina çapraz doğrulama çalışmadı: {e}")

    return comparison_rows


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Düşük exhaustiveness'le bulunan skorları yüksek exhaustiveness ile doğrula"
    )
    parser.add_argument(
        "--input", default="results/docking_scores.csv",
        help="İlk docking sonuçlarını içeren CSV dosyası (varsayılan: results/docking_scores.csv)"
    )
    parser.add_argument(
        "--receptor", required=True,
        help="Reseptör PDBQT dosyası"
    )
    parser.add_argument(
        "--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
        help="Docking kutusu merkezi"
    )
    parser.add_argument(
        "--size", nargs=3, type=float, default=[20, 20, 20], metavar=("SX", "SY", "SZ"),
        help="Docking kutusu boyutu (varsayılan: 20 20 20)"
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Doğrulanacak en iyi N molekül (varsayılan: 5)"
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=32,
        help="Doğrulama docking'i için exhaustiveness (varsayılan: 32)"
    )
    parser.add_argument(
        "--ligand-dirs", nargs="*",
        help="Hazır PDBQT dosyalarının aranacağı ek dizinler"
    )
    parser.add_argument(
        "--output", default="results/validated_candidates.csv",
        help="Karşılaştırma tablosunun çıktı CSV yolu (varsayılan: results/validated_candidates.csv)"
    )
    args = parser.parse_args()

    extra_dirs = [Path(d) for d in (args.ligand_dirs or [])]

    validate_top_candidates(
        scores_csv=Path(args.input),
        receptor_pdbqt=Path(args.receptor),
        center=args.center,
        box_size=args.size,
        top_n=args.top_n,
        exhaustiveness=args.exhaustiveness,
        ligand_dirs=extra_dirs if extra_dirs else None,
        output_csv=Path(args.output),
    )


if __name__ == "__main__":
    main()
