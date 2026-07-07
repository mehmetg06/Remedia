# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
docking.py
Docking artık GNINA ile GPU'da (Google Colab, ücretsiz T4) yapılıyor.
AutoDock Vina TAMAMEN kaldırıldı — bu script yerel docking ÇALIŞTIRMAZ.

Yeni akış:
    1. UI'da (app.py) veya CLI'da molekülleri üret  → data/generated.smi
    2. notebooks/gnina_colab.ipynb dosyasını Colab'da GPU ile çalıştır
    3. Colab senin için `docking_scores.csv` üretip indirir
    4. O dosyayı results/<run_id>/docking_scores.csv olarak yükle
    5. Pipeline (ADMET → sıralama → dashboard) aynı dosyayı okuyarak devam eder

Bu script'in TEK görevi: Colab'dan gelen `docking_scores.csv`'nin, pipeline'ın
geri kalanının (admet_filter.py, rank_report.py) doğrudan okuyabileceği DOĞRU
formatta olduğunu doğrulamak.

Beklenen format — tam olarak bu başlık, BAŞKA HİÇBİR SÜTUN YOK:

    ligand,affinity_kcal_mol
    mol_1,-8.4
    mol_2,-7.1
    ...

Kullanım:
    python docking.py --validate-only results/<run_id>/docking_scores.csv
"""
import argparse
import csv
import sys
from pathlib import Path

# Pipeline'ın geri kalanının beklediği tam başlık.
REQUIRED_HEADER = ["ligand", "affinity_kcal_mol"]

# GNINA docking'i Colab'da nasıl çalıştıracağını anlatan yardım metni.
COLAB_HELP = (
    "GNINA docking'i Google Colab'da (GPU) çalıştır:\n"
    "  1. notebooks/gnina_colab.ipynb dosyasını Colab'da aç:\n"
    "     https://colab.research.google.com/github/mehmetg06/Remedia/blob/"
    "main/notebooks/gnina_colab.ipynb\n"
    "  2. Runtime > Change runtime type > GPU (T4) seç.\n"
    "  3. Tüm hücreleri sırayla çalıştır (Shift+Enter).\n"
    "  4. İnen 'docking_scores.csv' dosyasını bu klasöre yükle (üzerine yaz).\n"
    "  5. Pipeline'ı tekrar çalıştır — dosya bulununca ADMET → sıralama → "
    "dashboard otomatik devam eder."
)


def validate_csv_format(csv_path) -> tuple[bool, str]:
    """Colab'dan gelen docking_scores.csv'nin formatını doğrular.

    Doğru format:
        - İlk satır tam olarak `ligand,affinity_kcal_mol` (başka sütun yok).
        - En az bir veri satırı olmalı.
        - Her satır 2 sütun: boş olmayan ligand ismi + sayısal (veya boş) affinity.
          Boş affinity, o molekülün docking'i başarısız olduğu anlamına gelir ve
          kabul edilir (rank_report.py boş skorları zaten tolere eder).

    (ok, mesaj) döndürür. ok=True ise mesaj bir özet, False ise net hata sebebidir.
    """
    path = Path(csv_path)
    if not path.exists():
        return False, (
            f"❌ Dosya bulunamadı: {path}\n\n"
            "Docking henüz yapılmamış görünüyor.\n\n" + COLAB_HELP
        )

    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:  # noqa: BLE001
        return False, f"❌ CSV okunamadı: {e}"

    if not rows:
        return False, "❌ Dosya tamamen boş."

    header = [h.strip() for h in rows[0]]
    if header != REQUIRED_HEADER:
        return False, (
            "❌ Başlık (sütun isimleri) hatalı.\n"
            f"   Beklenen : {','.join(REQUIRED_HEADER)}\n"
            f"   Bulunan  : {','.join(header) if header else '(boş)'}\n\n"
            "Colab notebook'unun ADIM 5'i tam olarak bu başlığı, ekstra sütun "
            "OLMADAN yazar. Dosyayı elle düzenlediysen fazladan sütunları sil."
        )

    data = [r for r in rows[1:] if r and any(c.strip() for c in r)]
    if not data:
        return False, "❌ Başlık var ama hiç veri satırı yok (hiç molekül skorlanmamış)."

    valid_scores = 0
    for i, row in enumerate(data, start=2):
        if len(row) != 2:
            return False, (
                f"❌ {i}. satırda 2 yerine {len(row)} sütun var: {row}\n"
                "Sadece `ligand,affinity_kcal_mol` olmalı — fazladan sütun ekleme."
            )
        ligand, affinity = row[0].strip(), row[1].strip()
        if not ligand:
            return False, f"❌ {i}. satırda ligand ismi boş."
        if affinity == "":
            continue  # docking başarısız olan molekül — boş skor kabul edilir
        try:
            float(affinity)
        except ValueError:
            return False, (
                f"❌ {i}. satırda affinity sayı değil: '{affinity}'\n"
                "affinity_kcal_mol sütunu kcal/mol cinsinden bir sayı olmalı "
                "(ör. -8.4). Daha negatif = daha güçlü bağlanma."
            )
        valid_scores += 1

    return True, (
        f"✅ Format doğru: {len(data)} ligand bulundu, {valid_scores} tanesi "
        "geçerli sayısal skora sahip. Pipeline (ADMET → sıralama → dashboard) "
        "bu dosyayı doğrudan okuyabilir."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Colab GNINA çıktısı docking_scores.csv format doğrulayıcı. "
            "Vina docking'i KALDIRILDI — docking artık Colab'da GNINA ile yapılır."
        )
    )
    parser.add_argument(
        "--validate-only",
        dest="validate_path",
        metavar="CSV",
        help="Doğrulanacak docking_scores.csv dosyasının yolu",
    )
    args = parser.parse_args()

    if not args.validate_path:
        parser.error(
            "Yerel Vina docking'i kaldırıldı. Docking'i Colab'da GNINA ile yap.\n\n"
            + COLAB_HELP
            + "\n\nBu script yalnızca sonucu doğrular:\n"
            "    python src/docking.py --validate-only results/<run_id>/docking_scores.csv"
        )

    ok, message = validate_csv_format(args.validate_path)
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
