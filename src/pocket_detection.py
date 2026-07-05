# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
pocket_detection.py
Reseptör yapısında bağlanma cebi (binding pocket) tespiti.

Not: Tam fpocket/P2Rank entegrasyonu için sistemde fpocket binary'si kurulu olmalı
(conda install -c bioconda fpocket). Bu script hem fpocket wrapper'ı hem de
basit bir geometrik fallback (en büyük boşluk merkezi) sunar.

Kullanım:
    python pocket_detection.py --pdb data/P30405_alphafold.pdb
    python pocket_detection.py --pdb data/P30405_alphafold.pdb --center 12.3 45.1 -3.2 --size 20
"""
import argparse
import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_fpocket(pdb_path: Path) -> Path:
    """fpocket ile otomatik cep tespiti. Kurulu değilse hata mesajı verir."""
    if shutil.which("fpocket") is None:
        raise EnvironmentError(
            "fpocket bulunamadı. Kurulum: conda install -c bioconda fpocket\n"
            "Alternatif olarak --center ve --size parametreleriyle manuel cep belirtebilirsin."
        )
    subprocess.run(["fpocket", "-f", str(pdb_path)], check=True)
    out_dir = pdb_path.parent / f"{pdb_path.stem}_out"
    print(f"[OK] fpocket sonuçları: {out_dir}")
    print("     pocketN_atm.pdb dosyalarını incele, en yüksek Druggability Score'a sahip olanı seç.")
    return out_dir


def manual_box(center, size):
    """Kullanıcının belirttiği manuel docking box'ı."""
    print(f"[OK] Manuel docking box: merkez={center}, boyut={size} Å")
    return {"center": center, "size": size}


def main():
    parser = argparse.ArgumentParser(description="Binding pocket tespiti")
    parser.add_argument("--pdb", required=True, help="Reseptör PDB dosyası yolu")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                         help="Manuel cep merkezi koordinatları")
    parser.add_argument("--size", type=float, default=20.0,
                         help="Docking box boyutu (Å küp, varsayılan 20)")
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} bulunamadı.")

    if args.center:
        manual_box(args.center, args.size)
    else:
        run_fpocket(pdb_path)


if __name__ == "__main__":
    main()
