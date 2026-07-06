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
import re
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
    out_dir = pdb_path.parent / f"{pdb_path.stem}_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # fpocket zaten var olan çıktı klasörüne yazmayı reddeder
    subprocess.run(["fpocket", "-f", str(pdb_path)], check=True)
    print(f"[OK] fpocket sonuçları: {out_dir}")
    print("     pocketN_atm.pdb dosyalarını incele, en yüksek Druggability Score'a sahip olanı seç.")
    return out_dir


def parse_fpocket_info(info_path: Path) -> list[dict]:
    """fpocket'in ürettiği {stem}_info.txt dosyasını ayrıştırır, her pocket için skorları döndürür."""
    pockets: list[dict] = []
    current: dict | None = None
    for raw_line in info_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"Pocket\s+(\d+)\s*:", line)
        if m:
            if current is not None:
                pockets.append(current)
            current = {"pocket": int(m.group(1))}
            continue
        if current is not None and ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            try:
                current[key] = float(val)
            except ValueError:
                current[key] = val
    if current is not None:
        pockets.append(current)
    return pockets


def pocket_center_from_atoms(atm_pdb: Path) -> tuple[float, float, float]:
    """pocketN_atm.pdb dosyasındaki (cebi çevreleyen) atomların ağırlık merkezini hesaplar."""
    xs, ys, zs = [], [], []
    for line in atm_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(f"{atm_pdb} içinde atom bulunamadı.")
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def best_druggable_pocket(pdb_path: Path) -> dict:
    """fpocket'i çalıştırır, en yüksek Druggability Score'a sahip pocket'ı seçer ve merkezini hesaplar."""
    out_dir = run_fpocket(pdb_path)
    info_path = out_dir / f"{pdb_path.stem}_info.txt"
    if not info_path.exists():
        raise FileNotFoundError(f"fpocket bilgi dosyası bulunamadı: {info_path}")
    pockets = parse_fpocket_info(info_path)
    if not pockets:
        raise ValueError("fpocket hiçbir bağlanma cebi bulamadı.")
    best = max(pockets, key=lambda p: p.get("Druggability Score", 0.0))
    atm_pdb = out_dir / "pockets" / f"pocket{best['pocket']}_atm.pdb"
    center = pocket_center_from_atoms(atm_pdb)
    return {
        "pocket_number": best["pocket"],
        "druggability": best.get("Druggability Score", 0.0),
        "score": best.get("Score", 0.0),
        "volume": best.get("Volume", 0.0),
        "center": center,
    }


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
