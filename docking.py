# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
docking.py
AutoDock Vina ile reseptör-ligand docking (batch virtual screening).

Kullanım:
    python docking.py --receptor data/structure.pdbqt \
                       --ligands-dir data/ligands_prepared \
                       --center 12.3 45.1 -3.2 --size 20 20 20 \
                       --output results/docking_scores.csv
"""
import argparse
import csv
from pathlib import Path

from vina import Vina


def dock_all(receptor_pdbqt: Path, ligands_dir: Path, center, box_size, exhaustiveness=8):
    v = Vina(sf_name="vina")
    v.set_receptor(str(receptor_pdbqt))

    results = []
    ligand_files = sorted(Path(ligands_dir).glob("*.pdbqt"))
    print(f"{len(ligand_files)} ligand bulundu, docking başlıyor...")

    for lig_path in ligand_files:
        try:
            v.set_ligand_from_file(str(lig_path))
            v.compute_vina_maps(center=center, box_size=box_size)
            v.dock(exhaustiveness=exhaustiveness, n_poses=10)
            energies = v.energies(n_poses=1)
            best_score = float(energies[0][0])  # kcal/mol, en düşük en iyi
            results.append({"ligand": lig_path.stem, "affinity_kcal_mol": best_score})
            print(f"[OK] {lig_path.stem}: {best_score:.3f} kcal/mol")

            out_pose = lig_path.parent / f"{lig_path.stem}_docked.pdbqt"
            v.write_poses(str(out_pose), n_poses=1, overwrite=True)
        except Exception as e:
            print(f"[HATA] {lig_path.stem}: {e}")
            results.append({"ligand": lig_path.stem, "affinity_kcal_mol": None})

    return results


def main():
    parser = argparse.ArgumentParser(description="Vina batch docking")
    parser.add_argument("--receptor", required=True, help="Reseptör PDBQT dosyası")
    parser.add_argument("--ligands-dir", required=True, help="Hazırlanmış ligand PDBQT klasörü")
    parser.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, default=[20, 20, 20], metavar=("SX", "SY", "SZ"))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--output", default="results/docking_scores.csv")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = dock_all(
        Path(args.receptor), Path(args.ligands_dir),
        center=args.center, box_size=args.size,
        exhaustiveness=args.exhaustiveness,
    )

    results.sort(key=lambda r: (r["affinity_kcal_mol"] is None, r["affinity_kcal_mol"]))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ligand", "affinity_kcal_mol"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] Sonuçlar kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
