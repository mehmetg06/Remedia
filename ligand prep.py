# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
ligand_prep.py
SMILES listesinden 3D konformasyon üretir ve docking için PDBQT'ye çevirir.

Girdi formatı (data/ligands.smi):
    SMILES  isim
    CC(=O)Oc1ccccc1C(=O)O  aspirin
    ...

Kullanım:
    python ligand_prep.py --input data/ligands.smi --output data/ligands_prepared/
"""
import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def prepare_ligand(smiles: str, name: str, out_dir: Path) -> Path:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[UYARI] Geçersiz SMILES, atlandı: {name} ({smiles})")
        return None

    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status != 0:
        print(f"[UYARI] 3D embedding başarısız: {name}")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    sdf_path = out_dir / f"{name}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return sdf_path


def convert_to_pdbqt(sdf_path: Path):
    """meeko ile SDF -> PDBQT dönüşümü (docking.py içinde de çağrılabilir)."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit.Chem import SDMolSupplier
    except ImportError:
        print("[UYARI] meeko kurulu değil: pip install meeko")
        return None

    supplier = SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(iter(supplier))
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    pdbqt_path = sdf_path.with_suffix(".pdbqt")
    pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
    pdbqt_path.write_text(pdbqt_string)
    return pdbqt_path


def main():
    parser = argparse.ArgumentParser(description="Ligand hazırlama (SMILES -> 3D -> PDBQT)")
    parser.add_argument("--input", required=True, help="SMILES dosyası (tab/space ayrılmış: smiles isim)")
    parser.add_argument("--output", default="data/ligands_prepared", help="Çıktı klasörü")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    count_ok, count_fail = 0, 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles, name = parts[0], (parts[1] if len(parts) > 1 else f"lig_{count_ok+count_fail}")

            sdf_path = prepare_ligand(smiles, name, out_dir)
            if sdf_path is None:
                count_fail += 1
                continue

            pdbqt_path = convert_to_pdbqt(sdf_path)
            if pdbqt_path:
                print(f"[OK] {name} -> {pdbqt_path}")
                count_ok += 1
            else:
                count_fail += 1

    print(f"\nToplam: {count_ok} başarılı, {count_fail} başarısız.")


if __name__ == "__main__":
    main()
