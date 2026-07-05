# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
fetch_structure.py
Reseptörün 3D yapısını AlphaFold DB REST API'den (veya PDB'den) indirir.

Kullanım:
    python fetch_structure.py --uniprot P30405
    python fetch_structure.py --pdb-id 4CAB
"""
import argparse
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_alphafold(uniprot_id: str) -> Path:
    """AlphaFold DB'den tahmin edilen yapıyı indirir."""
    meta_url = ALPHAFOLD_API.format(uniprot_id=uniprot_id)
    resp = requests.get(meta_url, timeout=30)
    resp.raise_for_status()
    entries = resp.json()
    if not entries:
        raise ValueError(f"AlphaFold DB'de {uniprot_id} için yapı bulunamadı.")

    pdb_url = entries[0]["pdbUrl"]
    pdb_resp = requests.get(pdb_url, timeout=30)
    pdb_resp.raise_for_status()

    out_path = DATA_DIR / f"{uniprot_id}_alphafold.pdb"
    out_path.write_text(pdb_resp.text)
    print(f"[OK] AlphaFold yapısı indirildi: {out_path}")
    print(f"     pLDDT ortalama güven skorunu kontrol etmeyi unutma (B-factor kolonunda).")
    return out_path


def fetch_pdb(pdb_id: str) -> Path:
    """Deneysel yapıyı RCSB PDB'den indirir (varsa AlphaFold yerine tercih edilebilir)."""
    url = PDB_DOWNLOAD.format(pdb_id=pdb_id.upper())
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    out_path = DATA_DIR / f"{pdb_id.upper()}.pdb"
    out_path.write_text(resp.text)
    print(f"[OK] Deneysel yapı indirildi: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Reseptör yapısı indirme")
    parser.add_argument("--uniprot", help="UniProt ID (AlphaFold DB için, örn. P30405)")
    parser.add_argument("--pdb-id", help="PDB ID (deneysel yapı için, örn. 4CAB)")
    args = parser.parse_args()

    if not args.uniprot and not args.pdb_id:
        parser.error("--uniprot veya --pdb-id belirtmelisin.")

    if args.uniprot:
        fetch_alphafold(args.uniprot)
    if args.pdb_id:
        fetch_pdb(args.pdb_id)


if __name__ == "__main__":
    main()
