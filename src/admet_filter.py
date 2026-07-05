# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
admet_filter.py
Docking sonrası aday moleküllerin ADMET (Absorpsiyon, Dağılım, Metabolizma,
Atılım, Toksisite) profiline göre filtrelenmesi.

İki mod:
1. Lokal Lipinski/Veber kuralları (hızlı, offline, RDKit ile)
2. ADMETlab 3.0 API çağrısı (daha kapsamlı, internet gerektirir)

Kullanım:
    python admet_filter.py --smiles-file data/ligands.smi --mode lipinski
    python admet_filter.py --smiles-file data/ligands.smi --mode admetlab
"""
import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def lipinski_veber_filter(smiles: str, name: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ligand": name, "pass": False, "reason": "geçersiz SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = []
    if mw > 500:
        violations.append("MW>500")
    if logp > 5:
        violations.append("LogP>5")
    if hbd > 5:
        violations.append("HBD>5")
    if hba > 10:
        violations.append("HBA>10")
    if rotb > 10:
        violations.append("RotB>10 (Veber)")
    if tpsa > 140:
        violations.append("TPSA>140 (Veber)")

    return {
        "ligand": name, "MW": round(mw, 1), "LogP": round(logp, 2),
        "HBD": hbd, "HBA": hba, "RotB": rotb, "TPSA": round(tpsa, 1),
        "pass": len(violations) <= 1,  # Lipinski: en fazla 1 ihlale izin ver
        "violations": ";".join(violations) if violations else "-",
    }


def admetlab_filter(smiles: str, name: str) -> dict:
    """ADMETlab 3.0 API'sine istek gönderir (endpoint güncel değilse hata verebilir,
    ADMETlab web arayüzünü kontrol et: https://admetlab3.scbdd.com/)."""
    import requests
    try:
        resp = requests.post(
            "https://admetlab3.scbdd.com/api/predict",
            json={"smiles": smiles}, timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"ligand": name, **data}
    except Exception as e:
        return {"ligand": name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ADMET filtreleme")
    parser.add_argument("--smiles-file", required=True, help="SMILES dosyası (smiles isim)")
    parser.add_argument("--mode", choices=["lipinski", "admetlab"], default="lipinski")
    parser.add_argument("--output", default="results/admet_results.csv")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.smiles_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles, name = parts[0], (parts[1] if len(parts) > 1 else "unnamed")

            if args.mode == "lipinski":
                rows.append(lipinski_veber_filter(smiles, name))
            else:
                rows.append(admetlab_filter(smiles, name))

    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    passed = sum(1 for r in rows if r.get("pass"))
    print(f"[OK] {len(rows)} molekül değerlendirildi, {passed} tanesi filtreyi geçti.")
    print(f"Sonuçlar: {args.output}")


if __name__ == "__main__":
    main()
