# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
rank_report.py
Docking skorlarını ve ADMET filtresini birleştirip nihai aday listesini sıralar.

Kullanım:
    python rank_report.py --docking results/docking_scores.csv \
                           --admet results/admet_results.csv \
                           --output results/final_ranking.csv
"""
import argparse
import csv
from pathlib import Path


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Nihai sıralama raporu")
    parser.add_argument("--docking", required=True)
    parser.add_argument("--admet", required=True)
    parser.add_argument("--output", default="results/final_ranking.csv")
    args = parser.parse_args()

    docking_rows = {r["ligand"]: r for r in load_csv(args.docking)}
    admet_rows = {r["ligand"]: r for r in load_csv(args.admet)}

    combined = []
    for name, d in docking_rows.items():
        a = admet_rows.get(name, {})
        affinity = d.get("affinity_kcal_mol")
        passed_admet = a.get("pass") == "True"

        combined.append({
            "ligand": name,
            "affinity_kcal_mol": affinity,
            "admet_pass": passed_admet,
            "MW": a.get("MW", "-"),
            "LogP": a.get("LogP", "-"),
            "violations": a.get("violations", "-"),
        })

    # Önce ADMET'i geçenler, sonra affinity'ye göre (düşük = daha iyi)
    def sort_key(r):
        aff = r["affinity_kcal_mol"]
        aff_val = float(aff) if aff not in (None, "", "None") else 999.0
        return (not r["admet_pass"], aff_val)

    combined.sort(key=sort_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined[0].keys())
        writer.writeheader()
        writer.writerows(combined)

    print(f"[OK] Nihai sıralama: {args.output}")
    print("\nTop 5 aday:")
    for r in combined[:5]:
        print(f"  {r['ligand']:20s} affinity={r['affinity_kcal_mol']:>8}  ADMET_pass={r['admet_pass']}")


if __name__ == "__main__":
    main()
