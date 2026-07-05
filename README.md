# Receptor-Focused Drug Discovery Pipeline

Açık kaynak, reseptör hedefli (structure-based) ilaç molekülü keşif pipeline'ı.
AlphaFold DB → Pocket Detection → Virtual Screening → Docking (Vina) → ADMET Filtreleme

## Mimari

```
Reseptör (UniProt ID / PDB ID)
        │
        ▼
1. fetch_structure.py     → AlphaFold/PDB'den 3D yapı indir
        │
        ▼
2. pocket_detection.py    → Binding site (cep) tespiti
        │
        ▼
3. ligand_prep.py         → Aday moleküller (ZINC/ChEMBL'den veya SMILES listesi)
        │
        ▼
4. docking.py             → AutoDock Vina ile skorlama
        │
        ▼
5. admet_filter.py        → ADMETlab 3.0 API ile farmakokinetik eleme
        │
        ▼
6. rank_report.py         → Birleşik skor + rapor (CSV/HTML)
```

## Kurulum (GitHub Codespaces / Linux)

```bash
pip install -r requirements.txt
# Vina binary (conda önerilir):
conda install -c conda-forge vina
# veya: pip install vina
```

## Hızlı Başlangıç

```bash
python src/fetch_structure.py --uniprot P30405   # örnek: CypD (PPIF)
python src/pocket_detection.py --pdb data/structure.pdb
python src/docking.py --receptor data/structure.pdbqt --ligands data/ligands.smi
python src/admet_filter.py --results results/docking_scores.csv
```

## Durum

🚧 İskelet aşaması — her modül minimal çalışır halde, geliştirilmeye açık.

## Lisans

AGPL-3.0 — özgürce fork'la, katkı ver; ancak türev çalışmalar (ağ üzerinden
sunulan servisler dahil) de açık kaynak kalmak zorunda. Bkz. [LICENSE](LICENSE).
