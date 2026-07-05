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
pip install -r requirements.txt   # snakemake dahil tüm bağımlılıklar
# Vina binary (conda önerilir):
conda install -c conda-forge vina
# veya: pip install vina
```

## Hızlı Başlangıç — Tek Komut

Tüm pipeline artık **Snakemake** ile otomatik çalışıyor. Parametreler
`config.yaml`'dan okunur; tek yapman gereken:

```bash
snakemake --cores 1
```

Bu komut sırayla ve **sadece eksik/değişen adımları** çalıştırır:

1. Yapı yoksa AlphaFold DB'den indirir (`fetch_structure.py`)
2. Reseptörü docking için PDBQT'ye çevirir (openbabel)
3. Binding pocket'ı `config.yaml`'daki merkezle sabitler (`pocket_detection.py`)
4. Ligandları hazırlar: SMILES → 3D → PDBQT (`ligand_prep.py`)
5. AutoDock Vina ile dockler (`docking.py`)
6. ADMET filtresi uygular (`admet_filter.py`)
7. Nihai sıralamayı üretir (`rank_report.py`)
8. `dashboard.html`'i güncel sonuçlarla yeniden yazar (`generate_dashboard.py`)

Snakemake input/output dosyalarını izlediği için, örneğin sadece ligand
listesini değiştirirsen yapı indirme adımı tekrar çalışmaz (caching).

### Parametreleri değiştirme (`config.yaml`)

Farklı bir reseptör, cep merkezi veya ligand listesi için `config.yaml`'ı
düzenle — komut satırında bir şey yazmana gerek yok:

```yaml
uniprot_id: "P30405"                 # AlphaFold DB UniProt ID (örn. CypD/PPIF)
pocket_center: [5.00, -1.02, -15.56] # docking box merkezi (x, y, z)
box_size: [20.0, 20.0, 20.0]         # kutu boyutu (Å)
ligands_file: "data/ligands_example.smi"
exhaustiveness: 8                    # Vina arama yoğunluğu
admet_mode: "lipinski"              # veya "admetlab"
```

Tek seferlik override de mümkün:

```bash
snakemake --cores 1 --config exhaustiveness=16
```

Faydalı komutlar:

```bash
snakemake -n            # dry-run: hangi adımların çalışacağını göster
snakemake --dag | dot -Tsvg > dag.svg   # bağımlılık grafiği
snakemake --forceall    # her şeyi baştan çalıştır
```

## Manuel çalıştırma (debug için)

Snakemake akışı önerilir. Tek tek adımları elle çalıştırmak istersen
(hata ayıklama, ara çıktı inceleme) script'ler bağımsız da çalışır:

```bash
python src/fetch_structure.py --uniprot P30405            # → data/P30405_alphafold.pdb
obabel data/P30405_alphafold.pdb -O data/P30405_alphafold.pdbqt -xr
python src/pocket_detection.py --pdb data/P30405_alphafold.pdb \
    --center 5.00 -1.02 -15.56 --size 20
python src/ligand_prep.py --input data/ligands_example.smi \
    --output data/ligands_prepared
python src/docking.py --receptor data/P30405_alphafold.pdbqt \
    --ligands-dir data/ligands_prepared \
    --center 5.00 -1.02 -15.56 --size 20 20 20 \
    --output results/docking_scores.csv
python src/admet_filter.py --smiles-file data/ligands_example.smi \
    --mode lipinski --output results/admet_results.csv
python src/rank_report.py --docking results/docking_scores.csv \
    --admet results/admet_results.csv --output results/final_ranking.csv
python src/generate_dashboard.py --ranking results/final_ranking.csv \
    --admet results/admet_results.csv --config config.yaml --output dashboard.html
```

## Durum

🚧 İskelet aşaması — her modül minimal çalışır halde, geliştirilmeye açık.

## Lisans

AGPL-3.0 — özgürce fork'la, katkı ver; ancak türev çalışmalar (ağ üzerinden
sunulan servisler dahil) de açık kaynak kalmak zorunda. Bkz. [LICENSE](LICENSE).
