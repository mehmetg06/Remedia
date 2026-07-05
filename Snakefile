# ============================================================================
# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı
# ----------------------------------------------------------------------------
# Tüm adımları birbirine bağlayan Snakemake iş akışı:
#
#   fetch_structure → prepare_receptor ┐
#                   → detect_pocket    ├→ docking ┐
#   prepare_ligands ───────────────────┘          ├→ rank_report → dashboard
#   admet_filter ─────────────────────────────────┘
#
# Tek komutla çalıştır:
#     snakemake --cores 1
#
# Snakemake her rule'un input/output dosyalarını izler; sadece eksik ya da
# girdisi değişmiş adımları yeniden çalıştırır (caching). Parametreler
# config.yaml'dan okunur.
# ============================================================================

configfile: "config.yaml"

# --- config.yaml'dan türetilen yardımcı değerler ---------------------------
UNIPROT   = config["uniprot_id"]
CENTER    = config["pocket_center"]            # [x, y, z]
BOX       = config["box_size"]                 # [sx, sy, sz]
LIGANDS   = config["ligands_file"]
EXHAUST   = config["exhaustiveness"]
ADMET_MODE = config["admet_mode"]

# Ara/çıktı dosya yolları (tek yerde tanımlı ki rule'lar tutarlı kalsın)
RECEPTOR_PDB   = f"data/{UNIPROT}_alphafold.pdb"
RECEPTOR_PDBQT = f"data/{UNIPROT}_alphafold.pdbqt"
POCKET_INFO    = "data/pocket_info.txt"
LIGANDS_DIR    = "data/ligands_prepared"
DOCKING_CSV    = "results/docking_scores.csv"
ADMET_CSV      = "results/admet_results.csv"
RANKING_CSV    = "results/final_ranking.csv"
DASHBOARD      = "dashboard.html"


# --- Varsayılan hedef: her şey hazır olsun ---------------------------------
rule all:
    input:
        DASHBOARD


# 1) Reseptör yapısını AlphaFold DB'den indir
rule fetch_structure:
    output:
        RECEPTOR_PDB
    shell:
        "python src/fetch_structure.py --uniprot {UNIPROT}"


# 2a) Reseptörü docking için PDBQT'ye çevir (PDB -> PDBQT, rigid receptor).
#     Script'ler bu köprüyü sağlamadığından openbabel ile burada yapılıyor.
rule prepare_receptor:
    input:
        RECEPTOR_PDB
    output:
        RECEPTOR_PDBQT
    shell:
        "obabel {input} -O {output} -xr"


# 2b) Binding pocket bilgisini config'deki merkezle sabitle (çıktı loglanır).
rule detect_pocket:
    input:
        RECEPTOR_PDB
    output:
        POCKET_INFO
    shell:
        "python src/pocket_detection.py --pdb {input} "
        "--center {CENTER[0]} {CENTER[1]} {CENTER[2]} --size {BOX[0]} "
        "> {output}"


# 3) Ligandları hazırla (SMILES -> 3D -> PDBQT)
rule prepare_ligands:
    input:
        LIGANDS
    output:
        directory(LIGANDS_DIR)
    shell:
        "python src/ligand_prep.py --input {input} --output {output}"


# 4) AutoDock Vina ile batch docking
rule docking:
    input:
        receptor=RECEPTOR_PDBQT,
        ligands=LIGANDS_DIR,
        pocket=POCKET_INFO,
    output:
        DOCKING_CSV
    shell:
        "python src/docking.py "
        "--receptor {input.receptor} "
        "--ligands-dir {input.ligands} "
        "--center {CENTER[0]} {CENTER[1]} {CENTER[2]} "
        "--size {BOX[0]} {BOX[1]} {BOX[2]} "
        "--exhaustiveness {EXHAUST} "
        "--output {output}"


# 5) ADMET filtresi (Lipinski/Veber veya ADMETlab)
rule admet_filter:
    input:
        LIGANDS
    output:
        ADMET_CSV
    shell:
        "python src/admet_filter.py "
        "--smiles-file {input} --mode {ADMET_MODE} --output {output}"


# 6) Docking + ADMET birleşik nihai sıralama
rule rank_report:
    input:
        docking=DOCKING_CSV,
        admet=ADMET_CSV,
    output:
        RANKING_CSV
    shell:
        "python src/rank_report.py "
        "--docking {input.docking} --admet {input.admet} --output {output}"


# 7) Dashboard'ı güncel verilerle yeniden üret (son adım)
rule dashboard:
    input:
        ranking=RANKING_CSV,
        admet=ADMET_CSV,
        pocket=POCKET_INFO,
        config="config.yaml",
    output:
        DASHBOARD
    shell:
        "python src/generate_dashboard.py "
        "--ranking {input.ranking} --admet {input.admet} "
        "--config {input.config} --output {output}"
