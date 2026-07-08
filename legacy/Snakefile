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
# UI'dan run_id ile çalıştır (Bölüm C — izolasyon):
#     snakemake --cores 1 --config ligands_file=data/generated.smi run_id=run_20260706_142031_a3f2
#
# Snakemake her rule'un input/output dosyalarını izler; sadece eksik ya da
# girdisi değişmiş adımları yeniden çalıştırır (caching). Parametreler
# config.yaml'dan okunur.
# ============================================================================

configfile: "config.yaml"

# --- config.yaml'dan türetilen yardımcı değerler ---------------------------
UNIPROT    = config["uniprot_id"]
CENTER     = config["pocket_center"]            # [x, y, z]
BOX        = config["box_size"]                 # [sx, sy, sz]
LIGANDS    = config["ligands_file"]
EXHAUST    = config["exhaustiveness"]
ADMET_MODE = config["admet_mode"]

# --- run_id izolasyonu (Bölüm C) -------------------------------------------
# run_id --config ile geçirilir; yoksa "default" klasörü kullanılır.
RUN_ID     = config.get("run_id", "default")
RUN_DIR    = f"results/{RUN_ID}"

# Ara/çıktı dosya yolları
RECEPTOR_PDB   = f"data/{UNIPROT}_alphafold.pdb"
RECEPTOR_PDBQT = f"data/{UNIPROT}_alphafold.pdbqt"
POCKET_INFO    = "data/pocket_info.txt"
LIGANDS_DIR    = f"data/ligands_prepared_{RUN_ID}"
DOCKING_CSV    = f"{RUN_DIR}/docking_scores.csv"
ADMET_CSV      = f"{RUN_DIR}/admet_results.csv"
RANKING_CSV    = f"{RUN_DIR}/final_ranking.csv"
DASHBOARD      = f"{RUN_DIR}/dashboard.html"
INPUT_COPY     = f"{RUN_DIR}/input_ligands.smi"

# Molekül üretimi (opsiyonel giriş adımı) parametreleri
GEN_METHOD  = config.get("gen_method", "brics")   # random | brics | genetic
GEN_SEEDS   = config.get("gen_seeds_file", LIGANDS)
GEN_N       = config.get("gen_n", 50)
GEN_OUTPUT  = "data/generated.smi"


# --- Varsayılan hedef: her şey hazır olsun ---------------------------------
rule all:
    input:
        DASHBOARD


# 0) (OPSİYONEL) Sıfırdan yeni molekül üret — kural tabanlı, model eğitimsiz.
#    Çıktısı doğrudan ligand_prep'e girebilen bir .smi dosyasıdır.
#    Üretilen molekülleri pipeline'a sokmak için:
#        snakemake --cores 1 generate
#        snakemake --cores 1 --config ligands_file=data/generated.smi run_id=<id>
rule generate:
    input:
        GEN_SEEDS
    output:
        GEN_OUTPUT
    shell:
        "python src/molecule_generator.py --method {GEN_METHOD} "
        "--seeds-file {input} --n {GEN_N} --output {output}"


# 0b) Ligand dosyasını run klasörüne kopyala (izolasyon için)
rule copy_input_ligands:
    input:
        LIGANDS
    output:
        INPUT_COPY
    shell:
        "mkdir -p {RUN_DIR} && cp {input} {output}"


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


# 4) GNINA docking — GPU'da Google Colab'da yapılır (Vina KALDIRILDI).
#    Bu kural yerel docking ÇALIŞTIRMAZ; sadece Colab'dan yüklenen
#    docking_scores.csv'yi bekler/doğrular:
#      - Dosya zaten varsa (kullanıcı Colab'dan indirip results/<run_id>/'a
#        yükledi)  → formatı doğrulanır ve pipeline devam eder.
#      - Dosya yoksa → kullanıcıya Colab'da GNINA çalıştırıp sonucu buraya
#        koyması söylenir ve pipeline burada NET bir mesajla durur.
#    Not: Snakemake, çıktı dosyası zaten mevcut ve girdilerinden yeniyse bu
#    kuralı hiç çalıştırmadan atlar; böylece yüklenen CSV doğrudan kullanılır.
rule docking:
    input:
        INPUT_COPY
    output:
        DOCKING_CSV
    shell:
        r"""
        if [ -f "{output}" ]; then
            echo "• docking_scores.csv bulundu — format doğrulanıyor..."
            python src/docking.py --validate-only "{output}"
        else
            echo "============================================================"
            echo "⏸️  DOCKING BEKLENİYOR — GNINA'yı Colab'da (GPU) çalıştır."
            echo "------------------------------------------------------------"
            echo "1. Colab'da aç:"
            echo "   https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/gnina_colab.ipynb"
            echo "2. Runtime > Change runtime type > GPU (T4) seç."
            echo "3. Tüm hücreleri sırayla çalıştır (Shift+Enter)."
            echo "4. İnen docking_scores.csv'yi ŞURAYA yükle:"
            echo "   {output}"
            echo "5. Bu komutu tekrar çalıştır — dosya bulununca ADMET →"
            echo "   sıralama → dashboard otomatik devam eder."
            echo "============================================================"
            exit 1
        fi
        """


# 5) ADMET filtresi (Lipinski/Veber veya ADMETlab)
rule admet_filter:
    input:
        LIGANDS
    output:
        ADMET_CSV
    shell:
        "mkdir -p {RUN_DIR} && "
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
#    run_id bilgisi de dashboard'a gömülür.
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
        "--config {input.config} --output {output} "
        "&& echo '{RUN_ID}' > results/latest_run.txt"
