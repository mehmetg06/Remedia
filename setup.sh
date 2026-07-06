#!/usr/bin/env bash
# Remedia pipeline için gerekli tüm araçları tek seferde kurar.
# İdempotent: zaten kuruluysa atlar, tekrar tekrar çalıştırılabilir.
set -e

echo "=================================================="
echo " Remedia kurulum scripti başlıyor..."
echo "=================================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── 1. Python paketleri ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Python paketleri kuruluyor (requirements.txt)..."
pip install -r requirements.txt --break-system-packages

# ── 2. Sistem araçları: openbabel ────────────────────────────────────────────
echo ""
echo "[2/6] Open Babel kontrol ediliyor..."
if command -v obabel >/dev/null 2>&1; then
    echo "  obabel zaten kurulu, atlanıyor."
else
    echo "  obabel bulunamadı, kuruluyor (apt)..."
    sudo apt-get update && sudo apt-get install -y openbabel
fi

# ── 3. Conda kontrolü / Miniconda kurulumu ───────────────────────────────────
echo ""
echo "[3/6] Conda kontrol ediliyor..."
if command -v conda >/dev/null 2>&1; then
    echo "  conda zaten kurulu, atlanıyor."
    # Bu shell içinde conda komutlarını kullanabilmek için ortamı yükle
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
else
    echo "  conda bulunamadı, Miniconda indirilip kuruluyor..."
    MINICONDA_DIR="$HOME/miniconda3"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL -o "$MINICONDA_INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$MINICONDA_INSTALLER" -b -u -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_INSTALLER"

    # shellcheck disable=SC1091
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    "$MINICONDA_DIR/bin/conda" init bash || true
    echo "  Miniconda kuruldu ($MINICONDA_DIR). Yeni terminallerde otomatik aktif olacak."
fi

# ── 4. fpocket ────────────────────────────────────────────────────────────────
echo ""
echo "[4/6] fpocket kontrol ediliyor..."
if command -v fpocket >/dev/null 2>&1; then
    echo "  fpocket zaten kurulu, atlanıyor."
else
    echo "  fpocket bulunamadı, kuruluyor (conda)..."
    conda install -y -c bioconda -c conda-forge fpocket
fi

# ── 5. vina (python paketi) ──────────────────────────────────────────────────
echo ""
echo "[5/6] vina (python) kontrol ediliyor..."
if python3 -c "import vina" >/dev/null 2>&1; then
    echo "  vina python paketi zaten kurulu, atlanıyor."
else
    echo "  vina bulunamadı, pip ile kuruluyor..."
    if ! pip install vina --break-system-packages; then
        echo "  pip ile kurulum başarısız, conda ile deneniyor..."
        conda install -y -c conda-forge vina
    fi
fi

# ── 6. smina (opsiyonel, ikinci docking motoru) ──────────────────────────────
echo ""
echo "[6/6] smina kontrol ediliyor (opsiyonel)..."
if command -v smina >/dev/null 2>&1; then
    echo "  smina zaten kurulu, atlanıyor."
else
    echo "  smina bulunamadı, kuruluyor (conda)..."
    conda install -y -c conda-forge smina || echo "  ⚠️  smina kurulamadı, çapraz doğrulama özelliği bu ortamda çalışmayabilir."
fi

# ── Özet ──────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Kurulum özeti"
echo "=================================================="

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        printf "  \xe2\x9c\x93 %s\n" "$label"
    else
        printf "  \xe2\x9c\x97 %s\n" "$label"
    fi
}

check "obabel"              command -v obabel
check "fpocket"              command -v fpocket
check "smina"                command -v smina
check "vina (python)"        python3 -c "import vina"
check "streamlit"            python3 -c "import streamlit"
check "snakemake"            python3 -c "import snakemake"

echo "=================================================="
echo " Kurulum tamamlandı."
echo "=================================================="
