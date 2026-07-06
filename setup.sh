#!/usr/bin/env bash
# Remedia pipeline için gerekli tüm araçları tek seferde kurar.
# İdempotent: zaten kuruluysa atlar, tekrar tekrar çalıştırılabilir.
#
# ÖNEMLİ: Bu script BİLEREK "set -e" KULLANMAZ. Her araç birbirinden bağımsız
# kurulur; biri başarısız olsa bile diğerleri denenmeye devam eder. Böylece
# örneğin fpocket kurulamasa da AutoDock Vina ve smina yine de kurulur.

echo "=================================================="
echo " Remedia kurulum scripti başlıyor..."
echo "=================================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Yardımcılar ───────────────────────────────────────────────────────────────
# sudo varsa kullan (devcontainer/Codespaces), yoksa boş bırak (root isek gerekmez).
SUDO=""
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

# İkili dosyaları koyacağımız, PATH üzerinde olan dizin.
if [ -w /usr/local/bin ] || [ -n "$SUDO" ]; then
    BIN_DIR="/usr/local/bin"
    BIN_SUDO="$SUDO"
else
    BIN_DIR="$HOME/.local/bin"
    BIN_SUDO=""
    mkdir -p "$BIN_DIR"
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) export PATH="$BIN_DIR:$PATH" ;;
    esac
fi

apt_get_update_done=0
apt_install() {
    # apt paketi kur (varsa sudo ile). Depo güncellemesini bir kez yapar.
    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi
    if [ "$apt_get_update_done" -eq 0 ]; then
        $SUDO apt-get update -y || true
        apt_get_update_done=1
    fi
    $SUDO apt-get install -y "$@"
}

# ── 1. Python paketleri ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Python paketleri kuruluyor (requirements.txt)..."
pip install -r requirements.txt --break-system-packages || \
    pip install -r requirements.txt || \
    echo "  ⚠️  Bazı python paketleri kurulamadı, aşağıdaki özete bak."

# ── 2. Sistem araçları: openbabel ────────────────────────────────────────────
echo ""
echo "[2/6] Open Babel kontrol ediliyor..."
if command -v obabel >/dev/null 2>&1; then
    echo "  obabel zaten kurulu (muhtemelen openbabel-wheel), atlanıyor."
else
    echo "  obabel bulunamadı, kuruluyor (apt)..."
    apt_install openbabel || echo "  ⚠️  obabel apt ile kurulamadı; openbabel-wheel python paketi yeterli olabilir."
fi

# ── 3. AutoDock Vina (python paketi) ─────────────────────────────────────────
# Uygulama Vina'yı `import vina` ile kontrol eder, komut satırı ikilisiyle değil.
echo ""
echo "[3/6] AutoDock Vina (python) kontrol ediliyor..."
if python3 -c "import vina" >/dev/null 2>&1; then
    echo "  vina python paketi zaten kurulu, atlanıyor."
else
    echo "  vina bulunamadı, pip ile kuruluyor (hazır wheel varsa hızlıdır)..."
    pip install vina --break-system-packages || pip install vina || true

    if ! python3 -c "import vina" >/dev/null 2>&1; then
        echo "  Hazır wheel yok; kaynaktan derlemek için Boost + SWIG kuruluyor..."
        apt_install libboost-all-dev swig build-essential || true
        pip install --no-cache-dir --force-reinstall vina --break-system-packages || \
            pip install --no-cache-dir --force-reinstall vina || true
    fi

    if ! python3 -c "import vina" >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
        echo "  pip ile kurulum başarısız, conda ile deneniyor..."
        conda install -y -c conda-forge vina || true
    fi

    if python3 -c "import vina" >/dev/null 2>&1; then
        echo "  ✓ vina kuruldu."
    else
        echo "  ⚠️  vina kurulamadı; skorlama QED tabanlı yedek fitness ile çalışır."
    fi
fi

# ── 4. fpocket ────────────────────────────────────────────────────────────────
# Ubuntu 24.04 depolarında paket yok; kaynaktan derlemek en güvenilir yol.
echo ""
echo "[4/6] fpocket kontrol ediliyor..."
if command -v fpocket >/dev/null 2>&1; then
    echo "  fpocket zaten kurulu, atlanıyor."
else
    echo "  fpocket bulunamadı, kaynaktan derleniyor..."
    apt_install build-essential git >/dev/null 2>&1 || true
    FP_SRC="$(mktemp -d)"
    if git clone --depth 1 https://github.com/Discngine/fpocket.git "$FP_SRC" >/dev/null 2>&1 \
        && make -C "$FP_SRC" >/dev/null 2>&1; then
        # `make install` BINDIR'i /usr/local/bin varsayar; hedef dizine kopyalıyoruz.
        for b in "$FP_SRC"/bin/*; do
            [ -f "$b" ] && [ -x "$b" ] && $BIN_SUDO cp "$b" "$BIN_DIR/"
        done
    fi
    rm -rf "$FP_SRC"

    if ! command -v fpocket >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
        echo "  Kaynaktan derleme başarısız, conda ile deneniyor..."
        conda install -y -c bioconda -c conda-forge fpocket || true
    fi

    if command -v fpocket >/dev/null 2>&1; then
        echo "  ✓ fpocket kuruldu ($(command -v fpocket))."
    else
        echo "  ⚠️  fpocket kurulamadı; cep tespiti çalışmayabilir."
    fi
fi

# ── 5. smina (statik ikili) ──────────────────────────────────────────────────
echo ""
echo "[5/6] smina kontrol ediliyor..."
if command -v smina >/dev/null 2>&1; then
    echo "  smina zaten kurulu, atlanıyor."
else
    echo "  smina bulunamadı, statik ikili indiriliyor..."
    SMINA_TMP="$(mktemp)"
    if curl -fsSL --retry 3 -o "$SMINA_TMP" \
        "https://sourceforge.net/projects/smina/files/smina.static/download" \
        && [ -s "$SMINA_TMP" ]; then
        chmod +x "$SMINA_TMP"
        $BIN_SUDO cp "$SMINA_TMP" "$BIN_DIR/smina"
    fi
    rm -f "$SMINA_TMP"

    if ! command -v smina >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
        echo "  İndirme başarısız, conda ile deneniyor..."
        conda install -y -c conda-forge smina || true
    fi

    if command -v smina >/dev/null 2>&1; then
        echo "  ✓ smina kuruldu ($(command -v smina))."
    else
        echo "  ⚠️  smina kurulamadı; çapraz doğrulama özelliği bu ortamda çalışmayabilir (opsiyonel)."
    fi
fi

# ── 6. Snakemake kontrolü (requirements.txt ile geldi) ───────────────────────
echo ""
echo "[6/6] Snakemake kontrol ediliyor..."
if python3 -c "import snakemake" >/dev/null 2>&1 || command -v snakemake >/dev/null 2>&1; then
    echo "  snakemake kurulu."
else
    echo "  snakemake bulunamadı, pip ile kuruluyor..."
    pip install snakemake --break-system-packages || pip install snakemake || \
        echo "  ⚠️  snakemake kurulamadı."
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
check "vina (python)"       python3 -c "import vina"
check "fpocket"             command -v fpocket
check "smina (opsiyonel)"   command -v smina
check "streamlit"           python3 -c "import streamlit"
check "snakemake"           python3 -c "import snakemake"

echo "=================================================="
echo " Kurulum tamamlandı."
echo " Not: Yeni açtığın terminalde araçların görünmesi için"
echo " gerekiyorsa PATH'i yenile:  hash -r"
echo "=================================================="
