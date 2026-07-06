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

# ── Miniconda ─────────────────────────────────────────────────────────────────
# conda; fpocket/vina/smina için son çare (fallback) kurulum yöntemi olarak
# kullanılıyor. Ortamda conda yoksa Miniconda'yı sessizce kur ki bu fallback'ler
# çalışabilsin. İdempotent: conda zaten varsa (ya da daha önce kurulduysa) atlar.
if ! command -v conda >/dev/null 2>&1; then
    # Daha önce bu script tarafından kurulmuş olabilir; PATH'e alıp tekrar bak.
    if [ -x "$HOME/miniconda3/bin/conda" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi

echo ""
if command -v conda >/dev/null 2>&1; then
    echo "conda zaten kurulu, Miniconda kurulumu atlanıyor."
else
    echo "conda bulunamadı, Miniconda kuruluyor..."
    # Mimariye göre doğru installer'ı seç.
    MC_ARCH="$(uname -m)"
    case "$MC_ARCH" in
        x86_64)          MC_FILE="Miniconda3-latest-Linux-x86_64.sh" ;;
        aarch64|arm64)   MC_FILE="Miniconda3-latest-Linux-aarch64.sh" ;;
        *)               MC_FILE="" ;;
    esac

    if [ -z "$MC_FILE" ]; then
        echo "  ⚠️  Desteklenmeyen mimari ($MC_ARCH); Miniconda kurulamadı."
    else
        MC_TMP="$(mktemp)"
        MC_PREFIX="$HOME/miniconda3"
        if curl -fsSL --retry 3 -o "$MC_TMP" "https://repo.anaconda.com/miniconda/$MC_FILE" \
            && bash "$MC_TMP" -b -p "$MC_PREFIX" >/dev/null 2>&1; then
            export PATH="$MC_PREFIX/bin:$PATH"
            echo "  ✓ Miniconda kuruldu ($MC_PREFIX)."
        else
            echo "  ⚠️  Miniconda kurulamadı; conda tabanlı yedek kurulumlar atlanacak."
        fi
        rm -f "$MC_TMP"
    fi
fi

# ── conda ToS onayı ───────────────────────────────────────────────────────────
# Miniconda kurulduktan HEMEN SONRA, conda ile herhangi bir paket (fpocket/vina/
# smina) kurulmaya çalışılmadan ÖNCE Anaconda kanallarının Kullanım Koşullarını
# (Terms of Service) otomatik ve sessizce onayla. Aksi halde conda install şu
# hatayla düşer:
#   CondaToSNonInteractiveError: Terms of Service have not been accepted ...
# Sadece conda varsa çalışır; idempotenttir (tekrar onaylamak zararsızdır).
if command -v conda >/dev/null 2>&1; then
    echo ""
    echo "conda kanalları için Kullanım Koşulları (ToS) onaylanıyor..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
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
    apt_install build-essential g++ git >/dev/null 2>&1 || true
    FP_SRC="$(mktemp -d)"
    FP_LOG="$(mktemp)"
    # ÖNEMLİ: fpocket Makefile'ı, ortamda CXX değişkeni tam olarak "g++" değilse
    # (birçok devcontainer/Codespaces CXX'i g++-13, clang++ vb. olarak export eder)
    # sessizce clang'a düşer. clang kurulu değilse derleme başarısız olur ve bu
    # yüzden "fpocket hariç her şey kuruldu" durumu oluşur. Derlemeyi gcc/g++ ile
    # sabitlemek için CXX=g++ CC=gcc'yi açıkça geçiriyoruz.
    # NOT: fpocket Makefile'ı paralel derlemeye uygun DEĞİL (qhull alt dizininde
    # yarış koşulu var), bu yüzden -j KULLANMIYORUZ; seri derleme güvenilir.
    if git clone --depth 1 https://github.com/Discngine/fpocket.git "$FP_SRC" >/dev/null 2>&1 \
        && make -C "$FP_SRC" CXX=g++ CC=gcc >"$FP_LOG" 2>&1; then
        # `make install` BINDIR'i /usr/local/bin varsayar; hedef dizine kopyalıyoruz.
        for b in "$FP_SRC"/bin/*; do
            [ -f "$b" ] && [ -x "$b" ] && $BIN_SUDO cp "$b" "$BIN_DIR/"
        done
    else
        echo "  ⚠️  Kaynaktan derleme başarısız. Son satırlar:"
        tail -n 15 "$FP_LOG" 2>/dev/null | sed 's/^/      /'
    fi
    rm -rf "$FP_SRC" "$FP_LOG"

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
