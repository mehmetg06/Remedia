#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/.remedia-tools"
BIN_DIR="${TOOLS_DIR}/bin"
ENV_PREFIX="${TOOLS_DIR}/env"
MICROMAMBA="${BIN_DIR}/micromamba"
GNINA="${BIN_DIR}/gnina"
GNINA_VERSION="${GNINA_VERSION:-v1.3}"

log() { printf '\n\033[1;36m[Remedia]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[Hata]\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "Bu kurulum Linux veya WSL2 içindir. macOS'ta mevcut CUDA/GNINA akışı yerel çalışmaz; Colab, Modal veya RunPod kullan."
[[ "$(uname -m)" == "x86_64" ]] || fail "Hazır GNINA binary'si için x86_64 işlemci gerekiyor."

for command in curl tar; do
  command -v "${command}" >/dev/null 2>&1 || fail "${command} kurulu değil."
done

if [[ "${REMEDIA_SKIP_GPU_CHECK:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi bulunamadı. NVIDIA sürücüsünü kur veya NVIDIA destekli Docker yolunu kullan."
  nvidia-smi >/dev/null 2>&1 || fail "NVIDIA GPU erişilemiyor."
fi

mkdir -p "${BIN_DIR}" "${ROOT_DIR}/local_workspace/Remedia_results" "${ROOT_DIR}/local_workspace/remedia_cache"

if [[ ! -x "${MICROMAMBA}" ]]; then
  log "micromamba indiriliyor"
  curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
    | tar -xj -C "${BIN_DIR}" --strip-components=1 bin/micromamba
  chmod 0755 "${MICROMAMBA}"
fi

if [[ "${REMEDIA_RECREATE_ENV:-0}" == "1" && -d "${ENV_PREFIX}" ]]; then
  log "Eski ortam siliniyor"
  rm -rf "${ENV_PREFIX}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  log "Conda ortamı oluşturuluyor"
  "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" -f "${ROOT_DIR}/environment.yml"
else
  log "Mevcut Conda ortamı kullanılacak. Yeniden kurmak için REMEDIA_RECREATE_ENV=1 kullan."
fi
"${MICROMAMBA}" clean --all --yes >/dev/null 2>&1 || true

if [[ ! -x "${GNINA}" ]]; then
  log "GNINA ${GNINA_VERSION} indiriliyor"
  curl --fail --location --retry 3 \
    "https://github.com/gnina/gnina/releases/download/${GNINA_VERSION}/gnina" \
    --output "${GNINA}"
  chmod 0755 "${GNINA}"
fi

log "Araçlar doğrulanıyor"
"${GNINA}" --version
"${MICROMAMBA}" run -p "${ENV_PREFIX}" fpocket --help >/dev/null
"${MICROMAMBA}" run -p "${ENV_PREFIX}" python - <<'PY'
import pandas
from rdkit import Chem
assert Chem.MolFromSmiles("CCO") is not None
print("Python, RDKit ve pandas hazır.")
PY

log "Jupyter kernel kaydediliyor"
"${MICROMAMBA}" run -p "${ENV_PREFIX}" \
  python -m ipykernel install --user \
  --name remedia-local \
  --display-name "Remedia Local (GPU)" >/dev/null

log "GPU gerektirmeyen testler çalıştırılıyor"
(
  cd "${ROOT_DIR}"
  "${MICROMAMBA}" run -p "${ENV_PREFIX}" python -m unittest discover -s tests -v
)

cat <<EOF

Kurulum tamamlandı.

Notebook'u aç:
  cd "${ROOT_DIR}"
  REMEDIA_HOME="${ROOT_DIR}" \
  REMEDIA_WORKSPACE="${ROOT_DIR}/local_workspace" \
  GNINA_PATH="${GNINA}" \
  "${MICROMAMBA}" run -p "${ENV_PREFIX}" \
  jupyter lab notebooks/remedia_local.ipynb

Sonuçlar:
  ${ROOT_DIR}/local_workspace/Remedia_results/
EOF
