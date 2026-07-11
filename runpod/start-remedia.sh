#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="/opt/remedia-src"
TARGET_DIR="${REMEDIA_HOME:-/workspace/Remedia}"
WORKSPACE_DIR="${REMEDIA_WORKSPACE:-/workspace}"
RESULTS_DIR="${WORKSPACE_DIR}/Remedia_results"
CACHE_DIR="${WORKSPACE_DIR}/remedia_cache"

mkdir -p "${TARGET_DIR}" "${RESULTS_DIR}" "${CACHE_DIR}"

if [[ "${REMEDIA_SYNC_CODE:-1}" == "1" ]]; then
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    "${SOURCE_DIR}/" "${TARGET_DIR}/"
elif [[ ! -f "${TARGET_DIR}/notebooks/remedia_runpod.ipynb" ]]; then
  rsync -a "${SOURCE_DIR}/" "${TARGET_DIR}/"
fi

export PYTHONPATH="${TARGET_DIR}/src:${PYTHONPATH:-}"
export GNINA_PATH="${GNINA_PATH:-/usr/local/bin/gnina}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "UYARI: nvidia-smi bulunamadı. NVIDIA GPU'lu RunPod Pod seçildiğinden emin ol."
fi

JUPYTER_TOKEN="${JUPYTER_PASSWORD:-remedia}"
NOTEBOOK_PATH="Remedia/notebooks/remedia_runpod.ipynb"

echo "============================================================"
echo "Remedia RunPod hazır"
echo "Notebook: /workspace/${NOTEBOOK_PATH}"
echo "Jupyter token: ${JUPYTER_TOKEN}"
echo "Sonuçlar: ${RESULTS_DIR}"
echo "============================================================"

exec jupyter lab \
  --allow-root \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.root_dir="${WORKSPACE_DIR}" \
  --ServerApp.default_url="/lab/tree/${NOTEBOOK_PATH}" \
  --ServerApp.allow_origin='*' \
  --IdentityProvider.token="${JUPYTER_TOKEN}"
