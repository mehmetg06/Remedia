#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REMEDIA_REPO_URL:-https://github.com/mehmetg06/Remedia.git}"
REPO_REF="${REMEDIA_REF:-main}"
WORKSPACE_DIR="${REMEDIA_WORKSPACE:-/workspace}"
REPO_DIR="${WORKSPACE_DIR}/Remedia"
TOOLS_DIR="${WORKSPACE_DIR}/.remedia-tools"
MARKER="${TOOLS_DIR}/installed-v2"

mkdir -p "${WORKSPACE_DIR}" "${TOOLS_DIR}/bin"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin "${REPO_REF}" --depth 1
  git -C "${REPO_DIR}" checkout "${REPO_REF}"
  git -C "${REPO_DIR}" pull --ff-only origin "${REPO_REF}" || true
fi

if [[ ! -f "${MARKER}" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r "${REPO_DIR}/runpod/requirements.txt"

  if [[ ! -x "${TOOLS_DIR}/bin/micromamba" ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "${TOOLS_DIR}/bin" --strip-components=1 bin/micromamba
  fi
  "${TOOLS_DIR}/bin/micromamba" create -y \
    -p "${TOOLS_DIR}/fpocket" -c conda-forge -c bioconda fpocket
  "${TOOLS_DIR}/bin/micromamba" clean --all --yes

  curl --fail --location --retry 3 \
    "https://github.com/gnina/gnina/releases/download/v1.3/gnina" \
    --output "${TOOLS_DIR}/bin/gnina"
  chmod 0755 "${TOOLS_DIR}/bin/gnina"
  touch "${MARKER}"
fi

export PATH="${TOOLS_DIR}/fpocket/bin:${TOOLS_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export GNINA_PATH="${TOOLS_DIR}/bin/gnina"

cat > "${WORKSPACE_DIR}/remedia-env.sh" <<EOF
export PATH="${TOOLS_DIR}/fpocket/bin:${TOOLS_DIR}/bin:\$PATH"
export PYTHONPATH="${REPO_DIR}/src:\${PYTHONPATH:-}"
export GNINA_PATH="${TOOLS_DIR}/bin/gnina"
export REMEDIA_HOME="${REPO_DIR}"
export REMEDIA_WORKSPACE="${WORKSPACE_DIR}"
EOF

POD_ID="${RUNPOD_POD_ID:-YOUR_POD_ID}"
URL="https://${POD_ID}-8888.proxy.runpod.net/lab/tree/Remedia/notebooks/remedia_runpod.ipynb"

echo
echo "✅ Remedia kuruldu."
echo "Notebook bağlantısı:"
echo "${URL}"
echo
echo "Yeni terminal açarsan önce şunu çalıştır:"
echo "source ${WORKSPACE_DIR}/remedia-env.sh"
