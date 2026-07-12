"""Deploy a fixed-URL Remedia JupyterLab on Modal with workspace authentication.

Deploy:
    modal deploy modal/remedia_web_jupyter.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "remedia-web-jupyter"
VOLUME_NAME = "remedia-data"
VOLUME_PATH = Path("/workspace")
REPO_PATH = VOLUME_PATH / "Remedia"

PYTHON_PACKAGES = [
    "requests>=2.31.0",
    "rdkit>=2023.9.1",
    "biopython>=1.81",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "meeko>=0.5.0",
    "gemmi>=0.6.5",
    "openbabel-wheel>=3.1.1",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "pillow>=10.0.0",
    "jupyterlab>=4.2.0",
    "ipywidgets>=8.1.0",
    "nvidia-cuda-runtime-cu12>=12.4",
    "nvidia-cublas-cu12>=12.4",
    "nvidia-cusparse-cu12>=12.3",
    "nvidia-cusolver-cu12>=11.6",
    "nvidia-curand-cu12>=10.3",
    "nvidia-nvjitlink-cu12>=12.4",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("bzip2", "ca-certificates", "curl", "git", "rsync")
    .pip_install(*PYTHON_PACKAGES)
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba",
        "micromamba create -y -p /opt/remedia-fpocket -c conda-forge -c bioconda fpocket",
        "micromamba clean --all --yes",
        "curl --fail --location --retry 3 "
        "https://github.com/gnina/gnina/releases/download/v1.3/gnina "
        "--output /usr/local/bin/gnina",
        "chmod 0755 /usr/local/bin/gnina",
        "git clone --depth 1 https://github.com/mehmetg06/Remedia.git /opt/remedia",
        "/usr/local/bin/gnina --version",
    )
    .env({"GNINA_PATH": "/usr/local/bin/gnina", "PYTHONUNBUFFERED": "1"})
    .entrypoint([])
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=8192,
    volumes={str(VOLUME_PATH): volume},
    timeout=4 * 60 * 60,
    max_containers=1,
    scaledown_window=120,
)
@modal.web_server(8888, startup_timeout=300, requires_proxy_auth=True)
def jupyter():
    """Start JupyterLab while the authenticated Modal URL is in use."""
    REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_PATH / "src").is_dir():
        shutil.copytree("/opt/remedia", REPO_PATH)
    else:
        subprocess.run(
            [
                "rsync", "-a", "--delete",
                "--exclude=.git/", "--exclude=__pycache__/", "--exclude=results/",
                "/opt/remedia/", f"{REPO_PATH}/",
            ],
            check=True,
        )

    (VOLUME_PATH / "Remedia_results").mkdir(parents=True, exist_ok=True)
    (VOLUME_PATH / "remedia_cache").mkdir(parents=True, exist_ok=True)
    volume.commit()

    env = os.environ.copy()
    env["PATH"] = f"/opt/remedia-fpocket/bin:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{REPO_PATH / 'src'}:{env.get('PYTHONPATH', '')}"
    env["GNINA_PATH"] = "/usr/local/bin/gnina"
    env["REMEDIA_HOME"] = str(REPO_PATH)
    env["REMEDIA_WORKSPACE"] = str(VOLUME_PATH)

    command = [
        "jupyter", "lab",
        "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888",
        f"--ServerApp.root_dir={VOLUME_PATH}",
        "--ServerApp.allow_origin=*",
        "--ServerApp.allow_remote_access=True",
        "--ServerApp.default_url=/lab/tree/Remedia/notebooks/remedia_modal.ipynb",
        "--IdentityProvider.token=",
        "--ServerApp.password=",
    ]
    subprocess.Popen(command, env=env)
