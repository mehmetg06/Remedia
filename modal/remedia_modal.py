"""Modal GPU/Jupyter launcher and reusable notebook image for Remedia.

One-time image registration:
    python -m pip install modal
    python -m modal setup
    modal deploy modal/remedia_modal.py

After deployment, select the ``remedia-modal / notebook_image`` image in a
hosted Modal Notebook. No web endpoint or continuously running container is
created by the deployment.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import modal

APP_NAME = "remedia-modal"
VOLUME_NAME = "remedia-data"
VOLUME_PATH = Path("/workspace")
REPO_PATH = VOLUME_PATH / "Remedia"
NOTEBOOK_PATH = REPO_PATH / "notebooks" / "remedia_modal_launcher.ipynb"
MAX_SESSION_MINUTES = 240

ALLOWED_GPUS = {
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "H100",
}
GPU = os.environ.get("REMEDIA_MODAL_GPU", "L4").upper()
if GPU not in ALLOWED_GPUS:
    allowed = ", ".join(sorted(ALLOWED_GPUS))
    raise ValueError(f"Unsupported REMEDIA_MODAL_GPU={GPU!r}. Choose one of: {allowed}")

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "modal" / "requirements.txt"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "bzip2",
        "ca-certificates",
        "curl",
        "git",
        "rsync",
    )
    .pip_install_from_requirements(str(REQUIREMENTS))
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba",
        "micromamba create -y -p /opt/remedia-fpocket "
        "-c conda-forge -c bioconda fpocket",
        "micromamba clean --all --yes",
        "curl --fail --location --retry 3 "
        "https://github.com/gnina/gnina/releases/download/v1.3/gnina "
        "--output /usr/local/bin/gnina",
        "chmod 0755 /usr/local/bin/gnina",
        "mkdir -p /opt/remedia-nvidia-libs && "
        "find /usr/local/lib/python3.11/site-packages/nvidia "
        "-path '*/lib/*.so*' -type f "
        "-exec ln -sf {} /opt/remedia-nvidia-libs/ \\;",
        "test -e /opt/remedia-nvidia-libs/libcusparse.so.12",
        "test -e /opt/remedia-nvidia-libs/libnvToolsExt.so.1",
        "LD_LIBRARY_PATH=/opt/remedia-nvidia-libs /usr/local/bin/gnina --version",
    )
    .env(
        {
            "GNINA_PATH": "/usr/local/bin/gnina",
            "LD_LIBRARY_PATH": "/opt/remedia-nvidia-libs",
            "PATH": "/opt/remedia-fpocket/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
            "REMEDIA_PREBUILT_IMAGE": "1",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/opt/remedia",
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "**/__pycache__",
            "**/*.pyc",
            "results",
        ],
    )
    .entrypoint([])
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sync_repo(refresh_code: bool) -> None:
    REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refresh_code or not (REPO_PATH / "src").is_dir():
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                "--exclude=.git/",
                "--exclude=__pycache__/",
                "--exclude=results/",
                "/opt/remedia/",
                f"{REPO_PATH}/",
            ],
            check=True,
        )

    (VOLUME_PATH / "Remedia_results").mkdir(parents=True, exist_ok=True)
    (VOLUME_PATH / "remedia_cache").mkdir(parents=True, exist_ok=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_PATH): volume},
    timeout=MAX_SESSION_MINUTES * 60,
    max_containers=1,
)
def run_jupyter(timeout_minutes: int = 60, refresh_code: bool = False) -> None:
    """Start a temporary token-protected JupyterLab session on a Modal GPU."""

    timeout_minutes = max(15, min(int(timeout_minutes), MAX_SESSION_MINUTES))
    _sync_repo(refresh_code=refresh_code)
    volume.commit()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_PATH / 'src'}:{env.get('PYTHONPATH', '')}"
    env["REMEDIA_HOME"] = str(REPO_PATH)
    env["REMEDIA_WORKSPACE"] = str(VOLUME_PATH)

    token = secrets.token_urlsafe(24)
    port = 8888
    command = [
        "jupyter",
        "lab",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={port}",
        f"--ServerApp.root_dir={VOLUME_PATH}",
        "--ServerApp.allow_origin=*",
        "--ServerApp.allow_remote_access=True",
        "--ServerApp.default_url=/lab/tree/Remedia/notebooks/remedia_modal_launcher.ipynb",
        f"--IdentityProvider.token={token}",
    ]

    with modal.forward(port) as tunnel:
        process = subprocess.Popen(command, env=env)
        direct_url = (
            f"{tunnel.url.rstrip('/')}"
            "/lab/tree/Remedia/notebooks/remedia_modal_launcher.ipynb"
            f"?token={quote(token)}"
        )
        print("=" * 72)
        print("Remedia Modal Jupyter hazır.")
        print(f"GPU: {GPU}")
        print(f"Otomatik kapanma: {timeout_minutes} dakika")
        print(f"Aç: {direct_url}")
        print("=" * 72)

        deadline = time.time() + timeout_minutes * 60
        try:
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Jupyter beklenmedik biçimde kapandı: {process.returncode}"
                    )
                time.sleep(5)
            print("Süre sınırına ulaşıldı; Jupyter kapatılıyor.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_PATH): volume},
    timeout=5 * 60,
)
def notebook_image() -> dict[str, str]:
    """Register the prebuilt image for hosted Modal Notebooks."""

    _sync_repo(refresh_code=False)
    volume.commit()
    return {
        "status": "ready",
        "gpu": GPU,
        "notebook": str(NOTEBOOK_PATH),
        "volume": VOLUME_NAME,
        "prebuilt": os.environ.get("REMEDIA_PREBUILT_IMAGE", "0"),
    }


@app.local_entrypoint()
def main(timeout_minutes: int = 60, refresh_code: bool = False) -> None:
    """Launch temporary Jupyter and keep the local command attached."""

    run_jupyter.remote(
        timeout_minutes=timeout_minutes,
        refresh_code=refresh_code,
    )
