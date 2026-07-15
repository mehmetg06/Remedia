"""Stable single-page Remedia Modal app.

Deploy:
    modal deploy modal/remedia_web_v2.py

Changes vs the first web version:
- no destructive rsync --delete
- REINVENT4 lives on the persistent Volume at /workspace/REINVENT4
- the worker runs from /workspace so REINVENT is reused on later jobs
- async Modal APIs are used by the web endpoint
- notebook widgets are suppressed when the existing pipeline is loaded
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import modal
from starlette.requests import Request

APP_NAME = "remedia-web"
VOLUME_NAME = "remedia-data"
VOLUME_PATH = Path("/workspace")
REPO_PATH = VOLUME_PATH / "Remedia"
REINVENT_PATH = VOLUME_PATH / "REINVENT4"
JOBS_PATH = VOLUME_PATH / "remedia_web_jobs"
RESULTS_PATH = VOLUME_PATH / "Remedia_results"
REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("bzip2", "ca-certificates", "curl", "git", "rsync")
    .pip_install_from_requirements(str(REPO_ROOT / "modal" / "requirements.txt"))
    .pip_install("fastapi>=0.115", "python-multipart>=0.0.18")
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | "
        "tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba",
        "micromamba create -y -p /opt/remedia-fpocket -c conda-forge -c bioconda fpocket",
        "micromamba clean --all --yes",
        "curl --fail --location --retry 3 "
        "https://github.com/gnina/gnina/releases/download/v1.3/gnina "
        "--output /usr/local/bin/gnina",
        "chmod 0755 /usr/local/bin/gnina",
        "mkdir -p /opt/remedia-nvidia-libs && "
        "find /usr/local/lib/python3.11/site-packages/nvidia -path '*/lib/*.so*' "
        "-type f -exec ln -sf {} /opt/remedia-nvidia-libs/ \\;",
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
            "REMEDIA_WORKSPACE": "/workspace",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/opt/remedia",
        copy=True,
        ignore=[".git", ".venv", "**/__pycache__", "**/*.pyc", "results"],
    )
    .entrypoint([])
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sync_repo() -> None:
    """Copy application code without deleting runtime data."""
    REPO_PATH.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=.git/",
            "--exclude=__pycache__/",
            "--exclude=results/",
            "--exclude=notebooks/REINVENT4/",
            "/opt/remedia/",
            f"{REPO_PATH}/",
        ],
        check=True,
    )
    JOBS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    REINVENT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _job_file(job_id: str) -> Path:
    return JOBS_PATH / f"{job_id}.json"


def _write_job_local(job_id: str, **changes) -> None:
    path = _job_file(job_id)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data.update(changes)
    data["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _write_job(job_id: str, **changes) -> None:
    _write_job_local(job_id, **changes)
    volume.commit()


class _ProgressStream(io.TextIOBase):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        if len(self.buffer) > 14000:
            self.buffer = self.buffer[-14000:]
        low = text.lower()
        if "pocket" in low or "fpocket" in low:
            _write_job(self.job_id, state="running", step=2, message="Pocket bulunuyor")
        elif "reinvent" in low or "sampling" in low or "prior" in low:
            _write_job(self.job_id, state="running", step=3, message="REINVENT4 hazırlanıyor ve molekül üretiyor")
        elif "gnina" in low or "docking" in low:
            _write_job(self.job_id, state="running", step=4, message="GNINA docking yapıyor")
        elif "admet" in low or "sıral" in low or "zip" in low:
            _write_job(self.job_id, state="running", step=5, message="Sonuçlar hazırlanıyor")
        return len(text)

    def flush(self) -> None:
        return None


def _load_pipeline():
    """Load the current tested notebook pipeline while hiding widget output."""
    notebook_path = REPO_PATH / "notebooks" / "remedia_modal.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    import IPython.display as ipd

    original_display = ipd.display
    ipd.display = lambda *args, **kwargs: None
    try:
        namespace = {"__name__": "remedia_embedded_pipeline"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, str(notebook_path), "exec"), namespace)
    finally:
        ipd.display = original_display

    if "_run_pipeline" not in namespace:
        raise RuntimeError("Remedia pipeline fonksiyonu bulunamadı.")
    namespace["PERSISTENT_ROOT"] = str(VOLUME_PATH)
    return namespace["_run_pipeline"]


def _prepare_reinvent_location() -> None:
    """Keep REINVENT outside the repo and on the persistent Volume."""
    old_candidates = [
        REPO_PATH / "notebooks" / "REINVENT4",
        Path("/root/REINVENT4"),
        Path("/root/remedia_workspace/Remedia/notebooks/REINVENT4"),
    ]
    if not REINVENT_PATH.exists():
        for old in old_candidates:
            if old.is_dir():
                try:
                    shutil.move(str(old), str(REINVENT_PATH))
                    break
                except Exception:
                    pass


@app.function(
    image=image,
    gpu="L4",
    volumes={str(VOLUME_PATH): volume},
    timeout=4 * 60 * 60,
    max_containers=1,
)
def run_job(job_id: str, uniprot_id: str, molecule_count: int) -> None:
    _sync_repo()
    _prepare_reinvent_location()
    volume.commit()
    _write_job(job_id, state="running", step=1, message="Reseptör hazırlanıyor")

    stream = _ProgressStream(job_id)
    original_cwd = Path.cwd()
    try:
        os.chdir(VOLUME_PATH)
        os.environ["REMEDIA_HOME"] = str(REPO_PATH)
        os.environ["REMEDIA_WORKSPACE"] = str(VOLUME_PATH)
        os.environ["GNINA_PATH"] = "/usr/local/bin/gnina"
        src = str(REPO_PATH / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

        run_pipeline = _load_pipeline()
        settings = {
            "uniprot_id": uniprot_id,
            "method": "pretrained",
            "generate_n": molecule_count,
            "profile": "balanced",
            "docking_mode": "iki_asamali",
            "box_dim": 20,
            "top_fraction": 0.10,
            "ga_generations": 3,
            "run_benchmark": False,
            "force_refresh": False,
            "install_reinvent": True,
        }

        before = {p.resolve() for p in RESULTS_PATH.glob("run_*") if p.is_dir()}
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = run_pipeline(settings)

        after = [
            p.resolve()
            for p in RESULTS_PATH.glob("run_*")
            if p.is_dir() and p.resolve() not in before
        ]
        result_dir = max(after, key=lambda p: p.stat().st_mtime) if after else None

        if isinstance(result, (str, Path)) and Path(result).exists():
            candidate = Path(result)
            result_dir = candidate if candidate.is_dir() else candidate.parent
        if result_dir is None:
            candidates = list(RESULTS_PATH.glob("run_*"))
            result_dir = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        if result_dir is None:
            raise RuntimeError("İşlem tamamlandı ancak sonuç klasörü bulunamadı.")

        zip_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
        _write_job(
            job_id,
            state="done",
            step=5,
            message="Tamamlandı",
            result_zip=str(Path(zip_path).relative_to(VOLUME_PATH)),
        )
    except Exception as exc:
        technical = stream.buffer[-8000:] + "\n" + traceback.format_exc()
        error_path = JOBS_PATH / f"{job_id}.error.txt"
        error_path.write_text(technical)
        message = str(exc).strip() or "Beklenmeyen bir hata oluştu."
        if "REINVENT4 sampling" in message:
            message = "REINVENT4 molekül üretemedi. Teknik kayıt kaydedildi."
        _write_job(
            job_id,
            state="error",
            message=message,
            technical_log=str(error_path.relative_to(VOLUME_PATH)),
        )
    finally:
        os.chdir(original_cwd)


HTML = r'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#171717;background:#f5f5f3}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}.card{width:min(560px,100%);background:#fff;border:1px solid #deded8;border-radius:24px;padding:28px;box-shadow:0 18px 55px #00000012}
h1{margin:0 0 6px;font-size:34px}.sub{margin:0 0 26px;color:#666}.field{margin:18px 0}label{display:block;font-weight:700;margin-bottom:8px}input{width:100%;padding:15px;border:1px solid #c9c9c3;border-radius:13px;font-size:18px}
button,a.btn{width:100%;display:block;text-align:center;border:0;border-radius:14px;padding:16px;font-size:17px;font-weight:800;text-decoration:none;cursor:pointer;background:#171717;color:#fff}.muted{color:#74746e;font-size:14px;margin-top:10px}.progress{display:none;margin-top:24px}.bar{height:10px;background:#e9e9e4;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:#171717;transition:width .35s}.step{font-weight:800;margin:14px 0 6px}.error{color:#a32020;background:#fff0f0;padding:14px;border-radius:12px;margin-top:14px}.done{display:none;margin-top:18px}.retry{margin-top:10px;background:#555}.spinner{display:inline-block;width:13px;height:13px;border:2px solid #bbb;border-top-color:#111;border-radius:50%;animation:s .8s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><main class="card"><h1>Remedia</h1><p class="sub">REINVENT4 → GNINA → ADMET</p>
<form id="form"><div class="field"><label for="u">UniProt ID</label><input id="u" value="P00918" autocomplete="off" required pattern="[A-Za-z0-9-]{4,16}"></div>
<div class="field"><label for="n">Molekül sayısı</label><input id="n" type="number" value="20" min="5" max="100" step="5" required></div>
<button id="start" type="submit">Remedia’yı Başlat</button><p class="muted">İlk REINVENT çalıştırması birkaç dakika sürebilir; sonraki çalıştırmalarda kurulum tekrar kullanılacaktır.</p></form>
<section id="progress" class="progress"><div class="bar"><div id="fill" class="fill"></div></div><div id="step" class="step"><span class="spinner"></span> Hazırlanıyor</div><div id="message" class="muted"></div><div id="error"></div></section>
<section id="done" class="done"><a id="download" class="btn">Sonuçları indir</a><button class="retry" onclick="location.reload()">Yeni işlem</button></section>
<script>
const form=document.querySelector('#form'),progress=document.querySelector('#progress'),fill=document.querySelector('#fill'),step=document.querySelector('#step'),msg=document.querySelector('#message'),err=document.querySelector('#error'),done=document.querySelector('#done');
form.addEventListener('submit',async e=>{e.preventDefault();document.querySelector('#start').disabled=true;progress.style.display='block';const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uniprot_id:document.querySelector('#u').value,molecule_count:Number(document.querySelector('#n').value)})});const x=await r.json();if(!r.ok){showError(x.detail||'Başlatılamadı');return;}poll(x.job_id);});
async function poll(id){try{const r=await fetch('/status/'+id,{cache:'no-store'}),x=await r.json();fill.style.width=((x.step||1)/5*100)+'%';msg.textContent=x.message||'';if(x.state==='done'){step.textContent='Tamamlandı';done.style.display='block';document.querySelector('#download').href='/download/'+id;return;}if(x.state==='error'){showError(x.message||'İşlem başarısız');return;}step.innerHTML='<span class="spinner"></span> '+(x.step||1)+'/5';setTimeout(()=>poll(id),2500);}catch(e){msg.textContent='Bağlantı bekleniyor…';setTimeout(()=>poll(id),4000);}}
function showError(t){step.textContent='İşlem durdu';err.className='error';err.innerHTML=t+'<button class="retry" onclick="location.reload()">Tekrar dene</button>';}
</script></main></body></html>'''


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    timeout=15 * 60,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    api = FastAPI(title="Remedia")

    @api.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @api.post("/start")
    async def start(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        uniprot = str(payload.get("uniprot_id", "")).strip().upper()
        try:
            molecule_count = int(payload.get("molecule_count", 20))
        except (TypeError, ValueError):
            raise HTTPException(400, "Molekül sayısı sayı olmalı.")

        if not re.fullmatch(r"[A-Z0-9-]{4,16}", uniprot):
            raise HTTPException(400, "Geçerli bir UniProt ID gir.")
        if not 5 <= molecule_count <= 100:
            raise HTTPException(400, "Molekül sayısı 5 ile 100 arasında olmalı.")

        job_id = uuid.uuid4().hex
        JOBS_PATH.mkdir(parents=True, exist_ok=True)
        _write_job_local(
            job_id,
            state="queued",
            step=1,
            message="GPU sırası bekleniyor",
            uniprot_id=uniprot,
            molecule_count=molecule_count,
        )
        await volume.commit.aio()
        await run_job.spawn.aio(job_id, uniprot, molecule_count)
        return {"job_id": job_id}

    @api.get("/status/{job_id}")
    async def status(job_id: str):
        await volume.reload.aio()
        path = _job_file(job_id)
        if not path.exists():
            raise HTTPException(404, "İş bulunamadı.")
        return json.loads(path.read_text())

    @api.get("/download/{job_id}")
    async def download(job_id: str):
        await volume.reload.aio()
        path = _job_file(job_id)
        if not path.exists():
            raise HTTPException(404, "İş bulunamadı.")
        data = json.loads(path.read_text())
        rel = data.get("result_zip")
        if data.get("state") != "done" or not rel:
            raise HTTPException(409, "Sonuç henüz hazır değil.")
        file_path = VOLUME_PATH / rel
        if not file_path.exists():
            raise HTTPException(404, "Sonuç dosyası bulunamadı.")
        return FileResponse(file_path, filename=file_path.name, media_type="application/zip")

    return api
