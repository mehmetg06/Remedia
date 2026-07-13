"""Single-page Modal web app for Remedia.

Deploy once:
    modal deploy modal/remedia_web.py

The public URL contains only UniProt ID, molecule count and a start button.
REINVENT4 is always used. The GPU worker runs on L4 and stores results in the
persistent ``remedia-data`` Volume.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import re
import shutil
import sys
import traceback
import uuid
from pathlib import Path

import modal

APP_NAME = "remedia-web"
VOLUME_NAME = "remedia-data"
VOLUME_PATH = Path("/workspace")
REPO_PATH = VOLUME_PATH / "Remedia"
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
    REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess = __import__("subprocess")
    subprocess.run(
        [
            "rsync", "-a", "--delete",
            "--exclude=.git/", "--exclude=__pycache__/", "--exclude=results/",
            "/opt/remedia/", f"{REPO_PATH}/",
        ],
        check=True,
    )
    JOBS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def _job_file(job_id: str) -> Path:
    return JOBS_PATH / f"{job_id}.json"


def _write_job(job_id: str, **changes) -> None:
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
    volume.commit()


class _ProgressStream(io.TextIOBase):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        if len(self.buffer) > 12000:
            self.buffer = self.buffer[-12000:]
        low = text.lower()
        if "pocket" in low or "fpocket" in low:
            _write_job(self.job_id, state="running", step=2, message="Pocket bulunuyor")
        elif "reinvent" in low or "sampling" in low or "molekül" in low:
            _write_job(self.job_id, state="running", step=3, message="REINVENT4 molekül üretiyor")
        elif "gnina" in low or "docking" in low:
            _write_job(self.job_id, state="running", step=4, message="GNINA docking yapıyor")
        elif "admet" in low or "sıral" in low or "zip" in low:
            _write_job(self.job_id, state="running", step=5, message="Sonuçlar hazırlanıyor")
        return len(text)

    def flush(self) -> None:
        return None


def _load_notebook_pipeline():
    """Load the existing tested pipeline without exposing Jupyter to the user."""
    notebook_path = REPO_PATH / "notebooks" / "remedia_modal.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    namespace = {"__name__": "remedia_embedded_pipeline"}
    exec(compile(code, str(notebook_path), "exec"), namespace)
    if "_run_pipeline" not in namespace:
        raise RuntimeError("Remedia pipeline fonksiyonu bulunamadı.")
    namespace["PERSISTENT_ROOT"] = str(VOLUME_PATH)
    return namespace["_run_pipeline"]


@app.function(
    image=image,
    gpu="L4",
    volumes={str(VOLUME_PATH): volume},
    timeout=4 * 60 * 60,
    max_containers=1,
)
def run_job(job_id: str, uniprot_id: str, molecule_count: int) -> None:
    _sync_repo()
    _write_job(job_id, state="running", step=1, message="Reseptör hazırlanıyor")

    stream = _ProgressStream(job_id)
    try:
        os.environ["REMEDIA_HOME"] = str(REPO_PATH)
        os.environ["REMEDIA_WORKSPACE"] = str(VOLUME_PATH)
        src = str(REPO_PATH / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

        run_pipeline = _load_notebook_pipeline()
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
        after = [p.resolve() for p in RESULTS_PATH.glob("run_*") if p.is_dir() and p.resolve() not in before]
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
        technical = stream.buffer[-6000:] + "\n" + traceback.format_exc()
        error_path = JOBS_PATH / f"{job_id}.error.txt"
        error_path.write_text(technical)
        message = str(exc).strip() or "Beklenmeyen bir hata oluştu."
        if "REINVENT4 sampling" in message:
            message = "REINVENT4 molekül üretemedi. Tekrar dene; yine olursa teknik kayıt indirilebilir."
        _write_job(
            job_id,
            state="error",
            message=message,
            technical_log=str(error_path.relative_to(VOLUME_PATH)),
        )


HTML = r'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#171717;background:#f5f5f3}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}.card{width:min(560px,100%);background:#fff;border:1px solid #deded8;border-radius:24px;padding:28px;box-shadow:0 18px 55px #00000012}
h1{margin:0 0 6px;font-size:34px}.sub{margin:0 0 26px;color:#666}.field{margin:18px 0}label{display:block;font-weight:700;margin-bottom:8px}input{width:100%;padding:15px;border:1px solid #c9c9c3;border-radius:13px;font-size:18px}
button,a.btn{width:100%;display:block;text-align:center;border:0;border-radius:14px;padding:16px;font-size:17px;font-weight:800;text-decoration:none;cursor:pointer;background:#171717;color:#fff}.muted{color:#74746e;font-size:14px;margin-top:10px}.progress{display:none;margin-top:24px}.bar{height:10px;background:#e9e9e4;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:#171717;transition:width .35s}.step{font-weight:800;margin:14px 0 6px}.error{color:#a32020;background:#fff0f0;padding:14px;border-radius:12px;margin-top:14px}.done{display:none;margin-top:18px}.retry{margin-top:10px;background:#555}.spinner{display:inline-block;width:13px;height:13px;border:2px solid #bbb;border-top-color:#111;border-radius:50%;animation:s .8s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><main class="card"><h1>Remedia</h1><p class="sub">REINVENT4 → GNINA → ADMET</p>
<form id="form"><div class="field"><label for="u">UniProt ID</label><input id="u" name="uniprot_id" value="P00918" autocomplete="off" required pattern="[A-Za-z0-9-]{4,16}"></div>
<div class="field"><label for="n">Molekül sayısı</label><input id="n" name="molecule_count" type="number" value="20" min="5" max="100" step="5" required></div>
<button id="start" type="submit">Remedia’yı Başlat</button><p class="muted">L4 GPU otomatik açılır. Sayfayı kapatsan da işlem sürer.</p></form>
<section id="progress" class="progress"><div class="bar"><div id="fill" class="fill"></div></div><div id="step" class="step"><span class="spinner"></span> Hazırlanıyor</div><div id="message" class="muted"></div><div id="error"></div></section>
<section id="done" class="done"><a id="download" class="btn">Sonuçları indir</a><button class="retry" onclick="location.reload()">Yeni işlem</button></section>
<script>
const form=document.querySelector('#form'), progress=document.querySelector('#progress'), fill=document.querySelector('#fill'), step=document.querySelector('#step'), msg=document.querySelector('#message'), err=document.querySelector('#error'), done=document.querySelector('#done');
form.addEventListener('submit',async e=>{e.preventDefault();document.querySelector('#start').disabled=true;progress.style.display='block';
 const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uniprot_id:document.querySelector('#u').value,molecule_count:Number(document.querySelector('#n').value)})});
 const x=await r.json(); if(!r.ok){showError(x.detail||'Başlatılamadı');return;} poll(x.job_id);});
async function poll(id){try{const r=await fetch('/status/'+id,{cache:'no-store'}),x=await r.json();fill.style.width=((x.step||1)/5*100)+'%';msg.textContent=x.message||'';
 if(x.state==='done'){step.textContent='Tamamlandı';done.style.display='block';document.querySelector('#download').href='/download/'+id;return;}
 if(x.state==='error'){showError(x.message||'İşlem başarısız');return;}step.innerHTML='<span class="spinner"></span> '+(x.step||1)+'/5';setTimeout(()=>poll(id),2500);
}catch(e){msg.textContent='Bağlantı bekleniyor…';setTimeout(()=>poll(id),4000);}}
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
    from pydantic import BaseModel, Field

    api = FastAPI(title="Remedia")

    class StartRequest(BaseModel):
        uniprot_id: str = Field(min_length=4, max_length=16)
        molecule_count: int = Field(default=20, ge=5, le=100)

    @api.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @api.post("/start")
    def start(req: StartRequest):
        uniprot = req.uniprot_id.strip().upper()
        if not re.fullmatch(r"[A-Z0-9-]{4,16}", uniprot):
            raise HTTPException(400, "Geçerli bir UniProt ID gir.")
        job_id = uuid.uuid4().hex
        JOBS_PATH.mkdir(parents=True, exist_ok=True)
        _write_job(job_id, state="queued", step=1, message="GPU sırası bekleniyor", uniprot_id=uniprot, molecule_count=req.molecule_count)
        run_job.spawn(job_id, uniprot, req.molecule_count)
        return {"job_id": job_id}

    @api.get("/status/{job_id}")
    def status(job_id: str):
        volume.reload()
        path = _job_file(job_id)
        if not path.exists():
            raise HTTPException(404, "İş bulunamadı.")
        return json.loads(path.read_text())

    @api.get("/download/{job_id}")
    def download(job_id: str):
        volume.reload()
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
