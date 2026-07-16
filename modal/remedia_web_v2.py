"""Stable single-page Remedia Modal app.

Deploy:
    modal deploy modal/remedia_web_v2.py
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
import time
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
    .apt_install(
        "bzip2",
        "ca-certificates",
        "curl",
        "git",
        "rsync",
        "libxrender1",
        "libxext6",
        "libsm6",
    )
    .pip_install_from_requirements(str(REPO_ROOT / "modal" / "requirements.txt"))
    .pip_install("fastapi>=0.115", "python-multipart>=0.0.18")
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | "
        "tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba",
        "micromamba create -y -p /opt/remedia-fpocket -c conda-forge -c bioconda fpocket",
        "micromamba clean --all --yes",
        "curl --fail --location --retry 3 "
        "https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8 "
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
            "GNINA_CPU": "8",
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


def _artifact_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = VOLUME_PATH / path
    path = path.resolve()
    root = VOLUME_PATH.resolve()
    if path != root and root not in path.parents:
        return None
    return path


#: Prefix marking a machine-readable structured progress line (see src/progress.py).
PROGRESS_SENTINEL = "[[REMEDIA_PROGRESS]]"


class _ProgressStream(io.TextIOBase):
    """Capture pipeline logs without stalling GNINA on frequent Volume commits.

    Phase 2: when the pipeline emits structured progress events (lines prefixed
    with :data:`PROGRESS_SENTINEL`), those precise stage/task/item counts drive
    the UI.  For any line without a sentinel the legacy heuristic scraping below
    remains as a fallback, so older pipeline code keeps reporting progress.
    """

    COMMIT_INTERVAL_SECONDS = 3.0

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = ""
        self.percent = 8
        self.last_committed_percent = 8
        self.last_committed_step = 1
        self.last_commit_at = 0.0
        self.gnina_phase = ""
        self.gnina_phase_started_at = time.monotonic()

    def _advance(
        self,
        percent: int,
        message: str,
        step: int,
        *,
        force: bool = False,
    ) -> None:
        target = max(self.percent, min(percent, 96))
        now = time.monotonic()
        significant = step != self.last_committed_step or target >= self.last_committed_percent + 3
        due = now - self.last_commit_at >= self.COMMIT_INTERVAL_SECONDS
        self.percent = target
        if not (force or significant or due):
            return
        _write_job(
            self.job_id,
            state="running",
            step=step,
            progress_percent=self.percent,
            message=message,
        )
        self.last_commit_at = now
        self.last_committed_percent = self.percent
        self.last_committed_step = step

    def _start_gnina_phase(self, phase: str, percent: int, message: str) -> None:
        self.gnina_phase = phase
        self.gnina_phase_started_at = time.monotonic()
        self._advance(percent, message, 4, force=True)

    def _gnina_percent(self) -> int:
        elapsed = max(0.0, time.monotonic() - self.gnina_phase_started_at)
        if self.gnina_phase == "fast":
            return min(76, 58 + int(elapsed / 7.0))
        if self.gnina_phase == "accurate":
            return min(90, 78 + int(elapsed / 10.0))
        return max(self.percent, 54)

    def _structured(self, event: dict) -> None:
        """Commit a structured progress event straight to the job file."""
        percent = event.get("percent")
        try:
            percent = int(round(float(percent)))
        except (TypeError, ValueError):
            percent = self.percent
        # Structured events carry the authoritative percent; still never regress.
        self.percent = max(self.percent, min(percent, 99))
        step = int(event.get("step", self.last_committed_step) or 1)
        label = event.get("stage_label") or event.get("task") or ""
        done, total = event.get("items_done"), event.get("items_total")
        if total:
            message = f"{label} ({done or 0}/{total})"
        else:
            message = event.get("message") or label
        _write_job(
            self.job_id,
            state="running",
            step=step,
            progress_percent=self.percent,
            message=message,
            stage=event.get("stage"),
            stage_label=label,
            task=event.get("task"),
            items_done=done,
            items_total=total,
            eta_seconds=event.get("eta_seconds"),
        )
        self.last_commit_at = time.monotonic()
        self.last_committed_percent = self.percent
        self.last_committed_step = step

    def write(self, text: str) -> int:
        self.buffer += text
        if len(self.buffer) > 50000:
            self.buffer = self.buffer[-50000:]

        # Structured events (Phase 2) take priority over heuristic scraping.
        if PROGRESS_SENTINEL in text:
            handled = False
            for line in text.splitlines():
                idx = line.find(PROGRESS_SENTINEL)
                if idx < 0:
                    continue
                payload = line[idx + len(PROGRESS_SENTINEL):].strip()
                try:
                    event = json.loads(payload)
                except (ValueError, TypeError):
                    continue
                if isinstance(event, dict) and event.get("schema") == "remedia.progress/1":
                    self._structured(event)
                    handled = True
            if handled:
                return len(text)

        low = text.lower()
        clean = text.strip()

        if "pocket" in low or "fpocket" in low:
            self._advance(18, "Bağlanma cebi belirleniyor", 2)
        elif "reinvent" in low or "sampling" in low or "prior" in low:
            self._advance(36, clean[-180:] if clean else "REINVENT4 molekül üretiyor", 3)
        elif "[1/2]" in low or "gnina] fast" in low or "fast batch" in low:
            self._start_gnina_phase("fast", 58, "GNINA hızlı tarama yapıyor")
        elif "[2/2]" in low or "gnina] accurate" in low or "accurate batch" in low:
            self._start_gnina_phase("accurate", 78, "GNINA ayrıntılı doğrulama yapıyor")
        elif "gnina" in low or "docking" in low:
            message = clean[-180:] if "tamamlandı" in low and clean else "GNINA docking yapıyor"
            self._advance(self._gnina_percent(), message, 4)
        elif "admet" in low or "sıral" in low or "zip" in low:
            self._advance(92, "ADMET ve sonuç dosyaları hazırlanıyor", 5, force=True)
        return len(text)

    def flush(self) -> None:
        return None


def _load_pipeline():
    """Load the current notebook pipeline while suppressing notebook widgets."""
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
    cpu=8.0,
    volumes={str(VOLUME_PATH): volume},
    timeout=4 * 60 * 60,
    max_containers=1,
)
def run_job(
    job_id: str,
    uniprot_id: str,
    molecule_count: int,
    generator: str = "reinvent4",
    pose_engine: str = "gnina",
) -> None:
    _sync_repo()
    _prepare_reinvent_location()
    volume.commit()
    _write_job(
        job_id,
        state="running",
        step=1,
        progress_percent=8,
        message="Reseptör hazırlanıyor",
    )

    stream = _ProgressStream(job_id)
    original_cwd = Path.cwd()
    try:
        os.chdir(VOLUME_PATH)
        os.environ["REMEDIA_HOME"] = str(REPO_PATH)
        os.environ["REMEDIA_WORKSPACE"] = str(VOLUME_PATH)
        os.environ["GNINA_PATH"] = "/usr/local/bin/gnina"
        os.environ["GNINA_CPU"] = "8"
        src = str(REPO_PATH / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

        run_pipeline = _load_pipeline()
        settings = {
            "uniprot_id": uniprot_id,
            "method": "pretrained",
            "generator": generator,
            "pose_engine": pose_engine,
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

        stream._advance(94, "Açıklamalı sonuç raporu hazırlanıyor", 5, force=True)
        from result_report import build_result_package

        report_info = build_result_package(
            result_dir,
            target_uniprot=uniprot_id,
            requested_molecules=molecule_count,
            settings=settings,
            pipeline_log=stream.buffer,
            job_id=job_id,
        )

        zip_path = Path(shutil.make_archive(str(result_dir), "zip", root_dir=result_dir))
        candidate_count = int(report_info.get("candidate_count", 0))
        scored_count = int(report_info.get("scored_candidate_count", 0))
        _write_job(
            job_id,
            state="done",
            step=5,
            progress_percent=100,
            message=f"Tamamlandı · {candidate_count} aday, {scored_count} docking skoru",
            result_zip=str(zip_path.resolve()),
            report_file=str(Path(report_info["report_path"]).resolve()),
            report_available=True,
            candidate_count=candidate_count,
            scored_candidate_count=scored_count,
            primary_candidate_table=report_info.get("primary_candidate_table"),
        )
    except Exception as exc:
        technical = stream.buffer[-12000:] + "\n" + traceback.format_exc()
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
            technical_excerpt=technical[-3500:],
        )
    finally:
        os.chdir(original_cwd)


HTML = r'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#171717;background:#f5f5f3}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}.card{width:min(590px,100%);background:#fff;border:1px solid #deded8;border-radius:24px;padding:28px;box-shadow:0 18px 55px #00000012}
h1{margin:0 0 6px;font-size:34px}.sub{margin:0 0 26px;color:#666}.field{margin:18px 0}label{display:block;font-weight:700;margin-bottom:8px}input{width:100%;padding:15px;border:1px solid #c9c9c3;border-radius:13px;font-size:18px}input[type=radio]{width:auto;padding:0}
.choices{display:flex;gap:8px;flex-wrap:wrap}.choice{flex:1;min-width:120px;display:flex;align-items:center;gap:8px;padding:12px 14px;border:1px solid #c9c9c3;border-radius:13px;font-weight:600;cursor:pointer}.choice:has(input:checked){border-color:#171717;background:#f0f0ec}
button,a.btn{width:100%;display:block;text-align:center;border:0;border-radius:14px;padding:16px;font-size:17px;font-weight:800;text-decoration:none;cursor:pointer;background:#171717;color:#fff}.btn.secondary{background:#ecece8;color:#171717;margin-bottom:10px}.muted{color:#74746e;font-size:14px;margin-top:10px}.progress{display:none;margin-top:24px}.progress-head{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:10px}.progress-title{font-weight:800}.percent{font-variant-numeric:tabular-nums;font-weight:800;color:#555}.bar{height:14px;background:#e9e9e4;border-radius:99px;overflow:hidden;position:relative}.fill{height:100%;width:0;background:linear-gradient(90deg,#111,#4d4d4d);transition:width .65s cubic-bezier(.2,.8,.2,1);position:relative}.fill:after{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 25%,#ffffff55 45%,transparent 65%);animation:shine 1.6s linear infinite}@keyframes shine{from{transform:translateX(-100%)}to{transform:translateX(100%)}}.stages{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-top:9px}.stage{height:4px;border-radius:99px;background:#e9e9e4}.stage.active{background:#555}.step{font-weight:800;margin:14px 0 6px}.error{color:#a32020;background:#fff0f0;padding:14px;border-radius:12px;margin-top:14px;white-space:pre-wrap;overflow-wrap:anywhere}.done{display:none;margin-top:18px}.done-note{padding:13px;background:#f6f6f3;border-radius:12px;margin-bottom:12px;color:#555}.retry{margin-top:10px;background:#555}.spinner{display:inline-block;width:13px;height:13px;border:2px solid #bbb;border-top-color:#111;border-radius:50%;animation:s .8s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><main class="card"><h1>Remedia</h1><p class="sub">REINVENT4 → GNINA → ADMET</p>
<form id="form"><div class="field"><label for="u">UniProt ID</label><input id="u" value="P00918" autocomplete="off" required pattern="[A-Za-z0-9-]{4,16}"></div>
<div class="field"><label for="n">Molekül sayısı</label><input id="n" type="number" value="20" min="5" max="100" step="5" required></div>
<div class="field"><label>Üretici (Generator)</label><div class="choices">
<label class="choice"><input type="radio" name="gen" value="reinvent4" checked> REINVENT4</label>
<label class="choice"><input type="radio" name="gen" value="molmim"> MolMIM</label>
<label class="choice"><input type="radio" name="gen" value="hybrid"> Hybrid</label>
</div></div>
<div class="field"><label>Poz motoru (Pose Engine)</label><div class="choices">
<label class="choice"><input type="radio" name="pose" value="gnina" checked> GNINA</label>
<label class="choice"><input type="radio" name="pose" value="diffdock"> DiffDock</label>
<label class="choice"><input type="radio" name="pose" value="hybrid"> Hybrid Validation</label>
</div></div>
<button id="start" type="submit">Remedia’yı Başlat</button><p class="muted">Sonuç paketi açıklamalı HTML raporu, tekleştirilmiş aday tablosu, veri sözlüğü, çalışma ayarları ve teknik log içerecektir.</p></form>
<section id="progress" class="progress"><div class="progress-head"><div class="progress-title">İşlem ilerliyor</div><div id="percent" class="percent">0%</div></div><div class="bar"><div id="fill" class="fill"></div></div><div class="stages"><div class="stage"></div><div class="stage"></div><div class="stage"></div><div class="stage"></div><div class="stage"></div></div><div id="step" class="step"><span class="spinner"></span> Hazırlanıyor</div><div id="message" class="muted"></div><div id="error"></div></section>
<section id="done" class="done"><div id="doneNote" class="done-note">Açıklamalı rapor ve ham veriler hazır.</div><a id="report" class="btn secondary" target="_blank">Raporu tarayıcıda aç</a><a id="download" class="btn">Tam sonuç paketini indir</a><button class="retry" onclick="location.reload()">Yeni işlem</button></section>
<script>
const form=document.querySelector('#form'),progress=document.querySelector('#progress'),fill=document.querySelector('#fill'),step=document.querySelector('#step'),msg=document.querySelector('#message'),err=document.querySelector('#error'),done=document.querySelector('#done'),pct=document.querySelector('#percent'),stages=[...document.querySelectorAll('.stage')];
form.addEventListener('submit',async e=>{e.preventDefault();document.querySelector('#start').disabled=true;progress.style.display='block';const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uniprot_id:document.querySelector('#u').value,molecule_count:Number(document.querySelector('#n').value),generator:document.querySelector('input[name=gen]:checked').value,pose_engine:document.querySelector('input[name=pose]:checked').value})});const x=await r.json();if(!r.ok){showError(x.detail||'Başlatılamadı');return;}poll(x.job_id);});
function paint(x){const p=Math.max(0,Math.min(100,Number(x.progress_percent??((x.step||1)/5*100))));fill.style.width=p+'%';pct.textContent=Math.round(p)+'%';stages.forEach((el,i)=>el.classList.toggle('active',i<(x.step||1)));}
function detail(x){let m=x.message||'';if(x.items_total){m=(x.stage_label||x.task||m)+' ('+(x.items_done||0)+'/'+x.items_total+')';}if(x.eta_seconds){m+=' · ~'+Math.round(x.eta_seconds)+'s kaldı';}return m;}
async function poll(id){try{const r=await fetch('/status/'+id,{cache:'no-store'}),x=await r.json();paint(x);msg.textContent=detail(x);if(x.state==='done'){step.textContent='Tamamlandı';done.style.display='block';document.querySelector('#download').href='/download/'+id;document.querySelector('#report').href='/report/'+id;document.querySelector('#doneNote').textContent=`${x.candidate_count??0} aday bulundu; ${x.scored_candidate_count??0} aday için docking skoru okundu.`;return;}if(x.state==='error'){showError((x.message||'İşlem başarısız')+(x.technical_excerpt?'\n\nTeknik ayrıntı:\n'+x.technical_excerpt:''));return;}const title=x.stage_label||((x.step||1)+'/5');step.innerHTML='<span class="spinner"></span> '+title;setTimeout(()=>poll(id),1800);}catch(e){msg.textContent='Bağlantı bekleniyor…';setTimeout(()=>poll(id),3500);}}
function showError(t){step.textContent='İşlem durdu';err.className='error';err.textContent=t;const b=document.createElement('button');b.className='retry';b.textContent='Tekrar dene';b.onclick=()=>location.reload();err.appendChild(b);}
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

        generator = str(payload.get("generator", "reinvent4")).strip().lower()
        if generator not in {"reinvent4", "molmim", "hybrid"}:
            generator = "reinvent4"
        pose_engine = str(payload.get("pose_engine", "gnina")).strip().lower()
        if pose_engine not in {"gnina", "diffdock", "hybrid"}:
            pose_engine = "gnina"

        job_id = uuid.uuid4().hex
        JOBS_PATH.mkdir(parents=True, exist_ok=True)
        _write_job_local(
            job_id,
            state="queued",
            step=1,
            progress_percent=3,
            message="GPU sırası bekleniyor",
            uniprot_id=uniprot,
            molecule_count=molecule_count,
            generator=generator,
            pose_engine=pose_engine,
        )
        await volume.commit.aio()
        await run_job.spawn.aio(job_id, uniprot, molecule_count, generator, pose_engine)
        return {"job_id": job_id}

    async def _read_job(job_id: str) -> dict:
        await volume.reload.aio()
        path = _job_file(job_id)
        if not path.exists():
            raise HTTPException(404, "İş bulunamadı.")
        return json.loads(path.read_text())

    @api.get("/status/{job_id}")
    async def status(job_id: str):
        return await _read_job(job_id)

    @api.get("/download/{job_id}")
    async def download(job_id: str):
        data = await _read_job(job_id)
        file_path = _artifact_path(data.get("result_zip"))
        if data.get("state") != "done" or file_path is None:
            raise HTTPException(409, "Sonuç henüz hazır değil.")
        if not file_path.exists():
            raise HTTPException(404, "Sonuç dosyası bulunamadı.")
        return FileResponse(file_path, filename=file_path.name, media_type="application/zip")

    @api.get("/report/{job_id}", response_class=HTMLResponse)
    async def report(job_id: str):
        data = await _read_job(job_id)
        report_path = _artifact_path(data.get("report_file"))
        if data.get("state") != "done" or report_path is None:
            raise HTTPException(409, "Rapor henüz hazır değil.")
        if not report_path.is_file():
            raise HTTPException(404, "Rapor dosyası bulunamadı.")
        report_html = report_path.read_text(encoding="utf-8")
        report_html = report_html.replace(
            "src='top_molecules.png'",
            f"src='/report-asset/{job_id}/top_molecules.png'",
        )
        return HTMLResponse(report_html)

    @api.get("/report-asset/{job_id}/{filename}")
    async def report_asset(job_id: str, filename: str):
        if Path(filename).name != filename:
            raise HTTPException(400, "Geçersiz dosya adı.")
        data = await _read_job(job_id)
        report_path = _artifact_path(data.get("report_file"))
        if report_path is None:
            raise HTTPException(404, "Rapor bulunamadı.")
        asset = report_path.parent / filename
        if not asset.is_file():
            raise HTTPException(404, "Rapor görseli bulunamadı.")
        return FileResponse(asset)

    return api
