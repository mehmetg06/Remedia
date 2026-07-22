"""Stable single-page Remedia Modal app with an auto-updating deploy.

Deploy once:
    modal deploy modal/remedia_web_v2.py

After that, pushing to the tracked GitHub branch (``REMEDIA_GIT_BRANCH``,
default ``main``) updates the live URL automatically — the worker pulls the
newest code on every run and a page refresh triggers a throttled pull, so there
is no need to redeploy or re-open a Modal notebook. Set ``REMEDIA_GIT_URL`` /
``REMEDIA_GIT_BRANCH`` to point at a different repo or a pinned branch.

The form's "Hız" control selects the GNINA staging mode: ``hizli`` runs a single
fast pass, ``dengeli`` runs the two-stage fast→accurate screening.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
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

# ---------------------------------------------------------------------------
# Auto-updating deploy.
#
# Deploy once with ``modal deploy modal/remedia_web_v2.py``. After that every
# job run and every page load pulls the newest code from GitHub, so pushing to
# the tracked branch is enough to update the live app — there is no need to
# redeploy or touch a Modal notebook again. The tracked branch is configurable
# via env so the URL can be pinned to a stable branch instead of ``main``.
# ---------------------------------------------------------------------------
GIT_URL = os.environ.get("REMEDIA_GIT_URL", "https://github.com/mehmetg06/Remedia.git")
GIT_BRANCH = os.environ.get("REMEDIA_GIT_BRANCH", "main")
GIT_STAMP = VOLUME_PATH / ".remedia_git_stamp"
GIT_LOCK = VOLUME_PATH / ".remedia_git.lock"
GIT_PULL_THROTTLE_SECONDS = 20

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
            "BOLTZ_CACHE": "/workspace/boltz_cache",
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


def _git(args: list[str], cwd: Path | None = None, timeout: int = 240):
    return subprocess.run(
        ["git", *args],
        cwd=(str(cwd) if cwd else None),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _repo_head() -> str | None:
    res = _git(["rev-parse", "--short", "HEAD"], cwd=REPO_PATH)
    return res.stdout.strip() if res.returncode == 0 else None


def _rsync_baked() -> None:
    """Offline fallback: copy the code baked into the image at deploy time.

    Used only when GitHub is unreachable, so the app still starts on a fresh
    volume even without network access to the repository.
    """
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
        check=False,
    )


def _git_sync(force: bool = False) -> str | None:
    """Bring ``/workspace/Remedia`` to the latest ``GIT_BRANCH`` commit.

    This is what removes the redeploy loop: the code is pulled from GitHub at
    run time instead of being frozen into the image at deploy time. Runtime data
    (``Remedia_results/``, ``remedia_cache/``, ``remedia_web_jobs/``,
    ``REINVENT4/``, ``boltz_cache/``) lives *outside* ``REPO_PATH``, so a hard
    reset of the checkout never destroys results. Falls back to the image-baked
    copy when GitHub cannot be reached.
    """
    VOLUME_PATH.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    REINVENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _throttled() -> bool:
        if force or not (REPO_PATH / "src").is_dir():
            return False
        try:
            return time.time() - float(GIT_STAMP.read_text()) < GIT_PULL_THROTTLE_SECONDS
        except Exception:
            return False

    if _throttled():
        return _repo_head()

    lock_fd = open(GIT_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Another container may have synced while we waited on the lock.
        if _throttled():
            return _repo_head()

        try:
            if (REPO_PATH / ".git").is_dir():
                fetch = _git(["fetch", "--depth", "1", "origin", GIT_BRANCH], cwd=REPO_PATH)
                if fetch.returncode == 0:
                    _git(["reset", "--hard", f"origin/{GIT_BRANCH}"], cwd=REPO_PATH)
                elif not (REPO_PATH / "src").is_dir():
                    _rsync_baked()
            else:
                # No git checkout yet (fresh volume or legacy rsync copy):
                # replace it with a shallow clone of the tracked branch.
                if REPO_PATH.exists():
                    shutil.rmtree(REPO_PATH, ignore_errors=True)
                clone = _git(
                    ["clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, str(REPO_PATH)]
                )
                if clone.returncode != 0:
                    _rsync_baked()
        except Exception:
            if not (REPO_PATH / "src").is_dir():
                _rsync_baked()

        with contextlib.suppress(Exception):
            GIT_STAMP.write_text(str(time.time()))
        with contextlib.suppress(Exception):
            volume.commit()
        return _repo_head()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


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


def _events_file(job_id: str) -> Path:
    return JOBS_PATH / f"{job_id}.events.jsonl"


class _ProgressStream(io.TextIOBase):
    """Capture pipeline logs and fan them out to two surfaces (Faz 1):

    * a collapsed per-job ``{job_id}.json`` snapshot that drives the top-line
      progress bar, and
    * an append-only per-job ``{job_id}.events.jsonl`` that drives the live
      experiment console (molecule feed, leaderboard, telemetry, raw log).

    Structured progress events (lines prefixed with :data:`PROGRESS_SENTINEL`)
    drive both; any line without a sentinel falls back to the legacy heuristic
    scraping so older pipeline code keeps reporting.  Volume commits are throttled
    — the snapshot and the event log are committed together at most once per
    :data:`COMMIT_INTERVAL_SECONDS` — so a burst of per-candidate events never
    stalls GNINA on frequent commits.  A background :meth:`heartbeat_loop` keeps
    the console alive while a blocking GNINA batch emits no log lines.
    """

    COMMIT_INTERVAL_SECONDS = 1.0

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = ""
        self.percent = 8
        self.last_committed_percent = 8
        self.last_committed_step = 1
        self.last_commit_at = 0.0
        self.gnina_phase = ""
        self.gnina_phase_started_at = time.monotonic()
        # Live console state.
        self.events_path = _events_file(job_id)
        self.event_seq = 0
        self.started_at = time.monotonic()
        self.last_event_at = self.started_at
        self.last_stage = "receptor"
        self.last_stage_label = "Reseptör hazırlanıyor"
        self.last_message = "Reseptör hazırlanıyor"
        self._lock = threading.Lock()
        try:  # start every run with a clean event log
            self.events_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    # -- live event log ----------------------------------------------------
    def _append_event(self, event: dict, *, real: bool = True) -> None:
        with self._lock:
            self.event_seq += 1
            record = dict(event)
            record["seq"] = self.event_seq
            record["server_ts"] = time.time()
            try:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
            if real:
                self.last_event_at = time.monotonic()

    def _maybe_commit(self, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self.last_commit_at < self.COMMIT_INTERVAL_SECONDS:
                return
            self.last_commit_at = now
        try:
            volume.commit()
        except Exception:
            pass

    def heartbeat(self) -> None:
        now = time.monotonic()
        self._append_event(
            {
                "schema": "remedia.progress/1",
                "event": "heartbeat",
                "stage": self.last_stage,
                "stage_label": self.last_stage_label,
                "message": self.last_message,
                "percent": self.percent,
                "elapsed_seconds": round(now - self.started_at, 1),
                "since_last_event": round(now - self.last_event_at, 1),
                "processed": self.event_seq,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            real=False,
        )
        self._maybe_commit()

    def heartbeat_loop(self, stop_event: "threading.Event") -> None:
        while not stop_event.wait(1.0):
            try:
                self.heartbeat()
            except Exception:
                pass

    # -- snapshot (top-line progress bar) ----------------------------------
    def _advance(self, percent: int, message: str, step: int, *, force: bool = False) -> None:
        self.percent = max(self.percent, min(percent, 96))
        self.last_message = message
        _write_job_local(
            self.job_id, state="running", step=step,
            progress_percent=self.percent, message=message,
        )
        self.last_committed_percent = self.percent
        self.last_committed_step = step
        self._maybe_commit(force=force)

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
        """Update the collapsed job snapshot from a structured progress event.

        Discrete live events (``candidate_scored``, ``leader_changed``, …) still
        refresh the ``stage``/``message`` shown on the top-line bar but never
        rewind item counts; stage/update events carry the authoritative percent.
        """
        percent = event.get("percent")
        try:
            percent = int(round(float(percent)))
        except (TypeError, ValueError):
            percent = self.percent
        self.percent = max(self.percent, min(percent, 99))
        step = int(event.get("step", self.last_committed_step) or 1)
        label = event.get("stage_label") or event.get("task") or ""
        self.last_stage = event.get("stage") or self.last_stage
        self.last_stage_label = label or self.last_stage_label
        done, total = event.get("items_done"), event.get("items_total")
        if event.get("event"):
            # A discrete live event: keep the last stage message, don't overwrite
            # the bar with per-candidate chatter.
            message = self.last_message
        elif total:
            message = f"{label} ({done or 0}/{total})"
        else:
            message = event.get("message") or label
        self.last_message = message or self.last_message
        _write_job_local(
            self.job_id,
            state="running",
            step=step,
            progress_percent=self.percent,
            message=self.last_message,
            stage=event.get("stage"),
            stage_label=label,
            task=event.get("task"),
            items_done=done,
            items_total=total,
            eta_seconds=event.get("eta_seconds"),
        )
        self.last_committed_step = step

    def write(self, text: str) -> int:
        self.buffer += text
        if len(self.buffer) > 50000:
            self.buffer = self.buffer[-50000:]

        # Structured events (Phase 2) take priority over heuristic scraping: every
        # one is appended to the live event log and folded into the snapshot.
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
                    self._append_event(event)
                    self._structured(event)
                    handled = True
            if handled:
                self._maybe_commit()
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
        elif "boltz" in low:
            self._advance(68, clean[-180:] if clean else "Boltz-2 kompleks ve afinite tahmini yapıyor", 4)
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
    secrets=[modal.Secret.from_name("molmim")],
    timeout=4 * 60 * 60,
    max_containers=1,
)
def run_job(
    job_id: str,
    uniprot_id: str,
    molecule_count: int,
    generator: str = "reinvent4",
    pose_engine: str = "gnina",
    speed: str = "hizli",
) -> None:
    # Migrate any in-repo REINVENT4 out to its persistent sibling location
    # *before* syncing, so the one-time legacy→git checkout swap in _git_sync
    # (which may replace the old rsync copy) never deletes an installed model.
    _prepare_reinvent_location()
    _git_sync(force=True)
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
    # Heartbeat thread: proves liveness on the live console even while a blocking
    # GNINA subprocess batch produces no log lines for many seconds.
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=stream.heartbeat_loop, args=(heartbeat_stop,), daemon=True
    )
    heartbeat_thread.start()
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

        # Speed control: "hizli" runs a single fast GNINA pass (skips the second,
        # slower accurate process); "dengeli" keeps the two-stage fast→accurate
        # screening. For DiffDock/Boltz/hybrid engines this only affects the
        # GNINA portion of the run.
        docking_mode = "iki_asamali" if str(speed).lower() == "dengeli" else "sadece_fast"

        run_pipeline = _load_pipeline()
        settings = {
            "uniprot_id": uniprot_id,
            "method": "pretrained",
            "generator": generator,
            "pose_engine": pose_engine,
            "generate_n": molecule_count,
            "profile": "balanced",
            "docking_mode": docking_mode,
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

        # Phase 7/7.5: richer scientist report layered additively on top of the
        # base package. Failures here must not fail the run — fall back to the
        # base report.html.
        report_file = str(Path(report_info["report_path"]).resolve())
        try:
            known_ligands = None
            pocket_center = None
            try:
                from known_ligands import fetch_known_ligands

                known_ligands, _ = fetch_known_ligands(uniprot_id, max_results=8)
            except Exception:
                known_ligands = None
            try:
                cache = json.loads(
                    (VOLUME_PATH / "remedia_cache" / "pocket_cache.json").read_text()
                )
                entry = cache.get(uniprot_id.upper())
                pocket_center = entry.get("center") if isinstance(entry, dict) else None
            except Exception:
                pocket_center = None

            from scientific_report import build_scientific_report

            sci = build_scientific_report(
                result_dir,
                target_uniprot=uniprot_id,
                requested_molecules=molecule_count,
                settings=settings,
                pipeline_log=stream.buffer,
                job_id=job_id,
                known_ligands=known_ligands,
                pocket_center=pocket_center,
            )
            if sci.get("report_path"):
                report_file = str(Path(sci["report_path"]).resolve())
        except Exception as exc:  # keep base report if the rich one fails
            stream.write(f"Bilimsel rapor üretilemedi (temel rapor kullanılacak): {exc}\n")

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
            report_file=report_file,
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
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=3)
        stream._maybe_commit(force=True)
        os.chdir(original_cwd)


HTML = r'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remedia</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#171717;background:#f5f5f3}*{box-sizing:border-box}
body{margin:0;min-height:100vh;padding:24px;display:flex;justify-content:center}
main{width:min(1120px,100%)}
.card{background:#fff;border:1px solid #deded8;border-radius:24px;padding:28px;box-shadow:0 18px 55px #00000012}
h1{margin:0 0 6px;font-size:34px}.sub{margin:0 0 22px;color:#666}.field{margin:18px 0}label{display:block;font-weight:700;margin-bottom:8px}input{width:100%;padding:15px;border:1px solid #c9c9c3;border-radius:13px;font-size:18px}input[type=radio]{width:auto;padding:0}
.choices{display:flex;gap:8px;flex-wrap:wrap}.choice{flex:1;min-width:120px;display:flex;align-items:center;gap:8px;padding:12px 14px;border:1px solid #c9c9c3;border-radius:13px;font-weight:600;cursor:pointer}.choice:has(input:checked){border-color:#171717;background:#f0f0ec}
button,a.btn{width:100%;display:block;text-align:center;border:0;border-radius:14px;padding:16px;font-size:17px;font-weight:800;text-decoration:none;cursor:pointer;background:#171717;color:#fff}.btn.secondary{background:#ecece8;color:#171717;margin-bottom:10px}.muted{color:#74746e;font-size:14px;margin-top:10px}
.console{display:none;margin-top:22px}
.now{background:#0f0f0f;color:#f3f3f0;border-radius:18px;padding:18px 20px}
.now-head{display:flex;align-items:center;gap:10px;font-weight:800;font-size:16px}
.hb{width:12px;height:12px;border-radius:50%;background:#3ddc84;box-shadow:0 0 0 0 #3ddc8455;animation:pulse 1s infinite}
.hb.stale{background:#e0a53d;animation:none}.hb.dead{background:#c94b4b;animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 #3ddc8477}70%{box-shadow:0 0 0 9px #3ddc8400}100%{box-shadow:0 0 0 0 #3ddc8400}}
.now-task{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.now-pct{font-variant-numeric:tabular-nums;font-weight:800}
.bar{height:12px;background:#2a2a2a;border-radius:99px;overflow:hidden;margin-top:12px;position:relative}.fill{height:100%;width:0;background:linear-gradient(90deg,#3ddc84,#7fe8ac);transition:width .6s cubic-bezier(.2,.8,.2,1)}
.tele{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}.chip{background:#1c1c1c;border:1px solid #333;border-radius:10px;padding:7px 11px;font-size:12px;color:#cfcfca}.chip b{color:#fff;font-variant-numeric:tabular-nums}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}@media(max-width:760px){.cols{grid-template-columns:1fr}}
.panel{border:1px solid #e2e2dc;border-radius:16px;padding:14px;background:#fbfbf9;min-height:120px}
.panel h3{margin:0 0 10px;font-size:15px}
table.lead{width:100%;border-collapse:collapse;font-size:13px}table.lead td,table.lead th{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}table.lead th{color:#888;font-weight:700;font-size:11px;text-transform:uppercase}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}.high{background:#2e9e5b}.mid{background:#c79320}.low{background:#b5b5ad}.na{background:#c94b4b}
.sc{font-variant-numeric:tabular-nums;font-weight:700}
.lead tr.flash{animation:flash 1.1s}@keyframes flash{from{background:#fff6cf}to{background:transparent}}
.feed{max-height:340px;overflow:auto;display:flex;flex-direction:column;gap:6px}
.fi{border:1px solid #eee;border-left:3px solid #bbb;border-radius:8px;padding:7px 10px;font-size:12px;background:#fff}
.fi.ok{border-left-color:#2e9e5b}.fi.rej{border-left-color:#c94b4b}.fi.gen{border-left-color:#8a8a82}.fi.lead{border-left-color:#c79320;background:#fffdf3}
.fi .mol{font-weight:800}.fi .rs{font-variant-numeric:tabular-nums;color:#333}.fi .sm{font-family:ui-monospace,monospace;font-size:11px;color:#777;overflow-wrap:anywhere}
.rawlog{margin-top:14px}.rawlog summary{cursor:pointer;color:#666;font-size:13px;font-weight:700}.rawlog pre{max-height:220px;overflow:auto;background:#0f0f0f;color:#d6d6cf;padding:12px;border-radius:12px;font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}
.warnbox{background:#fff6df;border:1px solid #eed69d;border-radius:12px;padding:11px;margin-top:12px;font-size:13px}
.error{color:#a32020;background:#fff0f0;padding:14px;border-radius:12px;margin-top:14px;white-space:pre-wrap;overflow-wrap:anywhere}
.done{display:none;margin-top:16px}.done-note{padding:13px;background:#f6f6f3;border-radius:12px;margin-bottom:12px;color:#555}.retry{margin-top:10px;background:#555}
</style></head><body><main><div class="card"><h1>Remedia</h1><p class="sub">Hedefe göre molekül tasarımı · canlı deney konsolu</p>
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
<label class="choice"><input type="radio" name="pose" value="boltz2"> Boltz-2</label>
<label class="choice"><input type="radio" name="pose" value="hybrid"> Hybrid Validation</label>
</div></div>
<div class="field"><label>Hız (GNINA docking modu)</label><div class="choices">
<label class="choice"><input type="radio" name="speed" value="hizli" checked> Sadece hızlı (tek geçiş)</label>
<label class="choice"><input type="radio" name="speed" value="dengeli"> İki aşamalı (hızlı → doğrulama)</label>
</div><p class="muted">Sadece hızlı: tek bir hızlı GNINA taraması. İki aşamalı: önce hızlı tarama, sonra en iyi adaylar için ayrıntılı doğrulama — daha yavaş, daha sağlam.</p></div>
<button id="start" type="submit">Remedia’yı Başlat</button>
<p class="muted">Docking sonucu üretilemeyen adaylar cezalandırılır ve ayrı gösterilir. Sıralama skoru geçici bir <b>heuristik</b>tir (eğitilmiş model değildir).</p></form>

<section id="console" class="console">
<div class="now"><div class="now-head"><span id="hb" class="hb"></span><span id="nowtask" class="now-task">Hazırlanıyor…</span><span id="nowpct" class="now-pct">0%</span></div>
<div class="bar"><div id="fill" class="fill"></div></div>
<div class="tele" id="tele"></div></div>

<div class="cols">
<div class="panel"><h3>Canlı lider tablosu</h3><table class="lead"><thead><tr><th>#</th><th>Molekül</th><th>Skor</th><th>Durum</th></tr></thead><tbody id="leadbody"><tr><td colspan="4" class="muted" style="margin:0">Aday bekleniyor…</td></tr></tbody></table></div>
<div class="panel"><h3>Canlı molekül akışı</h3><div class="feed" id="feed"><div class="muted" style="margin:0">Olaylar burada canlı akacak…</div></div></div>
</div>

<div class="warnbox"><b>Not:</b> Skorlar hesaplamalı tahmindir; deneysel aktivite, toksisite veya klinik uygunluk kanıtı değildir. Docking/pose araçları bağımsız bir fiziksel kontroldür, ana sıralama motoru değildir.</div>
<details class="rawlog"><summary>Ham teknik log</summary><pre id="rawlog"></pre></details>
<div id="error"></div>
</section>

<section id="done" class="done"><div id="doneNote" class="done-note">Açıklamalı rapor ve ham veriler hazır.</div><a id="report" class="btn secondary" target="_blank">Raporu tarayıcıda aç</a><a id="download" class="btn">Tam sonuç paketini indir</a><button class="retry" onclick="location.reload()">Yeni işlem</button></section>
</div></main>
<script>
const $=s=>document.querySelector(s);
const con=$('#console'),fill=$('#fill'),nowtask=$('#nowtask'),nowpct=$('#nowpct'),hb=$('#hb'),tele=$('#tele'),leadbody=$('#leadbody'),feedEl=$('#feed'),rawlog=$('#rawlog'),err=$('#error'),done=$('#done');
const S={since:0,leaders:{},feed:[],log:[],generated:0,scored:0,leader:null,round:null,elapsed:0,sinceLast:0,pct:0,engine:''};
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function band(v){if(v==null||v==='')return'na';v=Number(v);if(v>=0.75)return'high';if(v>=0.5)return'mid';return'low';}
function fmt(v,d){if(v==null||v==='')return'—';const n=Number(v);return isNaN(n)?String(v):n.toFixed(d==null?2:d);}
form.addEventListener('submit',async e=>{e.preventDefault();$('#start').disabled=true;$('#form').style.display='none';con.style.display='block';
 const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uniprot_id:$('#u').value,molecule_count:Number($('#n').value),generator:document.querySelector('input[name=gen]:checked').value,pose_engine:document.querySelector('input[name=pose]:checked').value,speed:document.querySelector('input[name=speed]:checked').value})});
 const x=await r.json();if(!r.ok){showError(x.detail||'Başlatılamadı');return;}S.engine=document.querySelector('input[name=gen]:checked').value+' · '+document.querySelector('input[name=pose]:checked').value;poll(x.job_id);});
function setHb(){const s=S.sinceLast;hb.className='hb'+(s>15?' dead':s>6?' stale':'');}
function renderTele(job){const cps=S.elapsed>0?(S.scored/S.elapsed).toFixed(2):'0';const eta=job&&job.eta_seconds?('~'+Math.round(job.eta_seconds)+'s'):'—';
 tele.innerHTML=[['Süre',Math.round(S.elapsed)+'s'],['Üretilen',S.generated],['Skorlanan',S.scored],['Aday/sn',cps],['Kalan (ETA)',eta],['Son olay',Math.round(S.sinceLast)+'s önce'],['Motor',S.engine||'—']].map(p=>'<span class="chip">'+p[0]+' <b>'+esc(p[1])+'</b></span>').join('');}
function renderLeaders(){const rows=Object.values(S.leaders).filter(c=>c.score!=null).sort((a,b)=>b.score-a.score).slice(0,10);
 if(!rows.length){return;}leadbody.innerHTML=rows.map((c,i)=>'<tr'+(c.name===S.leader&&i===0?' class="flash"':'')+'><td>'+(i+1)+'</td><td class="mol">'+esc(c.name)+'</td><td class="sc"><span class="dot '+band(c.score)+'"></span>'+fmt(c.score)+'</td><td>'+esc(c.status||'—')+'</td></tr>').join('');}
function pushFeed(html){S.feed.unshift(html);if(S.feed.length>40)S.feed.pop();feedEl.innerHTML=S.feed.join('');}
function handle(e){const t=e.event;
 if(t==='heartbeat'){S.elapsed=e.elapsed_seconds||S.elapsed;S.sinceLast=e.since_last_event||0;if(e.percent!=null)S.pct=Math.max(S.pct,e.percent);setHb();return;}
 S.sinceLast=0;
 if(e.elapsed_seconds!=null)S.elapsed=Math.max(S.elapsed,e.elapsed_seconds);
 if(t==='candidate_generated'){S.generated++;pushFeed('<div class="fi gen"><span class="mol">'+esc(e.candidate)+'</span> üretildi <span class="sm">'+esc(e.smiles)+'</span></div>');}
 else if(t==='candidate_scored'){S.scored++;S.leaders[e.candidate]={name:e.candidate,score:e.remedia_score,status:e.docking_status,accepted:e.accepted};
  pushFeed('<div class="fi '+(e.accepted?'ok':'rej')+'"><span class="mol">'+esc(e.candidate)+'</span> <span class="rs">skor '+fmt(e.remedia_score)+'</span> · '+esc(e.reason||'')+'</div>');renderLeaders();}
 else if(t==='leader_changed'){S.leader=e.candidate;pushFeed('<div class="fi lead">🏆 Yeni lider: <span class="mol">'+esc(e.candidate)+'</span> ('+fmt(e.remedia_score)+')</div>');renderLeaders();}
 else if(t==='round_started'){S.round=e.round;pushFeed('<div class="fi">▶ Tur '+esc(e.round)+' başladı</div>');}
 else if(t==='round_completed'){pushFeed('<div class="fi">■ Tur '+esc(e.round)+' tamamlandı</div>');}
 else if(t==='warning'||e.level==='warning'){pushFeed('<div class="fi rej">⚠ '+esc(e.message||'uyarı')+'</div>');}
 else{if(e.stage_label||e.message){nowtask.textContent=(e.items_total?(e.stage_label||e.message)+' ('+(e.items_done||0)+'/'+e.items_total+')':(e.message||e.stage_label));}if(e.percent!=null)S.pct=Math.max(S.pct,e.percent);}
 if(e.message){S.log.push(e.message);if(S.log.length>400)S.log.shift();rawlog.textContent=S.log.slice(-200).join('\n');}
}
function applyJob(job){if(!job)return;if(job.progress_percent!=null)S.pct=Math.max(S.pct,job.progress_percent);const p=Math.max(0,Math.min(100,S.pct));fill.style.width=p+'%';nowpct.textContent=Math.round(p)+'%';renderTele(job);}
async function poll(id){try{const r=await fetch('/events/'+id+'?since='+S.since,{cache:'no-store'}),x=await r.json();
 S.since=x.last_seq||S.since;(x.events||[]).forEach(handle);applyJob(x.job||{});
 const st=(x.job||{}).state;
 if(st==='done'){nowtask.textContent='Tamamlandı';hb.className='hb';done.style.display='block';$('#download').href='/download/'+id;$('#report').href='/report/'+id;$('#doneNote').textContent=((x.job.candidate_count??0)+' aday; '+(x.job.scored_candidate_count??0)+' aday için docking skoru. Skor geçici bir heuristiktir.');return;}
 if(st==='error'){showError(((x.job||{}).message||'İşlem başarısız')+((x.job||{}).technical_excerpt?'\n\nTeknik ayrıntı:\n'+x.job.technical_excerpt:''));return;}
 setTimeout(()=>poll(id),900);}catch(e){setTimeout(()=>poll(id),2500);}}
function showError(t){nowtask.textContent='İşlem durdu';hb.className='hb dead';err.className='error';err.textContent=t;const b=document.createElement('button');b.className='retry';b.textContent='Tekrar dene';b.onclick=()=>location.reload();err.appendChild(b);}
</script></body></html>'''


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
        # Refreshing the URL pulls the newest code from GitHub (throttled), so a
        # push is reflected on the next run without redeploying.
        with contextlib.suppress(Exception):
            _git_sync()
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
        if pose_engine not in {"gnina", "diffdock", "boltz2", "hybrid"}:
            pose_engine = "gnina"
        speed = str(payload.get("speed", "hizli")).strip().lower()
        if speed not in {"hizli", "dengeli"}:
            speed = "hizli"

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
            speed=speed,
        )
        await volume.commit.aio()
        await run_job.spawn.aio(job_id, uniprot, molecule_count, generator, pose_engine, speed)
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

    @api.get("/events/{job_id}")
    async def events(job_id: str, since: int = 0):
        """Live event stream for the experiment console.

        Returns every structured event with ``seq > since`` from the per-job
        ``events.jsonl`` plus the latest collapsed job snapshot, so one poll (~1s)
        drives the molecule feed, leaderboard, telemetry and terminal state.
        """
        await volume.reload.aio()
        job: dict = {}
        job_path = _job_file(job_id)
        if job_path.exists():
            try:
                job = json.loads(job_path.read_text())
            except Exception:
                job = {}
        new_events: list[dict] = []
        last_seq = since
        events_path = _events_file(job_id)
        if events_path.exists():
            try:
                for line in events_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    seq = int(record.get("seq", 0) or 0)
                    if seq > since:
                        new_events.append(record)
                    last_seq = max(last_seq, seq)
            except Exception:
                pass
        return {"events": new_events, "last_seq": last_seq, "job": job}

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
        # Rewrite any local image reference (top_molecules.png, fig_*.png, …) to
        # the asset endpoint so both the base and scientist reports render images.
        report_html = re.sub(
            r"src=(['\"])([^/'\"]+\.png)\1",
            lambda m: f"src={m.group(1)}/report-asset/{job_id}/{m.group(2)}{m.group(1)}",
            report_html,
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
