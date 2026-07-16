# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Structured progress & observability for the Remedia pipeline.

Phase 2 of the V3 modernization.  Historically the Modal web worker inferred
progress by string-matching pipeline stdout (``_ProgressStream``).  That approach
is fragile: it breaks when a log phrase changes and cannot report real item
counts ("Docked 8/20").  This module gives the pipeline a small, dependency-free
API to emit **structured** progress events with a stable, machine-readable
schema, while remaining fully compatible with that existing UI.

Design goals
------------
* **Stdlib only.**  Importable in the test environment (no rdkit/pandas/numpy).
* **Backward compatible.**  Every event is also printed to stdout as a human
  sentence, so the legacy stdout-scraping fallback keeps working unchanged.  The
  machine-readable payload is printed on its own line prefixed with
  :data:`SENTINEL` so a consumer can parse it without regexes.
* **Persistent logs.**  Events are appended to ``progress.jsonl`` and the latest
  state is mirrored to ``progress_state.json`` in the run directory.
* **Full exception visibility.**  :meth:`ProgressReporter.exception` records the
  complete traceback; nothing is ever swallowed.

Event schema (one JSON object per event)::

    {
      "schema": "remedia.progress/1",
      "stage": "dock_fast",            # machine key, see STAGES
      "stage_label": "GNINA FAST docking",
      "task": "Docking molecule library",
      "items_done": 18,
      "items_total": 50,
      "percent": 63.0,                 # overall 0-100
      "step": 4,                       # legacy 1..5 stage bucket
      "eta_seconds": 42.0,             # estimate, may be null
      "elapsed_seconds": 12.3,
      "level": "info",                 # info | warning | error
      "message": "GNINA FAST docking (18/50)",
      "timestamp": "2026-07-16T20:17:00+00:00"
    }
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

SCHEMA = "remedia.progress/1"
#: Prefix that marks a machine-readable progress line on stdout.
SENTINEL = "[[REMEDIA_PROGRESS]]"

LEVEL_INFO = "info"
LEVEL_WARNING = "warning"
LEVEL_ERROR = "error"


@dataclass(frozen=True)
class Stage:
    """Static description of a pipeline stage."""

    key: str
    label: str
    #: Legacy 1..5 bucket used by the existing 5-dot Modal UI.
    step: int
    #: Fraction of the whole run this stage occupies, used to turn a per-stage
    #: item fraction into an overall percentage.  Values need not sum to 1.0;
    #: they are normalised at runtime.
    weight: float


# Ordered list of stages that make up a standard Remedia run.  New pose engines
# or generators reuse these keys so downstream consumers stay engine-agnostic.
STAGES: tuple[Stage, ...] = (
    Stage("receptor", "Downloading AlphaFold structure", 1, 0.05),
    Stage("pocket", "Detecting binding pocket", 2, 0.05),
    Stage("seeds", "Fetching known ligands", 2, 0.03),
    Stage("generate", "Generating molecules", 3, 0.17),
    Stage("dock_fast", "GNINA FAST docking", 4, 0.30),
    Stage("dock_accurate", "GNINA ACCURATE docking", 4, 0.20),
    Stage("pose", "Pose prediction", 4, 0.30),
    Stage("admet", "ADMET filtering", 5, 0.07),
    Stage("rank", "Ranking candidates", 5, 0.03),
    Stage("report", "Generating report", 5, 0.05),
    Stage("done", "Completed", 5, 0.0),
    Stage("error", "Failed", 5, 0.0),
)

_STAGE_BY_KEY: dict[str, Stage] = {s.key: s for s in STAGES}


def stage_for(key: str) -> Stage:
    """Return the :class:`Stage` for *key*, or a synthetic one if unknown.

    Unknown keys never raise — an evolving pipeline may introduce new stage
    names, and observability must degrade gracefully rather than crash a run.
    """
    stage = _STAGE_BY_KEY.get(key)
    if stage is not None:
        return stage
    return Stage(key, key.replace("_", " ").title(), 5, 0.0)


def _cumulative_bounds() -> dict[str, tuple[float, float]]:
    """Map each stage key to its (start, end) fraction of the overall run."""
    total = sum(s.weight for s in STAGES) or 1.0
    bounds: dict[str, tuple[float, float]] = {}
    acc = 0.0
    for stage in STAGES:
        start = acc / total
        acc += stage.weight
        end = acc / total
        bounds[stage.key] = (start, end)
    return bounds


_BOUNDS = _cumulative_bounds()


@dataclass
class ProgressEvent:
    """A single structured progress event (see module docstring for schema)."""

    stage: str
    stage_label: str
    task: str
    items_done: int | None
    items_total: int | None
    percent: float
    step: int
    eta_seconds: float | None
    elapsed_seconds: float
    level: str
    message: str
    timestamp: str
    schema: str = SCHEMA
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Flatten ``extra`` so consumers see a single object.
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data

    def human_line(self) -> str:
        """Render a human-friendly one-liner, e.g. ``GNINA FAST docking (18/50)``."""
        if self.message:
            return self.message
        base = self.task or self.stage_label
        if self.items_total:
            return f"{base} ({self.items_done or 0}/{self.items_total})"
        return base


class ProgressReporter:
    """Emit structured progress events for a single pipeline run.

    Parameters
    ----------
    log_dir:
        Directory for persistent logs (``progress.jsonl`` + ``progress_state.json``).
        If ``None`` no files are written (useful for tests / library use).
    sink:
        Optional callback invoked with each event's ``dict`` — the Modal worker
        uses this to mirror progress into its per-job JSON on the shared volume.
    emit_stdout:
        When ``True`` (default) each event is echoed to stdout: a human sentence
        plus a :data:`SENTINEL` line carrying the JSON payload.  This preserves
        the legacy stdout-scraping fallback and needs no cross-process plumbing.
    stream:
        Stream to write stdout echoes to (defaults to the real ``sys.stdout`` at
        call time, so it participates in any active redirect).
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        sink: Callable[[dict[str, Any]], None] | None = None,
        emit_stdout: bool = True,
        stream: Any | None = None,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.sink = sink
        self.emit_stdout = emit_stdout
        self._stream = stream
        self._start = time.monotonic()
        self._stage_start = self._start
        self._current = STAGES[0]
        self._items_done: int | None = None
        self._items_total: int | None = None
        self._last_percent = 0.0
        self._task = ""
        self.events: list[dict[str, Any]] = []
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = self.log_dir / "progress.jsonl"
            self._state = self.log_dir / "progress_state.json"
        else:
            self._jsonl = None
            self._state = None

    # -- public API --------------------------------------------------------
    def stage(
        self,
        key: str,
        *,
        task: str | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> "ProgressReporter":
        """Begin a new stage.  Resets item counters and the stage timer."""
        self._current = stage_for(key)
        self._stage_start = time.monotonic()
        self._items_total = total
        self._items_done = 0 if total else None
        self._task = task or self._current.label
        self._emit(message=message, level=LEVEL_INFO)
        return self

    def update(
        self,
        done: int | None = None,
        *,
        total: int | None = None,
        task: str | None = None,
        message: str | None = None,
        level: str = LEVEL_INFO,
        **extra: Any,
    ) -> None:
        """Update item progress within the current stage."""
        if total is not None:
            self._items_total = total
        if done is not None:
            self._items_done = done
        if task is not None:
            self._task = task
        self._emit(message=message, level=level, extra=extra)

    def log(self, message: str, *, level: str = LEVEL_INFO, **extra: Any) -> None:
        """Emit an informational (or warning) message without changing counts."""
        self._emit(message=message, level=level, extra=extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._emit(message=message, level=LEVEL_WARNING, extra=extra)

    def exception(self, exc: BaseException, *, context: str | None = None) -> dict[str, Any]:
        """Record a full traceback and emit an error event.  Never swallows."""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = context or f"{type(exc).__name__}: {exc}"
        event = self._build_event(
            stage_key="error",
            level=LEVEL_ERROR,
            message=message,
            extra={"traceback": tb, "error_type": type(exc).__name__},
        )
        self._publish(event)
        return event

    def done(self, message: str = "Completed") -> None:
        self._current = stage_for("done")
        self._items_done = self._items_total
        event = self._build_event(stage_key="done", level=LEVEL_INFO, message=message)
        event["percent"] = 100.0
        self._publish(event)

    def snapshot(self) -> dict[str, Any]:
        """Return the most recent event dict (or an empty starting state)."""
        if self.events:
            return dict(self.events[-1])
        return self._build_event(stage_key=self._current.key, level=LEVEL_INFO, message="")

    # -- internals ---------------------------------------------------------
    def _percent(self) -> float:
        start, end = _BOUNDS.get(self._current.key, (0.0, 0.0))
        span = end - start
        frac = 0.0
        if self._items_total:
            frac = min(1.0, max(0.0, (self._items_done or 0) / self._items_total))
        elif self._current.key in ("done",):
            frac = 1.0
        percent = (start + span * frac) * 100.0
        # Progress must never move backwards on the bar.
        percent = max(percent, self._last_percent)
        return round(min(100.0, percent), 1)

    def _eta(self) -> float | None:
        if not self._items_total or not self._items_done:
            return None
        elapsed = time.monotonic() - self._stage_start
        rate = self._items_done / elapsed if elapsed > 0 else 0
        if rate <= 0:
            return None
        remaining = max(0, self._items_total - self._items_done)
        return round(remaining / rate, 1)

    def _build_event(
        self,
        *,
        stage_key: str,
        level: str,
        message: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = stage_for(stage_key)
        percent = self._percent()
        event = ProgressEvent(
            stage=stage.key,
            stage_label=stage.label,
            task=self._task or stage.label,
            items_done=self._items_done,
            items_total=self._items_total,
            percent=percent,
            step=stage.step,
            eta_seconds=self._eta(),
            elapsed_seconds=round(time.monotonic() - self._start, 2),
            level=level,
            message="",
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            extra=extra or {},
        )
        data = event.to_dict()
        data["message"] = message if message is not None else event.human_line()
        return data

    def _emit(
        self,
        *,
        message: str | None,
        level: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event = self._build_event(
            stage_key=self._current.key, level=level, message=message, extra=extra
        )
        self._publish(event)

    def _publish(self, event: dict[str, Any]) -> None:
        self._last_percent = max(self._last_percent, float(event.get("percent", 0.0)))
        self.events.append(event)
        self._write_persistent(event)
        if self.emit_stdout:
            self._echo(event)
        if self.sink is not None:
            try:
                self.sink(event)
            except Exception:  # sink failures must not abort the pipeline
                pass

    def _write_persistent(self, event: dict[str, Any]) -> None:
        if self._jsonl is None or self._state is None:
            return
        line = json.dumps(event, ensure_ascii=False)
        try:
            with self._jsonl.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            tmp = self._state.with_suffix(".tmp")
            tmp.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._state)
        except Exception:  # logging must never crash the run
            pass

    def _echo(self, event: dict[str, Any]) -> None:
        stream = self._stream or sys.stdout
        try:
            # Human line first (drives the legacy scraper fallback), then the
            # machine payload on its own SENTINEL line.
            stream.write(event.get("message", "") + "\n")
            stream.write(f"{SENTINEL} {json.dumps(event, ensure_ascii=False)}\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            pass


def parse_sentinel(line: str) -> dict[str, Any] | None:
    """Parse a stdout line into an event dict if it carries the sentinel.

    Returns ``None`` for ordinary log lines so a consumer can fall back to
    heuristic scraping.
    """
    idx = line.find(SENTINEL)
    if idx < 0:
        return None
    payload = line[idx + len(SENTINEL):].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if isinstance(data, dict) and data.get("schema") == SCHEMA:
        return data
    return None
