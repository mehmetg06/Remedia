# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""NVIDIA MolMIM generator behind the :class:`BaseGenerator` interface (Phase 4).

MolMIM optimizes molecules around a seed SMILES using a hosted or self-hosted
NVIDIA NIM ``/generate`` endpoint.  This client is:

* **async** — :meth:`MolMIMGenerator.agenerate` runs the blocking HTTP work in a
  worker thread; :meth:`generate` is the synchronous entry point.
* **retrying** — transient failures (timeouts, 429, 5xx) are retried with
  exponential backoff.
* **timeout-guarded** — every request has a hard timeout.
* **structured-logging** — progress/errors flow through the Phase 2 reporter and
  ``log_fn``; nothing is swallowed silently.

Output matches the generator contract exactly (:class:`GenerationResult` with
valid SMILES), so downstream code cannot tell MolMIM from REINVENT4.

The HTTP transport is injectable so the whole client is unit-tested without
network access or credentials.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from .base import BaseGenerator, GenerationResult
from .molmim_config import MolMIMConfig, MolMIMConfigError

# A transport takes (url, headers, payload, timeout) and returns (status, body).
Transport = Callable[[str, dict, dict, float], "tuple[int, dict]"]

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _default_transport(url: str, headers: dict, payload: dict, timeout: float) -> "tuple[int, dict]":
    import requests  # imported lazily; available at runtime

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {"_raw": resp.text}
    return resp.status_code, body


def extract_smiles(body: Any) -> list[str]:
    """Pull SMILES from a MolMIM response, tolerant of schema variants.

    Handles: ``{"generated": ["SMILES", ...]}`` (documented self-hosted shape),
    ``{"generated": "<json string>"}``, and object lists under
    ``generated``/``molecules``/``samples`` whose items carry
    ``sample``/``smiles``/``smi``.
    """
    if body is None:
        return []
    if isinstance(body, str):
        try:
            return extract_smiles(json.loads(body))
        except (ValueError, TypeError):
            return [body] if body.strip() else []
    if isinstance(body, list):
        out: list[str] = []
        for item in body:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                smi = item.get("sample") or item.get("smiles") or item.get("smi")
                if smi:
                    out.append(str(smi))
        return out
    if isinstance(body, dict):
        for key in ("generated", "molecules", "samples", "smiles"):
            if key in body:
                return extract_smiles(body[key])
    return []


class MolMIMGenerator(BaseGenerator):
    """Generate molecules via the NVIDIA MolMIM NIM ``/generate`` endpoint."""

    name = "molmim"

    def __init__(
        self,
        *,
        config: MolMIMConfig | None = None,
        transport: Transport | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        log_fn: Callable[[str], None] = print,
        validate_fn: Callable[[str], str | None] | None = None,
        **config_overrides: Any,
    ) -> None:
        self.config = config or MolMIMConfig.from_env(**config_overrides)
        self._transport = transport or _default_transport
        self._sleep = sleep_fn or time.sleep
        self._log = log_fn
        self._validate = validate_fn  # optional SMILES canonicaliser (rdkit-based)

    # -- HTTP with retries --------------------------------------------------
    def _post_with_retries(self, payload: dict, reporter: Any | None) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                status, body = self._transport(
                    self.config.base_url, self.config.headers(), payload, self.config.timeout
                )
            except Exception as exc:  # network error / timeout
                last_error = exc
                self._log(f"[MolMIM] deneme {attempt} ağ hatası: {exc}")
                if reporter is not None:
                    reporter.warning(f"MolMIM ağ hatası (deneme {attempt}): {exc}")
            else:
                if status < 400:
                    return body
                message = f"[MolMIM] HTTP {status}: {str(body)[:200]}"
                last_error = RuntimeError(message)
                self._log(message)
                if status not in RETRYABLE_STATUS:
                    raise RuntimeError(message)
                if reporter is not None:
                    reporter.warning(f"MolMIM HTTP {status} (deneme {attempt})")
            if attempt < self.config.max_retries:
                self._sleep(self.config.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"MolMIM {self.config.max_retries} denemeden sonra başarısız: {last_error}")

    def _clean(self, smiles: list[str]) -> list[str]:
        if self._validate is None:
            return [s for s in smiles if s and s.strip()]
        cleaned = []
        for smi in smiles:
            canon = self._validate(smi)
            if canon:
                cleaned.append(canon)
        return cleaned

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        target: str | None = None,
        n: int = 30,
        *,
        seeds: list[str] | None = None,
        output_path: Any | None = None,
        cache_dir: Any | None = None,
        reporter: Any | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        seeds = [s for s in (seeds or []) if s]
        if not seeds:
            raise MolMIMConfigError(
                "MolMIM tohum molekül ister; bilinen ligand bulunamadı. "
                "REINVENT4 tohumsuz çalışır — hedef için ligand yoksa onu kullan."
            )
        self.config.require_ready()

        if reporter is not None:
            reporter.log("MolMIM molekül üretimi başlatılıyor")
        started = time.monotonic()

        collected: list[str] = []
        seen: set[str] = set()
        errors: list[str] = []
        # Draw from seeds in turn until we reach n (each call caps at 100).
        for idx, smi in enumerate(seeds):
            if len(collected) >= n:
                break
            remaining = n - len(collected)
            payload = self.config.build_payload(smi, remaining)
            try:
                body = self._post_with_retries(payload, reporter)
            except Exception as exc:
                errors.append(str(exc))
                self._log(f"[MolMIM] tohum {idx} başarısız: {exc}")
                continue
            for candidate in self._clean(extract_smiles(body)):
                if candidate not in seen:
                    seen.add(candidate)
                    collected.append(candidate)
            if reporter is not None:
                reporter.update(min(len(collected), n), total=n,
                                message=f"MolMIM üretti: {len(collected)}/{n}")

        if not collected and errors:
            raise RuntimeError("MolMIM hiç molekül üretemedi: " + " | ".join(errors[:3]))

        collected = collected[:n]
        elapsed = round(time.monotonic() - started, 2)

        if output_path is not None:
            self._write_smi(collected, output_path)

        return GenerationResult(
            smiles=collected,
            source=self.name,
            seeds=seeds,
            requested=n,
            metadata={
                "endpoint": self.config.base_url,
                "algorithm": self.config.algorithm,
                "property_name": self.config.property_name,
                "min_similarity": self.config.min_similarity,
                "iterations": self.config.iterations,
                "particles": self.config.particles,
                "elapsed_seconds": elapsed,
                "errors": errors,
            },
        )

    async def agenerate(self, target: str | None = None, n: int = 30, **kwargs: Any) -> GenerationResult:
        """Async entry point — runs :meth:`generate` in a worker thread."""
        return await asyncio.to_thread(lambda: self.generate(target, n, **kwargs))

    @staticmethod
    def _write_smi(smiles: list[str], output_path: Any) -> None:
        """Write results in the pipeline's .smi format without importing rdkit."""
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# SMILES  isim   (MolMIM tarafından üretildi)"]
        for i, smi in enumerate(smiles):
            lines.append(f"{smi}  molmim_{i:04d}")
        path.write_text("\n".join(lines) + "\n")
