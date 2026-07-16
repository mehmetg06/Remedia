# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""REINVENT4 generator behind the :class:`BaseGenerator` interface (Phase 3).

This is a faithful wrapper around the existing ``generative_model`` functions —
it makes the *same* underlying calls the pipeline made before, so behavior is
unchanged.  REINVENT4 remains the stable benchmarking baseline and is never
removed; it simply becomes one implementation among several.

Heavy dependencies (rdkit, torch, the ``reinvent`` package) are imported lazily
inside :meth:`generate`, and the sampler/installer are injectable so the class
can be unit-tested without a GPU or REINVENT installed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .base import BaseGenerator, GenerationResult


class ReinventGenerator(BaseGenerator):
    """Sample molecules from the pretrained REINVENT4 prior."""

    name = "reinvent4"

    def __init__(
        self,
        *,
        sampler: Callable[..., list[str]] | None = None,
        installer: Callable[..., Any] | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self._sampler = sampler
        self._installer = installer
        self._log = log_fn

    def _resolve(self) -> None:
        """Bind the real REINVENT functions on first use (unless injected)."""
        if self._sampler is not None and self._installer is not None:
            return
        from generative_model import generate_with_reinvent, install_reinvent

        if self._sampler is None:
            self._sampler = generate_with_reinvent
        if self._installer is None:
            self._installer = install_reinvent

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
        device: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        self._resolve()
        if reporter is not None:
            reporter.log("REINVENT4 prior hazırlanıyor / örnekleniyor")
        started = time.monotonic()

        # Same two calls the pipeline made directly: install (idempotent) then
        # sample from the prior.  drive_cache_dir preserves the install cache.
        self._installer(log_fn=self._log, drive_cache_dir=cache_dir)
        smiles = self._sampler(
            num_molecules=n,
            output_path=str(output_path) if output_path is not None else "generated_reinvent.smi",
            drive_cache_dir=cache_dir,
            device=device,
            seed=seed,
            log_fn=self._log,
        )
        smiles = list(smiles or [])
        elapsed = round(time.monotonic() - started, 2)
        if reporter is not None:
            reporter.update(len(smiles), total=max(len(smiles), n),
                            message=f"REINVENT4 üretti: {len(smiles)} molekül")
        return GenerationResult(
            smiles=smiles,
            source=self.name,
            seeds=list(seeds or []),
            requested=n,
            metadata={
                "model": "reinvent4.prior",
                "seed": seed,
                "device": device,
                "elapsed_seconds": elapsed,
                "output_path": str(output_path) if output_path is not None else None,
            },
        )
