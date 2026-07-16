# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""RDKit heuristic generators behind the :class:`BaseGenerator` interface.

Wraps the existing ``molecule_generator`` methods (fusion, genetic, BRICS,
random mutation) so they are available through the same interface as REINVENT4
and MolMIM.  This is what makes seed-based generation usable inside Hybrid pools
and the benchmark framework (Phase 8) without special-casing.

``molecule_generator`` imports rdkit at module top, so it is imported lazily here
and the underlying functions are injectable for testing.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .base import BaseGenerator, GenerationResult

VALID_METHODS = ("fusion", "genetic", "brics", "random")


class HeuristicGenerator(BaseGenerator):
    """Seed-based RDKit generation (fusion / genetic / brics / random)."""

    def __init__(
        self,
        method: str = "random",
        *,
        functions: dict[str, Callable[..., Any]] | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        if method not in VALID_METHODS:
            raise ValueError(f"Bilinmeyen heuristic yöntem: {method!r} (geçerli: {VALID_METHODS})")
        self.method = method
        self.name = f"heuristic:{method}"
        self._functions = functions
        self._log = log_fn

    def _resolve(self) -> dict[str, Callable[..., Any]]:
        if self._functions is not None:
            return self._functions
        from molecule_generator import (
            brics_recombination,
            fusion_generation,
            genetic_algorithm,
            random_mutation,
        )

        self._functions = {
            "fusion": fusion_generation,
            "genetic": genetic_algorithm,
            "brics": brics_recombination,
            "random": random_mutation,
        }
        return self._functions

    def generate(
        self,
        target: str | None = None,
        n: int = 30,
        *,
        seeds: list[str] | None = None,
        output_path: Any | None = None,
        reporter: Any | None = None,
        seed: int | None = None,
        generations: int = 3,
        **kwargs: Any,
    ) -> GenerationResult:
        seeds = list(seeds or [])
        if not seeds:
            raise ValueError("Heuristic üretim için en az bir tohum molekül gerekir.")
        funcs = self._resolve()
        if reporter is not None:
            reporter.log(f"Heuristic üretim: {self.method}")
        started = time.monotonic()

        if self.method == "fusion":
            final, _ = funcs["fusion"](
                seeds, docking_opts=None, log_fn=self._log,
                population_size=max(10, n), generations=generations,
            )
            smiles = [smi for smi, _score in final]
        elif self.method == "genetic":
            final, _ = funcs["genetic"](
                seeds, generations=generations, population_size=max(10, n),
                docking_opts=None, log_fn=self._log,
            )
            smiles = [smi for smi, _score in final]
        elif self.method == "brics":
            smiles = list(funcs["brics"](seeds, n=n))
        else:  # random
            smiles = list(funcs["random"](seeds, n=n))

        smiles = [s for s in smiles if s][:n]
        elapsed = round(time.monotonic() - started, 2)
        if reporter is not None:
            reporter.update(len(smiles), total=max(len(smiles), n),
                            message=f"{self.method} üretti: {len(smiles)} molekül")
        return GenerationResult(
            smiles=smiles,
            source=self.name,
            seeds=seeds,
            requested=n,
            metadata={"method": self.method, "generations": generations,
                      "seed": seed, "elapsed_seconds": elapsed},
        )
