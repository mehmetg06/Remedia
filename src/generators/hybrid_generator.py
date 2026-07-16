# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Hybrid generator: merge candidate pools from several generators (Phase 4).

By default this runs REINVENT4 and MolMIM at a 50/50 split and merges the
results, tagging each molecule with the generator that produced it.  If one
generator fails (e.g. MolMIM has no API key, or no seeds are available), the
hybrid degrades gracefully and returns the other generator's pool — a missing
MolMIM key never blocks a run.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .base import BaseGenerator, GenerationResult


class HybridGenerator(BaseGenerator):
    """Combine multiple generators into a single merged candidate pool."""

    name = "hybrid"

    def __init__(
        self,
        *,
        components: list[str] | None = None,
        weights: list[float] | None = None,
        generators: list[BaseGenerator] | None = None,
        log_fn: Callable[[str], None] = print,
        **component_config: Any,
    ) -> None:
        # Either explicit generator instances, or names built via the factory.
        self._generators = generators
        self.components = components or ["reinvent4", "molmim"]
        self.weights = weights or [0.5, 0.5]
        self._log = log_fn
        self._component_config = component_config

    def _resolve(self) -> list[BaseGenerator]:
        if self._generators is not None:
            return self._generators
        from .factory import build_generator

        built = []
        for name in self.components:
            try:
                built.append(build_generator(name, log_fn=self._log, **self._component_config))
            except Exception as exc:  # a component that can't be built is skipped
                self._log(f"[Hybrid] '{name}' üreticisi atlandı: {exc}")
        self._generators = built
        return built

    @staticmethod
    def _split(n: int, weights: list[float], k: int) -> list[int]:
        """Split n across k components by weight, distributing the remainder."""
        if k == 0:
            return []
        w = weights[:k] if len(weights) >= k else weights + [1.0] * (k - len(weights))
        total = sum(w) or 1.0
        counts = [int(n * (wi / total)) for wi in w]
        # Hand any rounding remainder to components in order.
        i = 0
        while sum(counts) < n:
            counts[i % k] += 1
            i += 1
        return counts

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
        generators = self._resolve()
        if not generators:
            raise RuntimeError("Hybrid için kullanılabilir üretici yok.")

        counts = self._split(n, self.weights, len(generators))
        merged: list[str] = []
        seen: set[str] = set()
        per_source: dict[str, str] = {}
        component_meta: list[dict[str, Any]] = []
        started = time.monotonic()

        for gen, share in zip(generators, counts):
            if share <= 0:
                continue
            if reporter is not None:
                reporter.log(f"Hybrid: {gen.name} için {share} molekül isteniyor")
            try:
                result = gen.generate(
                    target=target, n=share, seeds=seeds,
                    cache_dir=cache_dir, reporter=reporter, seed=seed,
                )
            except Exception as exc:  # graceful: keep going with other components
                self._log(f"[Hybrid] {gen.name} başarısız, atlanıyor: {exc}")
                if reporter is not None:
                    reporter.warning(f"Hybrid bileşeni {gen.name} başarısız: {exc}")
                component_meta.append({"source": gen.name, "produced": 0, "error": str(exc)})
                continue
            produced = 0
            for smi in result.smiles:
                if smi not in seen:
                    seen.add(smi)
                    merged.append(smi)
                    per_source[smi] = result.per_molecule_source.get(smi, gen.name)
                    produced += 1
            component_meta.append({"source": gen.name, "produced": produced,
                                   "metadata": result.metadata})

        if not merged:
            raise RuntimeError("Hybrid üretim hiç molekül döndürmedi (tüm bileşenler başarısız).")

        # If components under-delivered, top up from any working generator by
        # asking it for a full batch and keeping the molecules not already seen.
        if len(merged) < n:
            for gen in generators:
                if len(merged) >= n:
                    break
                try:
                    extra = gen.generate(target=target, n=n, seeds=seeds,
                                         cache_dir=cache_dir, reporter=reporter, seed=seed)
                except Exception:
                    continue
                for smi in extra.smiles:
                    if smi not in seen and len(merged) < n:
                        seen.add(smi)
                        merged.append(smi)
                        per_source[smi] = gen.name

        merged = merged[:n]
        if output_path is not None:
            self._write_smi(merged, output_path)

        return GenerationResult(
            smiles=merged,
            source=self.name,
            seeds=list(seeds or []),
            requested=n,
            metadata={
                "components": [g.name for g in generators],
                "requested_split": counts,
                "component_results": component_meta,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            },
            per_molecule_source=per_source,
        )

    @staticmethod
    def _write_smi(smiles: list[str], output_path: Any) -> None:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# SMILES  isim   (Hybrid üretici tarafından üretildi)"]
        for i, smi in enumerate(smiles):
            lines.append(f"{smi}  hybrid_{i:04d}")
        path.write_text("\n".join(lines) + "\n")
