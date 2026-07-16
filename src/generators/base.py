# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Generator abstraction for Remedia (Phase 3).

Every molecule source — REINVENT4 (baseline), NVIDIA MolMIM (Phase 4), the
heuristic RDKit generators, or a Hybrid pool — implements the same
:class:`BaseGenerator` interface and returns the same :class:`GenerationResult`.
Downstream code (docking, ADMET, ranking, reporting) therefore never needs to
know *which* generator produced a molecule; it only consumes ``result.smiles``.

This module is dependency-free (stdlib only).  Concrete generators import their
heavy dependencies (rdkit, torch, REINVENT, HTTP clients) lazily so this package
stays importable in minimal/test environments.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Uniform output of every generator.

    Attributes
    ----------
    smiles:
        Valid, de-duplicated candidate SMILES — the contract downstream stages
        rely on.  Order is preserved.
    source:
        Generator identifier, e.g. ``"reinvent4"``, ``"molmim"``, ``"hybrid"``.
    seeds:
        Seed SMILES used (may be empty; REINVENT sampling is seed-free).
    requested:
        Number of molecules requested (``n``).
    metadata:
        Free-form provenance (model version, seed, device, timings, params).
        Captured into ``run_manifest.json`` for reproducibility (Phase 9).
    per_molecule_source:
        Maps each SMILES to the generator that produced it.  For single
        generators this is uniform; for Hybrid it records the real origin.
    """

    smiles: list[str]
    source: str
    seeds: list[str] = field(default_factory=list)
    requested: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    per_molecule_source: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # De-duplicate while preserving order; guarantee the source map is filled.
        seen: set[str] = set()
        ordered: list[str] = []
        for smi in self.smiles:
            if not smi or smi in seen:
                continue
            seen.add(smi)
            ordered.append(smi)
            self.per_molecule_source.setdefault(smi, self.source)
        self.smiles = ordered

    @property
    def count(self) -> int:
        return len(self.smiles)

    def as_molecule_list(self, prefix: str = "mol") -> list[tuple[str, str]]:
        """Return ``[(name, smiles), ...]`` in the pipeline's canonical shape."""
        return [(f"{prefix}_{i:03d}", smi) for i, smi in enumerate(self.smiles, 1)]

    def to_manifest(self) -> dict[str, Any]:
        """Compact, JSON-serialisable provenance record."""
        return {
            "source": self.source,
            "requested": self.requested,
            "produced": self.count,
            "seeds": list(self.seeds),
            "metadata": dict(self.metadata),
            "per_source_counts": self.per_source_counts(),
        }

    def per_source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for smi in self.smiles:
            src = self.per_molecule_source.get(smi, self.source)
            counts[src] = counts.get(src, 0) + 1
        return counts


class BaseGenerator(abc.ABC):
    """Interface every molecule generator implements.

    Parameters passed to :meth:`generate`:

    target:
        Target context (e.g. UniProt id).  Included for a uniform signature even
        though prior-sampling generators are not target-conditioned.
    n:
        Number of molecules to produce.
    seeds:
        Optional seed SMILES (used by seed-based / heuristic generators).
    output_path:
        Optional ``.smi`` path; when given the generator writes results in the
        pipeline's existing ``write_smi`` format for backward compatibility.
    reporter:
        Optional ``progress.ProgressReporter`` for structured progress/logging.
    """

    #: Stable identifier used by the factory and recorded in provenance.
    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        target: str | None = None,
        n: int = 30,
        *,
        seeds: list[str] | None = None,
        output_path: Any | None = None,
        reporter: Any | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Produce candidate molecules.  Must return a :class:`GenerationResult`."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
