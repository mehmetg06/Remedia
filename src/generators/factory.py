# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Factory that builds a generator from a name/config (Phase 3).

The pipeline and web UI pick a generator by name; this factory maps that name to
a concrete :class:`BaseGenerator`.  REINVENT4 is the default so the current
behavior is preserved when no generator is specified.

Phase 4 extends the registry with ``molmim`` and ``hybrid`` (imported lazily so
this module stays importable even before those are configured).
"""
from __future__ import annotations

from typing import Any

from .base import BaseGenerator
from .heuristic_generator import VALID_METHODS, HeuristicGenerator
from .reinvent_generator import ReinventGenerator

# Canonical aliases -> normalized generator key.
_ALIASES = {
    "reinvent": "reinvent4",
    "reinvent4": "reinvent4",
    "pretrained": "reinvent4",
    "molmim": "molmim",
    "nvidia": "molmim",
    "hybrid": "hybrid",
}


def available_generators() -> list[str]:
    """Names accepted by :func:`build_generator` (excludes heuristic:* variants)."""
    return ["reinvent4", "molmim", "hybrid", *[f"heuristic:{m}" for m in VALID_METHODS]]


def build_generator(name: str | None = None, **config: Any) -> BaseGenerator:
    """Build a generator instance.

    Parameters
    ----------
    name:
        Generator name (case-insensitive).  ``None``/empty defaults to REINVENT4.
        Accepts ``reinvent4``/``molmim``/``hybrid``, the heuristic methods
        (``fusion``/``genetic``/``brics``/``random`` or ``heuristic:<method>``).
    config:
        Passed through to the concrete generator's constructor.
    """
    key = (name or "reinvent4").strip().lower()

    # Heuristic methods, with or without the "heuristic:" prefix.
    if key.startswith("heuristic:"):
        return HeuristicGenerator(key.split(":", 1)[1], **config)
    if key in VALID_METHODS:
        return HeuristicGenerator(key, **config)

    key = _ALIASES.get(key, key)

    if key == "reinvent4":
        return ReinventGenerator(**config)
    if key == "molmim":
        # Phase 4 component; imported lazily so Phase 3 does not depend on it.
        from .molmim_generator import MolMIMGenerator

        return MolMIMGenerator(**config)
    if key == "hybrid":
        from .hybrid_generator import HybridGenerator

        return HybridGenerator(**config)

    raise ValueError(
        f"Bilinmeyen üretici: {name!r}. Geçerli seçenekler: {', '.join(available_generators())}"
    )
