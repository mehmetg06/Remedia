# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Factory that builds a pose predictor from a name/config."""
from __future__ import annotations

from typing import Any, Callable

from .base import BasePosePredictor
from .boltz_predictor import BoltzPredictor
from .diffdock_predictor import DEFAULT_DIFFDOCK_URL, DiffDockPredictor  # noqa: F401
from .gnina_predictor import DEFAULT_GNINA_PATH, GninaPredictor
from .hybrid_validation import HybridValidationPredictor

_ALIASES = {
    "gnina": "gnina",
    "vina": "gnina",
    "diffdock": "diffdock",
    "boltz": "boltz2",
    "boltz2": "boltz2",
    "boltz-2": "boltz2",
    "hybrid": "hybrid",
    "hybrid_validation": "hybrid",
    "hybrid validation": "hybrid",
    "hibrit": "hybrid",
}


def available_pose_engines() -> list[str]:
    return ["gnina", "diffdock", "boltz2", "hybrid"]


def build_pose_predictor(
    name: str | None = None,
    *,
    profile: str = "balanced",
    docking_mode: str = "iki_asamali",
    top_fraction: float = 0.10,
    gnina_path: str = DEFAULT_GNINA_PATH,
    diffdock_results_csv: Any | None = None,
    diffdock_runner: Callable[..., Any] | None = None,
    hybrid_top_fraction: float = 0.25,
    boltz_cache_dir: Any | None = None,
    boltz_use_msa_server: bool = False,
    boltz_recycling_steps: int = 2,
    boltz_sampling_steps: int = 80,
    log_fn: Callable[[str], None] = print,
    **_ignored: Any,
) -> BasePosePredictor:
    """Build a pose predictor. ``None``/empty defaults to GNINA."""
    key = _ALIASES.get((name or "gnina").strip().lower(), (name or "gnina").strip().lower())

    if key == "gnina":
        return GninaPredictor(
            profile=profile,
            docking_mode=docking_mode,
            top_fraction=top_fraction,
            gnina_path=gnina_path,
            log_fn=log_fn,
        )
    if key == "diffdock":
        return DiffDockPredictor(
            results_csv=diffdock_results_csv,
            runner=diffdock_runner,
            log_fn=log_fn,
        )
    if key == "boltz2":
        return BoltzPredictor(
            cache_dir=boltz_cache_dir,
            use_msa_server=boltz_use_msa_server,
            recycling_steps=boltz_recycling_steps,
            sampling_steps=boltz_sampling_steps,
            log_fn=log_fn,
        )
    if key == "hybrid":
        return HybridValidationPredictor(
            gnina=GninaPredictor(
                profile=profile,
                docking_mode=docking_mode,
                top_fraction=top_fraction,
                gnina_path=gnina_path,
                log_fn=log_fn,
            ),
            diffdock=DiffDockPredictor(
                results_csv=diffdock_results_csv,
                runner=diffdock_runner,
                log_fn=log_fn,
            ),
            top_fraction=hybrid_top_fraction,
            log_fn=log_fn,
        )
    raise ValueError(
        f"Bilinmeyen poz motoru: {name!r}. Geçerli: {', '.join(available_pose_engines())}"
    )
