# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Configuration + credential resolution for the NVIDIA MolMIM generator (Phase 4).

MolMIM is served as an NVIDIA NIM.  Two deployment shapes are supported:

* **Hosted** (build.nvidia.com / health.api.nvidia.com) — requires an API key
  (``nvapi-...``) sent as a Bearer token.
* **Self-hosted** NIM container — usually ``http://localhost:8000`` with no key.

Credentials are read from the environment; nothing is hard-coded.  When no key
is configured the generator raises :class:`MolMIMConfigError` with setup guidance
(see ``docs/molmim_setup.md``).  Hybrid mode catches that and falls back to
REINVENT4, so a missing key never blocks a run.

Reference: https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Hosted NVIDIA NIM default. Override with MOLMIM_BASE_URL for self-hosted NIMs
# (e.g. http://localhost:8000/generate).
DEFAULT_BASE_URL = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

# Environment variables consulted for the API key, in priority order.
API_KEY_ENV_VARS = ("MOLMIM_API_KEY", "NVIDIA_API_KEY", "NGC_API_KEY", "NVCF_RUN_KEY")
BASE_URL_ENV_VAR = "MOLMIM_BASE_URL"

# MolMIM /generate defaults and valid ranges (per NVIDIA docs).
DEFAULTS = {
    "algorithm": "CMA-ES",       # "CMA-ES" | "none"
    "property_name": "QED",      # "QED" | "plogP"
    "minimize": False,
    "min_similarity": 0.3,       # 0.0 - 0.7
    "particles": 30,             # 2 - 1000
    "iterations": 10,            # 1 - 1000
    "scaled_radius": 1.0,        # 0.0 - 2.0
}
MAX_NUM_MOLECULES = 100          # hard cap per /generate call


class MolMIMConfigError(RuntimeError):
    """Raised when MolMIM is requested but not configured (e.g. no API key)."""


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Return the first configured API key, or ``None`` if unset."""
    if explicit:
        return explicit
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class MolMIMConfig:
    """Resolved MolMIM client configuration."""

    base_url: str = ""
    api_key: str | None = None
    algorithm: str = DEFAULTS["algorithm"]
    property_name: str = DEFAULTS["property_name"]
    minimize: bool = DEFAULTS["minimize"]
    min_similarity: float = DEFAULTS["min_similarity"]
    particles: int = DEFAULTS["particles"]
    iterations: int = DEFAULTS["iterations"]
    scaled_radius: float = DEFAULTS["scaled_radius"]
    timeout: float = 120.0
    max_retries: int = 3
    backoff_seconds: float = 2.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "MolMIMConfig":
        cfg = cls(
            base_url=os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL),
            api_key=resolve_api_key(overrides.pop("api_key", None)),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.normalise()
        return cfg

    def normalise(self) -> "MolMIMConfig":
        self.min_similarity = _clamp(float(self.min_similarity), 0.0, 0.7)
        self.scaled_radius = _clamp(float(self.scaled_radius), 0.0, 2.0)
        self.particles = int(_clamp(int(self.particles), 2, 1000))
        self.iterations = int(_clamp(int(self.iterations), 1, 1000))
        if self.algorithm not in ("CMA-ES", "none"):
            self.algorithm = "CMA-ES"
        if self.property_name not in ("QED", "plogP"):
            self.property_name = "QED"
        return self

    def require_ready(self) -> None:
        """Raise :class:`MolMIMConfigError` unless the client can make a call."""
        if not self.base_url:
            raise MolMIMConfigError("MOLMIM_BASE_URL / endpoint yapılandırılmadı.")
        hosted = "api.nvidia.com" in self.base_url
        if hosted and not self.api_key:
            raise MolMIMConfigError(
                "NVIDIA MolMIM API anahtarı bulunamadı. "
                f"Şu ortam değişkenlerinden birini ayarla: {', '.join(API_KEY_ENV_VARS)}. "
                "Kurulum: docs/molmim_setup.md"
            )

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_payload(self, smi: str, num_molecules: int) -> dict[str, Any]:
        payload = {
            "smi": smi,
            "algorithm": self.algorithm,
            "num_molecules": int(max(1, min(num_molecules, MAX_NUM_MOLECULES))),
            "property_name": self.property_name,
            "minimize": bool(self.minimize),
            "min_similarity": self.min_similarity,
            "particles": self.particles,
            "iterations": self.iterations,
            "scaled_radius": self.scaled_radius,
        }
        payload.update(self.extra)
        return payload
