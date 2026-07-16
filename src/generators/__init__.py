# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Remedia generator abstraction (Phase 3+).

REINVENT4 and every other molecule source implement :class:`BaseGenerator` and
return a :class:`GenerationResult`, so downstream stages stay generator-agnostic.

Only stdlib-safe symbols are re-exported eagerly; concrete generators import
their heavy dependencies lazily on use.
"""
from .base import BaseGenerator, GenerationResult
from .factory import available_generators, build_generator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "build_generator",
    "available_generators",
]
