# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Remedia pose predictor abstraction (Phase 5+).

GNINA and DiffDock (and Hybrid Validation) implement :class:`BasePosePredictor`
and return a :class:`PoseResult`, so ranking/reporting stay engine-agnostic.
GNINA remains the default and is never removed.
"""
from .base import BasePosePredictor, PoseResult, PoseScore
from .diffdock_predictor import DiffDockPredictor, DiffDockUnavailable
from .factory import available_pose_engines, build_pose_predictor
from .gnina_predictor import GninaPredictor
from .hybrid_validation import HybridValidationPredictor

#: Alias: the exception a caller catches to fall back to GNINA.
PoseUnavailable = DiffDockUnavailable

__all__ = [
    "BasePosePredictor",
    "PoseResult",
    "PoseScore",
    "GninaPredictor",
    "DiffDockPredictor",
    "DiffDockUnavailable",
    "PoseUnavailable",
    "HybridValidationPredictor",
    "build_pose_predictor",
    "available_pose_engines",
]
