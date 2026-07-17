# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).

"""Boltz-2 protein-ligand co-folding and affinity predictor.

Boltz has a NumPy <2 requirement while the main Remedia/REINVENT environment
uses NumPy 2. To avoid dependency conflicts, Boltz is installed on first use in
an isolated virtual environment persisted on the Modal Volume.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from .base import BasePosePredictor, PoseResult, PoseScore


class BoltzUnavailable(RuntimeError):
    """Raised when Boltz-2 cannot run in the current environment."""


def _sequence_from_pdb(path: Path) -> str:
    """Extract the longest standard amino-acid chain from a receptor PDB."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
        from Bio.SeqUtils import seq1
    except Exception as exc:  # pragma: no cover
        raise BoltzUnavailable(f"Biopython yüklenemedi: {exc}") from exc

    structure = PDBParser(QUIET=True).get_structure("receptor", str(path))
    sequences: list[str] = []
    for chain in next(structure.get_models()):
        residues = [r for r in chain if is_aa(r, standard=True)]
        sequence = "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in residues)
        if sequence:
            sequences.append(sequence)
    if not sequences:
        raise BoltzUnavailable("Reseptör PDB dosyasından protein dizisi çıkarılamadı")
    return max(sequences, key=len)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


class BoltzPredictor(BasePosePredictor):
    """Boltz-2 complex structure + affinity prediction."""

    name = "boltz2"

    def __init__(
        self,
        *,
        executable: str | None = None,
        cache_dir: str | Path | None = None,
        env_dir: str | Path | None = None,
        use_msa_server: bool = False,
        use_potentials: bool = True,
        recycling_steps: int = 2,
        sampling_steps: int = 80,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.executable = executable or os.environ.get("BOLTZ_PATH", "boltz")
        self.cache_dir = Path(cache_dir or os.environ.get("BOLTZ_CACHE", "/workspace/boltz_cache"))
        self.env_dir = Path(env_dir or os.environ.get("BOLTZ_ENV", "/workspace/boltz_env"))
        self.use_msa_server = use_msa_server
        self.use_potentials = use_potentials
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self._log = log_fn

    def _ensure_executable(self, reporter: Any | None) -> str:
        direct = shutil.which(self.executable)
        if direct:
            return direct

        candidate = self.env_dir / "bin" / "boltz"
        if candidate.is_file():
            return str(candidate)

        message = "Boltz-2 ayrı ortamı ilk kez kuruluyor; bu işlem yalnızca ilk koşuda uzun sürebilir"
        self._log(message)
        if reporter is not None:
            reporter.log(message)

        self.env_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.env_dir)],
                check=True,
                text=True,
                capture_output=True,
            )
            python_bin = self.env_dir / "bin" / "python"
            install = subprocess.run(
                [
                    str(python_bin),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                    "setuptools",
                    "wheel",
                    "boltz==2.2.1",
                ],
                text=True,
                capture_output=True,
            )
        except Exception as exc:
            raise BoltzUnavailable(f"Boltz-2 ayrı ortamı oluşturulamadı: {exc}") from exc

        if install.returncode != 0:
            detail = (install.stderr or install.stdout or "bilinmeyen hata")[-5000:]
            raise BoltzUnavailable(f"Boltz-2 kurulumu başarısız: {detail}")
        if not candidate.is_file():
            raise BoltzUnavailable("Boltz-2 kuruldu ancak komut bulunamadı")
        return str(candidate)

    def predict_pose(
        self,
        molecules: list[tuple[str, str]],
        *,
        receptor: str | None = None,
        center: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        out_dir: Any | None = None,
        reporter: Any | None = None,
        **kwargs: Any,
    ) -> PoseResult:
        del center, size, kwargs
        executable = self._ensure_executable(reporter)
        receptor_path = Path(receptor or "")
        if not receptor_path.is_file():
            raise BoltzUnavailable(f"Boltz-2 reseptör PDB dosyasını bulamadı: {receptor}")
        if not molecules:
            return PoseResult(engine=self.name, scores=[], rows=[], stage_info={}, metadata={})

        root = Path(out_dir or ".") / "boltz2"
        inputs = root / "inputs"
        outputs = root / "outputs"
        inputs.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sequence = _sequence_from_pdb(receptor_path)

        for name, smiles in molecules:
            safe = _safe_name(name)
            payload = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
                    {"ligand": {"id": "B", "smiles": smiles}},
                ],
                "templates": [{"pdb": str(receptor_path)}],
                "properties": [{"affinity": {"binder": "B"}}],
            }
            (inputs / f"{safe}.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

        command = [
            executable,
            "predict",
            str(inputs),
            "--out_dir",
            str(outputs),
            "--cache",
            str(self.cache_dir),
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--recycling_steps",
            str(self.recycling_steps),
            "--sampling_steps",
            str(self.sampling_steps),
            "--diffusion_samples",
            "1",
            "--output_format",
            "pdb",
            "--override",
        ]
        if self.use_potentials:
            command.append("--use_potentials")
        if self.use_msa_server:
            command.append("--use_msa_server")

        message = f"Boltz-2: {len(molecules)} kompleks birlikte değerlendiriliyor"
        self._log(message)
        if reporter is not None:
            reporter.log(message)
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            self._log(result.stdout[-4000:])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "bilinmeyen hata")[-4000:]
            raise BoltzUnavailable(f"Boltz-2 başarısız (kod {result.returncode}): {detail}")

        scores: list[PoseScore] = []
        for name, _smiles in molecules:
            safe = _safe_name(name)
            pred_dir = outputs / "predictions" / safe
            confidence_file = pred_dir / f"confidence_{safe}_model_0.json"
            affinity_file = pred_dir / f"affinity_{safe}.json"
            confidence_data = json.loads(confidence_file.read_text()) if confidence_file.is_file() else {}
            affinity_data = json.loads(affinity_file.read_text()) if affinity_file.is_file() else {}
            structure_confidence = confidence_data.get("confidence_score")
            binder_probability = affinity_data.get("affinity_probability_binary")
            affinity_log10_ic50 = affinity_data.get("affinity_pred_value")
            success = structure_confidence is not None or binder_probability is not None
            scores.append(
                PoseScore(
                    ligand=name,
                    affinity_kcal_mol=None,
                    confidence=float(binder_probability if binder_probability is not None else structure_confidence)
                    if success else None,
                    success=success,
                    source="boltz2",
                    error=None if success else "Boltz-2 çıktı skoru bulunamadı",
                    extra={
                        "boltz_structure_confidence": structure_confidence,
                        "boltz_binder_probability": binder_probability,
                        "boltz_affinity_log10_ic50_uM": affinity_log10_ic50,
                        "boltz_output_dir": str(pred_dir),
                    },
                )
            )

        rows = [score.to_row() for score in scores]
        ok = sum(score.success for score in scores)
        if reporter is not None:
            reporter.update(ok, total=len(scores), message=f"Boltz-2: {ok}/{len(scores)} kompleks")
        return PoseResult(
            engine=self.name,
            scores=scores,
            rows=rows,
            stage_info={"boltz2_scored": ok, "actual_pose_engine": "boltz2"},
            metadata={
                "actual_pose_engine": "boltz2",
                "cache_dir": str(self.cache_dir),
                "env_dir": str(self.env_dir),
                "recycling_steps": self.recycling_steps,
                "sampling_steps": self.sampling_steps,
                "msa_mode": "server" if self.use_msa_server else "single_sequence",
            },
        )
