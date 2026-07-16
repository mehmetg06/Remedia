# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""DiffDock pose predictor behind the :class:`BasePosePredictor` interface.

Confidence scores are obtained, in order, from an injected runner, an existing
``diffdock_results.csv``, or NVIDIA's hosted/self-hosted DiffDock NIM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .base import BasePosePredictor, PoseResult, PoseScore

# Hosted NVIDIA API. Self-hosted users can set DIFFDOCK_BASE_URL to the full
# /molecular-docking/diffdock/generate endpoint.
DEFAULT_DIFFDOCK_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
API_KEY_ENV_VARS = (
    "DIFFDOCK_API_KEY",
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
    "NVCF_RUN_KEY",
)


class DiffDockUnavailable(RuntimeError):
    """Raised when DiffDock cannot provide scores."""


def diffdock_api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _smiles_to_sdf(smiles: str, name: str) -> str:
    """Create one 3D SDF record accepted by DiffDock NIM."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DiffDockUnavailable(f"RDKit yüklenemedi: {exc}") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DiffDockUnavailable(f"Geçersiz SMILES: {name}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    if status != 0:
        raise DiffDockUnavailable(f"DiffDock için 3B ligand hazırlanamadı: {name}")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    mol.SetProp("_Name", name)
    return Chem.MolToMolBlock(mol) + "\n$$$$\n"


def _endpoint(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/molecular-docking/diffdock/generate"):
        return url
    # Hosted build.nvidia.com endpoint uses the legacy full route above; a
    # self-hosted base URL is normally host:port and needs the documented path.
    if url.endswith("/diffdock") or "/v1/biology/" in url:
        return url
    return url + "/molecular-docking/diffdock/generate"


class DiffDockPredictor(BasePosePredictor):
    """Deep-learning pose prediction (confidence scores) via DiffDock."""

    name = "diffdock"

    def __init__(
        self,
        *,
        results_csv: str | Path | None = None,
        runner: Callable[..., Any] | None = None,
        loader: Callable[[Path], dict[str, float | None]] | None = None,
        base_url: str | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        self.results_csv = Path(results_csv) if results_csv else None
        self._runner = runner
        self._loader = loader
        self.base_url = base_url or os.environ.get("DIFFDOCK_BASE_URL", DEFAULT_DIFFDOCK_URL)
        self._log = log_fn
        self.last_source = "unavailable"

    def _nim_confidences(
        self,
        molecules: list[tuple[str, str]],
        receptor: str | None,
        out_dir: Any,
        reporter: Any,
    ) -> dict[str, float | None]:
        import requests

        key = diffdock_api_key()
        if not key:
            raise DiffDockUnavailable("DIFFDOCK_API_KEY bulunamadı")
        if not receptor or not Path(receptor).is_file():
            raise DiffDockUnavailable(f"DiffDock reseptör PDB dosyasını bulamadı: {receptor}")

        protein = Path(receptor).read_text(encoding="utf-8", errors="ignore")
        endpoint = _endpoint(self.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        output_dir = Path(out_dir or ".") / "diffdock_poses"
        output_dir.mkdir(parents=True, exist_ok=True)
        confidences: dict[str, float | None] = {}

        for index, (name, smiles) in enumerate(molecules, 1):
            if reporter is not None:
                reporter.log(f"DiffDock NIM: {name} ({index}/{len(molecules)})")
            payload = {
                "ligand": _smiles_to_sdf(smiles, name),
                "ligand_file_type": "sdf",
                "protein": protein,
                "num_poses": 1,
                "time_divisions": 20,
                "steps": 18,
                "save_trajectory": False,
                "is_staged": False,
            }
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
            except requests.RequestException as exc:
                raise DiffDockUnavailable(f"DiffDock NIM bağlantı hatası: {exc}") from exc
            if not response.ok:
                detail = response.text.replace("\n", " ")[:600]
                raise DiffDockUnavailable(
                    f"DiffDock NIM HTTP {response.status_code}: {detail}"
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise DiffDockUnavailable("DiffDock NIM geçersiz JSON döndürdü") from exc
            if str(data.get("status", "success")).lower() not in {"success", "ok"}:
                raise DiffDockUnavailable(
                    f"DiffDock NIM başarısız: {data.get('details') or data}"
                )
            scores = data.get("position_confidence") or data.get("pose_confidence") or []
            confidence = float(scores[0]) if scores else None
            confidences[name] = confidence
            poses = data.get("ligand_positions") or data.get("docked_ligand") or []
            if isinstance(poses, str):
                poses = [poses]
            if poses:
                (output_dir / f"{name}_rank01.sdf").write_text(
                    str(poses[0]), encoding="utf-8"
                )

        self.last_source = "nvidia_nim"
        return confidences

    def _confidences(
        self,
        molecules: list[tuple[str, str]],
        receptor: str | None,
        center: Any,
        size: Any,
        out_dir: Any,
        reporter: Any,
        **kwargs: Any,
    ) -> dict[str, float | None]:
        if self._runner is not None:
            result = self._runner(
                molecules=molecules,
                receptor=receptor,
                center=center,
                size=size,
                out_dir=out_dir,
                **kwargs,
            )
            self.last_source = "runner"
            if isinstance(result, (str, Path)):
                return self._load_csv(Path(result))
            return dict(result or {})

        csv_path = self._find_results_csv(out_dir)
        if csv_path is not None:
            if reporter is not None:
                reporter.log(f"DiffDock sonuçları okunuyor: {csv_path}")
            self.last_source = "csv"
            return self._load_csv(csv_path)

        if diffdock_api_key():
            return self._nim_confidences(molecules, receptor, out_dir, reporter)
        raise DiffDockUnavailable(
            "DiffDock için runner, diffdock_results.csv veya DIFFDOCK_API_KEY bulunamadı"
        )

    def _find_results_csv(self, out_dir: Any) -> Path | None:
        candidates = []
        if self.results_csv is not None:
            candidates.append(self.results_csv)
        if out_dir is not None:
            candidates.append(Path(out_dir) / "diffdock_results.csv")
            candidates.append(Path(out_dir).parent / "diffdock_results.csv")
        for path in candidates:
            if path and Path(path).exists():
                return Path(path)
        return None

    def _load_csv(self, path: Path) -> dict[str, float | None]:
        if self._loader is not None:
            return self._loader(path)
        from merge_diffdock_results import load_diffdock

        return load_diffdock(path)

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
        if reporter is not None:
            reporter.log("DiffDock poz tahmini başlatılıyor")
        try:
            confidences = self._confidences(
                molecules, receptor, center, size, out_dir, reporter, **kwargs
            )
        except DiffDockUnavailable as exc:
            message = f"DiffDock kullanılamadı; GNINA fallback uygulanacak. Neden: {exc}"
            self._log(message)
            if reporter is not None:
                reporter.log(message)
            raise

        scores: list[PoseScore] = []
        for name, _smiles in molecules:
            conf = confidences.get(name)
            scores.append(
                PoseScore(
                    ligand=name,
                    affinity_kcal_mol=None,
                    confidence=conf,
                    success=conf is not None,
                    source="diffdock",
                    error=None if conf is not None else "DiffDock skoru yok",
                )
            )
        rows = [s.to_row() for s in scores]
        ok = sum(1 for s in scores if s.success)
        if reporter is not None:
            reporter.update(ok, total=len(scores), message=f"DiffDock: {ok}/{len(scores)} poz")
        return PoseResult(
            engine=self.name,
            scores=scores,
            rows=rows,
            stage_info={"diffdock_scored": ok, "actual_pose_engine": "diffdock"},
            metadata={
                "endpoint": _endpoint(self.base_url),
                "source": self.last_source,
                "source_csv": str(self._find_results_csv(out_dir) or ""),
                "actual_pose_engine": "diffdock",
            },
        )
