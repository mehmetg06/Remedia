"""Prepare protein structures for GNINA and validate the resulting PDBQT."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ReceptorPreparationError(RuntimeError):
    """Raised when a receptor cannot be converted to a usable PDBQT file."""


def validate_receptor_pdbqt(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 100:
        raise ReceptorPreparationError(f"Reseptör PDBQT bulunamadı veya boş: {path}")
    atom_lines = [
        line for line in path.read_text(errors="ignore").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    if not atom_lines:
        raise ReceptorPreparationError(f"Reseptör PDBQT içinde atom kaydı yok: {path}")
    malformed = [line for line in atom_lines if len(line.split()) < 10]
    if len(malformed) == len(atom_lines):
        raise ReceptorPreparationError(f"Reseptör PDBQT atom tipleri/yükleri okunamıyor: {path}")
    return path


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stderr or proc.stdout or "").strip().splitlines()
    return " | ".join(text[-5:])


def prepare_receptor_pdbqt(
    receptor: str | Path,
    output_dir: str | Path | None = None,
    ph: float = 7.4,
    force: bool = False,
) -> Path:
    """Return a validated receptor PDBQT, preparing PDB/mmCIF input when needed.

    Open Babel is used because it is already available in Remedia environments.
    Hydrogens are added at the requested pH, Gasteiger charges are assigned, and
    the receptor is written as a rigid PDBQT file.
    """
    source = Path(receptor).expanduser().resolve()
    if not source.is_file():
        raise ReceptorPreparationError(f"Reseptör dosyası bulunamadı: {source}")
    if source.suffix.lower() == ".pdbqt":
        return validate_receptor_pdbqt(source)

    output_dir = Path(output_dir or source.parent / "prepared_receptors")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}_prepared.pdbqt"
    if output.exists() and not force:
        return validate_receptor_pdbqt(output)

    obabel = shutil.which("obabel")
    if not obabel:
        raise ReceptorPreparationError(
            "Open Babel (obabel) bulunamadı; reseptör PDBQT hazırlanamadı."
        )

    output.unlink(missing_ok=True)
    cmd = [
        obabel,
        str(source),
        "-O", str(output),
        "-xr",
        "-p", str(ph),
        "--partialcharge", "gasteiger",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        output.unlink(missing_ok=True)
        raise ReceptorPreparationError(
            f"Reseptör hazırlama başarısız (exit={proc.returncode}): {_tail(proc)}"
        )
    try:
        return validate_receptor_pdbqt(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
