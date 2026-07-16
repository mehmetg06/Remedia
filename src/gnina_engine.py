# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Batched GNINA fast/accurate screening for Remedia."""
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import import Path
from statistics import mean, median

from receptor_prep import ReceptorPreparationError, prepare_receptor_pdbqt

MODE_FAST = "fast"
MODE_ACCURATE = "accurate"
PROFILE_BALANCED = "balanced"
PROFILE_FINAL = "final"
KAYNAK_FAST = "gnina_fast"
KAYNAK_ACCURATE = "gnina_accurate"
DEFAULT_GNINA_PATH = "/usr/local/bin/gnina"

MODE_PROFILES = {
    PROFILE_BALANCED: {
        MODE_FAST: dict(cnn="fast", cnn_scoring="rescore", exhaustiveness=2, num_modes=1),
        MODE_ACCURATE: dict(cnn=None, cnn_scoring="rescore", exhaustiveness=4, num_modes=1),
    },
    PROFILE_FINAL: {
        MODE_FAST: dict(cnn="fast", cnn_scoring="rescore", exhaustiveness=4, num_modes=1),
        MODE_ACCURATE: dict(cnn=None, cnn_scoring="rescore", exhaustiveness=16, num_modes=9),
    },
}
MODE_FLAGS = MODE_PROFILES[PROFILE_BALANCED]


class GninaScreeningError(RuntimeError):
    """Raised when GNINA produces no scientifically usable score."""


@dataclass
class DockResult:
    ligand: str
    mode: str
    affinity_kcal_mol: float | None
    elapsed_seconds: float
    success: bool
    error: str | None = None
    out_path: str | None = None


def _flags(mode, profile):
    if profile not in MODE_PROFILES:
        raise ValueError(f"Bilinmeyen profil: {profile}")
    if mode not in (MODE_FAST, MODE_ACCURATE):
        raise ValueError(f"Bilinmeyen mod: {mode}")
    return MODE_PROFILES[profile][mode]


def build_gnina_command(gnina_path, receptor, ligand, center, size, mode=MODE_FAST,
                        out_path=None, seed=42, extra_args=None,
                        profile=PROFILE_BALANCED):
    flags = _flags(mode, profile)
    cmd = [str(gnina_path), "-r", str(receptor), "-l", str(ligand),
           "--center_x", str(center[0]), "--center_y", str(center[1]),
           "--center_z", str(center[2]), "--size_x", str(size[0]),
           "--size_y", str(size[1]), "--size_z", str(size[2]),
           "--cnn_scoring", flags["cnn_scoring"]]
    if flags["cnn"]:
        cmd += ["--cnn", flags["cnn"]]
    cmd += ["--exhaustiveness", str(flags["exhaustiveness"]),
            "--num_modes", str(flags["num_modes"]), "--seed", str(seed)]
    if out_path is not None:
        cmd += ["-o", str(out_path)]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def parse_affinity(out_path, stdout):
    out_path = Path(out_path) if out_path else None
    if out_path and out_path.exists():
        try:
            from rdkit import Chem
            for mol in Chem.SDMolSupplier(str(out_path), removeHs=False):
                if mol is None:
                    continue
                values = [float(mol.GetProp(k)) for k in
                          ("minimizedAffinity", "CNNaffinity", "affinity")
                          if mol.HasProp(k)]
                if values:
                    return min(values)
        except Exception:
            pass
    for line in (stdout or "").splitlines():
        match = re.match(r"\s*1\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1))
    return None


def parse_batch_affinities(path):
    from rdkit import Chem
    scores = {}
    for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
        if mol is None:
            continue
        name = next((mol.GetProp(k).strip() for k in
                     ("RemediaLigandName", "_Name", "Name", "ligand")
                     if mol.HasProp(k) and mol.GetProp(k).strip()),
                    f"mol_{index:03d}")
        value = next((float(mol.GetProp(k)) for k in
                      ("minimizedAffinity", "CNNaffinity", "affinity")
                      if mol.HasProp(k)), None)
        if value is not None and math.isfinite(value) and (name not in scores or value < scores[name]):
            scores[name] = value
    return scores


def prepare_ligand_sdf(smiles, name, out_dir):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol.SetProp("_Name", name)
    mol.SetProp("RemediaLigandName", name)
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.sdf"
    writer = Chem.SDWriter(str(path)); writer.write(mol); writer.close()
    return path


def prepare_ligand_library(molecules, out_dir, prepare_fn=prepare_ligand_sdf):
    prepared, failures = {}, {}
    for name, smiles in molecules:
        try:
            path = prepare_fn(smiles, name, out_dir)
        except Exception as exc:
            path = None
            failures[name] = f"{type(exc).__name__}: {exc}"
        if path is None:
            failures.setdefault(name, "geçersiz SMILES / 3D üretilemedi")
        else:
            prepared[name] = Path(path)
    return prepared, failures


def write_batch_sdf(prepared, output_path):
    from rdkit import Chem
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, written = Chem.SDWriter(str(output_path)), 0
    try:
        for name, path in prepared.items():
            for mol in Chem.SDMolSupplier(str(path), removeHs=False):
                if mol is None:
                    continue
                mol.SetProp("_Name", name)
                mol.SetProp("RemediaLigandName", name)
                writer.write(mol); written += 1; break
    finally:
        writer.close()
    if not written:
        output_path.unlink(missing_ok=True)
        raise ValueError("Batch SDF boş")
    return output_path


def _prepared_receptor(receptor, out_dir):
    path = Path(receptor)
    if not path.exists():
        return receptor
    return prepare_receptor_pdbqt(path, Path(out_dir) / "receptor")


def _proc_error(proc):
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return " | ".join(lines[-5:]) or f"GNINA exit={proc.returncode}"


def dock_batch_with_gnina(receptor, prepared_ligands, center, size, mode=MODE_FAST,
                          gnina_path=DEFAULT_GNINA_PATH, out_dir=Path("gnina_out"),
                          seed=42, extra_args=None, timeout=None,
                          profile=PROFILE_BALANCED):
    names = list(prepared_ligands)
    if not names:
        return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        receptor = _prepared_receptor(receptor, out_dir)
    except ReceptorPreparationError as exc:
        return [DockResult(n, mode, None, 0, False, str(exc)) for n in names]
    batch_in = write_batch_sdf(prepared_ligands, out_dir / f"{mode}_input.sdf")
    batch_out = out_dir / f"{mode}_docked.sdf"; batch_out.unlink(missing_ok=True)
    cmd = build_gnina_command(gnina_path, receptor, batch_in, center, size, mode,
                              batch_out, seed, extra_args, profile)
    flags = _flags(mode, profile)
    print(
        f"[GNINA] {mode}: {len(names)} ligand, "
        f"exhaustiveness={flags['exhaustiveness']}, "
        f"num_modes={flags['num_modes']}",
        flush=True,
    )
    print("[GNINA] Komut başlatıldı", flush=True)

    started = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            output_lines.append(line)
            print("[GNINA] " + line, flush=True)

        return_code = proc.wait(timeout=timeout)
        combined_output = "\n".join(output_lines)

        class ProcResult:
            pass

        result = ProcResult()
        result.returncode = return_code
        result.stdout = combined_output
        result.stderr = combined_output
        proc = result

    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        elapsed = time.time() - started
        return [DockResult(n, mode, None, elapsed, False, str(exc)) for n in names]

    elapsed = time.time() - started
    print(
        f"[GNINA] {mode} tamamlandı: {elapsed:.1f} saniye "
        f"({elapsed / len(names):.1f} sn/ligand)",
        flush=True,
    )
    each = elapsed / len(names)
    if proc.returncode != 0:
        batch_out.unlink(missing_ok=True)
        error = _proc_error(proc)
        return [DockResult(n, mode, None, each, False, error) for n in names]
    if not batch_out.exists():
        return [DockResult(n, mode, None, each, False, "GNINA çıktı üretmedi") for n in names]
    try:
        scores = parse_batch_affinities(batch_out)
    except Exception as exc:
        return [DockResult(n, mode, None, each, False, f"Skor ayrıştırma hatası: {exc}") for n in names]
    return [DockResult(n, mode, scores.get(n), each, n in scores,
                       None if n in scores else "GNINA çıktısında ligand skoru yok",
                       str(batch_out)) for n in names]


def dock_with_gnina(receptor, ligand_file, center, size, mode=MODE_FAST,
                    ligand_name=None, gnina_path=DEFAULT_GNINA_PATH, out_dir=None,
                    seed=42, extra_args=None, timeout=None,
                    profile=PROFILE_BALANCED):
    ligand_file = Path(ligand_file); name = ligand_name or ligand_file.stem
    out_dir = Path(out_dir or ligand_file.parent); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{mode}_docked.sdf"; out_path.unlink(missing_ok=True)
    try:
        receptor = _prepared_receptor(receptor, out_dir)
    except ReceptorPreparationError as exc:
        return DockResult(name, mode, None, 0, False, str(exc))
    cmd = build_gnina_command(gnina_path, receptor, ligand_file, center, size, mode,
                              out_path, seed, extra_args, profile)
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return DockResult(name, mode, None, time.time() - started, False, str(exc))
    elapsed = time.time() - started
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        return DockResult(name, mode, None, elapsed, False, _proc_error(proc))
    affinity = parse_affinity(out_path, proc.stdout)
    return DockResult(name, mode, affinity, elapsed, affinity is not None,
                      None if affinity is not None else "GNINA skoru okunamadı", str(out_path))


def select_top_candidates(results, top_n=None, top_fraction=0.10):
    scored = sorted((r for r in results if r.success and r.affinity_kcal_mol is not None),
                    key=lambda r: r.affinity_kcal_mol)
    if not scored:
        return []
    count = top_n if top_n is not None else max(1, math.ceil(len(scored) * top_fraction))
    return scored[:max(0, int(count))]


def _ordered_results(molecules, prepared, failures, mode, batch_results):
    found = {r.ligand: r for r in batch_results}
    return [found.get(name, DockResult(name, mode, None, 0, False,
                                      failures.get(name, "batch sonucu yok")))
            for name, _ in molecules]


def _rows(results, mode):
    source = KAYNAK_FAST if mode == MODE_FAST else KAYNAK_ACCURATE
    return [{"ligand": r.ligand, "affinity_kcal_mol": r.affinity_kcal_mol,
             "skor_kaynagi": source if r.success else None,
             "docking_success": r.success, "docking_error": r.error,
             "fast_affinity_kcal_mol": r.affinity_kcal_mol if mode == MODE_FAST else None,
             "accurate_affinity_kcal_mol": r.affinity_kcal_mol if mode == MODE_ACCURATE else None,
             "fast_seconds": round(r.elapsed_seconds, 3) if mode == MODE_FAST else None,
             "accurate_seconds": round(r.elapsed_seconds, 3) if mode == MODE_ACCURATE else None}
            for r in results]


def _raise_if_no_scores(results, stage):
    valid = [r for r in results if r.success and r.affinity_kcal_mol is not None]
    if valid:
        return
    errors = []
    for result in results:
        if result.error and result.error not in errors:
            errors.append(result.error)
    detail = " | ".join(errors[:3]) or "GNINA geçerli skor üretmedi"
    raise GninaScreeningError(f"{stage} docking başarısız: {detail}")


def run_single_mode_screening(molecules, receptor, center, size, mode=MODE_FAST,
                              gnina_path=DEFAULT_GNINA_PATH, out_dir=Path("gnina_out"),
                              seed=42, log_fn=print, extra_args=None, timeout=None,
                              prepare_fn=prepare_ligand_sdf,
                              batch_dock_fn=dock_batch_with_gnina,
                              profile=PROFILE_BALANCED):
    prepared, failures = prepare_ligand_library(molecules, Path(out_dir) / "prepared", prepare_fn)
    log_fn(f"[{mode.upper()}] {len(prepared)} ligand tek GNINA batch sürecinde")
    batch = batch_dock_fn(receptor, prepared, center, size, mode=mode,
                          gnina_path=gnina_path, out_dir=Path(out_dir) / mode,
                          seed=seed, extra_args=extra_args, timeout=timeout, profile=profile)
    results = _ordered_results(molecules, prepared, failures, mode, batch)
    _raise_if_no_scores(results, mode.upper())
    return _rows(results, mode), {mode: results, "prepared": prepared, "gnina_processes": 1}


def run_two_stage_screening(molecules, receptor, center, size,
                            gnina_path=DEFAULT_GNINA_PATH, out_dir=Path("gnina_out"),
                            top_n=None, top_fraction=0.10, seed=42, log_fn=print,
                            extra_args=None, timeout=None, prepare_fn=prepare_ligand_sdf,
                            batch_dock_fn=dock_batch_with_gnina,
                            profile=PROFILE_BALANCED):
    out_dir = Path(out_dir)
    prepared, failures = prepare_ligand_library(molecules, out_dir / "prepared", prepare_fn)
    log_fn(f"[1/2] FAST batch: {len(prepared)} ligand, tek GNINA süreci")
    fast_batch = batch_dock_fn(receptor, prepared, center, size, mode=MODE_FAST,
                               gnina_path=gnina_path, out_dir=out_dir / "fast", seed=seed,
                               extra_args=extra_args, timeout=timeout, profile=profile)
    fast = _ordered_results(molecules, prepared, failures, MODE_FAST, fast_batch)
    _raise_if_no_scores(fast, "FAST")
    top = select_top_candidates(fast, top_n, top_fraction)
    selected = {r.ligand: prepared[r.ligand] for r in top if r.ligand in prepared}
    log_fn(f"[2/2] ACCURATE batch ({profile}): {len(selected)} ligand, tek GNINA süreci")
    accurate = batch_dock_fn(receptor, selected, center, size, mode=MODE_ACCURATE,
                             gnina_path=gnina_path, out_dir=out_dir / "accurate", seed=seed,
                             extra_args=extra_args, timeout=timeout, profile=profile)
    acc_by_name = {r.ligand: r for r in accurate}
    rows = []
    for fr in fast:
        acc = acc_by_name.get(fr.ligand)
        use_acc = acc is not None and acc.success and acc.affinity_kcal_mol is not None
        rows.append({"ligand": fr.ligand,
                     "affinity_kcal_mol": acc.affinity_kcal_mol if use_acc else fr.affinity_kcal_mol,
                     "skor_kaynagi": KAYNAK_ACCURATE if use_acc else KAYNAK_FAST,
                     "docking_success": fr.success or use_acc,
                     "docking_error": None if (fr.success or use_acc) else fr.error,
                     "fast_affinity_kcal_mol": fr.affinity_kcal_mol,
                     "accurate_affinity_kcal_mol": acc.affinity_kcal_mol if acc else None,
                     "fast_seconds": round(fr.elapsed_seconds, 3),
                     "accurate_seconds": round(acc.elapsed_seconds, 3) if acc else None})
    return rows, {"fast": fast, "accurate": accurate,
                  "top_ligands": list(selected), "prepared": prepared,
                  "profile": profile, "gnina_processes": 1 + bool(selected)}


def benchmark_fast_vs_accurate(molecules, receptor, center, size,
                               gnina_path=DEFAULT_GNINA_PATH, out_dir=Path("benchmark"),
                               seed=42, log_fn=print, extra_args=None, timeout=None,
                               prepare_fn=prepare_ligand_sdf,
                               batch_dock_fn=dock_batch_with_gnina,
                               profile=PROFILE_BALANCED):
    prepared, _ = prepare_ligand_library(molecules, Path(out_dir) / "prepared", prepare_fn)
    common = dict(receptor=receptor, prepared_ligands=prepared, center=center, size=size,
                  gnina_path=gnina_path, seed=seed, extra_args=extra_args,
                  timeout=timeout, profile=profile)
    fast = batch_dock_fn(mode=MODE_FAST, out_dir=Path(out_dir) / "fast", **common)
    accurate = batch_dock_fn(mode=MODE_ACCURATE, out_dir=Path(out_dir) / "accurate", **common)
    _raise_if_no_scores(fast, "FAST benchmark")
    _raise_if_no_scores(accurate, "ACCURATE benchmark")
    f, a = {r.ligand: r for r in fast}, {r.ligand: r for r in accurate}
    rows = []
    for name in prepared:
        fr, ar = f.get(name), a.get(name)
        if not fr or not ar:
            continue
        gap = abs(fr.affinity_kcal_mol - ar.affinity_kcal_mol) if fr.success and ar.success else None
        ratio = ar.elapsed_seconds / fr.elapsed_seconds if fr.elapsed_seconds else None
        rows.append({"ligand": name, "fast_affinity_kcal_mol": fr.affinity_kcal_mol,
                     "accurate_affinity_kcal_mol": ar.affinity_kcal_mol,
                     "skor_farki": round(gap, 4) if gap is not None else None,
                     "fast_seconds": round(fr.elapsed_seconds, 3),
                     "accurate_seconds": round(ar.elapsed_seconds, 3),
                     "hiz_orani": round(ratio, 2) if ratio is not None else None})
    valid = [r for r in rows if r["skor_farki"] is not None]
    if not valid:
        return rows, {"n": len(rows), "n_karsilastirilabilir": 0}
    return rows, {"n": len(rows), "n_karsilastirilabilir": len(valid),
                  "fast_ortalama_sn": round(mean(r["fast_seconds"] for r in valid), 3),
                  "accurate_ortalama_sn": round(mean(r["accurate_seconds"] for r in valid), 3),
                  "hiz_orani_ortalama": round(mean(r["hiz_orani"] for r in valid), 2),
                  "skor_farki_ortalama": round(mean(r["skor_farki"] for r in valid), 4),
                  "skor_farki_medyan": round(median(r["skor_farki"] for r in valid), 4),
                  "skor_farki_maksimum": round(max(r["skor_farki"] for r in valid), 4)}


def _read_smi(path):
    molecules = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if not parts or line.lstrip().startswith("#"):
            continue
        molecules.append((parts[1] if len(parts) > 1 else f"mol_{len(molecules)+1}", parts[0]))
    return molecules


def _write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--ligands", required=True)
    parser.add_argument("--center", nargs=3, type=float, required=True)
    parser.add_argument("--size", nargs=3, type=float, default=[20, 20, 20])
    parser.add_argument("--mode", choices=["auto", "fast", "accurate", "compare"], default="auto")
    parser.add_argument("--profile", choices=["balanced", "final"], default="balanced")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--gnina-path", default=DEFAULT_GNINA_PATH)
    parser.add_argument("--out-dir", default="gnina_out")
    parser.add_argument("--output", default="results/docking_scores.csv")
    args = parser.parse_args()
    molecules = _read_smi(args.ligands)
    common = dict(receptor=args.receptor, center=args.center, size=args.size,
                  gnina_path=args.gnina_path, out_dir=Path(args.out_dir), profile=args.profile)
    if args.mode == "compare":
        rows, summary = benchmark_fast_vs_accurate(molecules, **common); print(summary)
    elif args.mode in (MODE_FAST, MODE_ACCURATE):
        rows, _ = run_single_mode_screening(molecules, mode=args.mode, **common)
    else:
        rows, _ = run_two_stage_screening(molecules, top_n=args.top_n,
                                          top_fraction=args.top_fraction, **common)
    _write_csv(args.output, rows)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    main()
