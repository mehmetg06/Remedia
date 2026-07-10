# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
gnina_engine.py
GNINA docking motoru — iki net mod: Fast Screening (varsayılan) ve Accurate Mode.

FAST SCREENING (mode="fast", varsayılan):
    --cnn fast --cnn_scoring rescore --exhaustiveness 4 --num_modes 1
    Amaç: çok sayıda adayı hızlıca elemek. Skorlar kaba tahmindir, nihai
    karar için KULLANILMAZ.

ACCURATE MODE (mode="accurate"):
    --cnn_scoring rescore --exhaustiveness 16 --num_modes 9
    (--cnn BELİRTİLMEZ → GNINA'nın varsayılan çok-modelli ensemble'ı kullanılır)
    Amaç: fast taramadan geçen küçük bir kısayolun (top-N) güvenilir şekilde
    yeniden skorlanması. Nihai sıralama BU skorları kullanır.

Otomatik iki-aşamalı akış (run_two_stage_screening):
    1) TÜM adaylar FAST modda docklanır.
    2) En iyi top_n / top_fraction ACCURATE modda TEKRAR docklanır.
    3) Nihai skor: accurate'e girenler için accurate skoru, girmeyenler için
       fast skoru (yalnızca eleme amaçlı, "skor_kaynagi" sütunuyla işaretli).

Bu modül, önceden yalnızca Colab notebook'u içinde satır içi (inline) kod
olarak yaşayan GNINA çağrısını tek, test edilebilir bir yere taşır; notebook
artık bu modülü import edip çağırır (bkz. notebooks/remedia_pipeline.ipynb
Hücre 5). validate_top_candidates.py'nin Vina için yaptığı "top adayları
yeniden dockla" işini, GNINA akışında AYRI bir script yerine bu pipeline'ın
doğal bir parçası olarak yapar.

CLI kullanımı:
    python src/gnina_engine.py \\
        --receptor data/P00918_alphafold.pdb \\
        --ligands data/generated.smi \\
        --center 1.51 17.76 -0.97 --size 20 20 20 \\
        --mode auto --top-fraction 0.15 \\
        --output results/docking_scores.csv

    --mode auto      (varsayılan) iki-aşamalı otomatik akış
    --mode fast      yalnızca fast screening (hızlı keşif için yeterli olabilir)
    --mode accurate  yalnızca accurate mode (TÜM adaylar accurate ile dockla)
    --mode compare   AYNI molekül setini hem fast hem accurate ile dockla,
                     süre ve skor farkını raporla (gerçek ölçüm / benchmark)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

MODE_FAST = "fast"
MODE_ACCURATE = "accurate"

# skor_kaynagi sütununda kullanılan değerler (rank_report.py / generate_dashboard.py
# zaten bilinmeyen değerleri de tolere eder — bkz. molecule_generator.py'deki
# "real_docking" / "qed_fallback" konvansiyonu).
KAYNAK_FAST = "gnina_fast"
KAYNAK_ACCURATE = "gnina_accurate"

DEFAULT_GNINA_PATH = "/usr/local/bin/gnina"

# Her mod için GNINA CLI bayrakları. accurate modda "cnn": None → --cnn HİÇ
# verilmez, GNINA varsayılan (3 modelli) ensemble'ı kullanır.
MODE_FLAGS = {
    MODE_FAST: {
        "cnn": "fast",
        "cnn_scoring": "rescore",
        "exhaustiveness": 4,
        "num_modes": 1,
    },
    MODE_ACCURATE: {
        "cnn": None,
        "cnn_scoring": "rescore",
        "exhaustiveness": 16,
        "num_modes": 9,
    },
}


def build_gnina_command(
    gnina_path,
    receptor,
    ligand,
    center,
    size,
    mode=MODE_FAST,
    out_path=None,
    seed=42,
    extra_args=None,
):
    """Verilen moda göre GNINA komut satırını inşa eder (subprocess'e verilecek liste)."""
    if mode not in MODE_FLAGS:
        raise ValueError(f"Bilinmeyen mod: {mode!r} — 'fast' ya da 'accurate' olmalı")
    flags = MODE_FLAGS[mode]

    cmd = [
        str(gnina_path),
        "-r", str(receptor),
        "-l", str(ligand),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--cnn_scoring", flags["cnn_scoring"],
    ]
    if flags["cnn"] is not None:
        cmd += ["--cnn", flags["cnn"]]
    cmd += [
        "--exhaustiveness", str(flags["exhaustiveness"]),
        "--num_modes", str(flags["num_modes"]),
        "--seed", str(seed),
    ]
    if out_path is not None:
        cmd += ["-o", str(out_path)]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def parse_affinity(out_path, stdout):
    """En iyi (1. sıra) pozun affinity'sini kcal/mol olarak döndürür.
    Önce çıktı SDF'indeki property'lere (rdkit varsa), olmazsa stdout
    tablosundaki 1. satıra bakar. rdkit kurulu değilse SDF adımı sessizce
    atlanır — bu modül rdkit'siz de import edilebilir olsun diye."""
    out_path = Path(out_path) if out_path else None
    if out_path and out_path.exists():
        try:
            from rdkit import Chem  # local import: rdkit opsiyonel bağımlılık

            supp = Chem.SDMolSupplier(str(out_path), removeHs=False)
            for m in supp:
                if m is None:
                    continue
                for key in ("minimizedAffinity", "CNNaffinity", "affinity"):
                    if m.HasProp(key):
                        return float(m.GetProp(key))
                break
        except ImportError:
            pass
        except Exception:
            pass
    for line in (stdout or "").splitlines():
        m = re.match(r"\s*1\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1))
    return None


@dataclass
class DockResult:
    """Tek bir ligand + tek bir mod için docking sonucu."""

    ligand: str
    mode: str
    affinity_kcal_mol: float | None
    elapsed_seconds: float
    success: bool
    error: str | None = None
    out_path: str | None = None


def dock_with_gnina(
    receptor,
    ligand_file,
    center,
    size,
    mode=MODE_FAST,
    ligand_name=None,
    gnina_path=DEFAULT_GNINA_PATH,
    out_dir=None,
    seed=42,
    extra_args=None,
    timeout=None,
) -> DockResult:
    """Hazırlanmış TEK bir ligand dosyasını (SDF) GNINA ile dockler.

    Args:
        receptor:    Reseptör dosyası (.pdb/.pdbqt).
        ligand_file: Hazırlanmış ligand dosyası (RDKit ile 3D'ye getirilmiş .sdf).
        center:      Docking kutusu merkezi (x, y, z).
        size:        Docking kutusu boyutu (sx, sy, sz).
        mode:        "fast" (varsayılan, hızlı eleme) veya "accurate" (nihai karar).
        ligand_name: Sonuçtaki ligand ismi; verilmezse dosya adından türetilir.
        gnina_path:  GNINA binary'sinin yolu.
        out_dir:     Docklanmış pozun (out SDF) yazılacağı dizin.
        seed:        Rastgelelik tohumu (tekrarlanabilirlik için sabit).
        extra_args:  GNINA'ya eklenecek ek CLI argümanları.
        timeout:     Saniye cinsinden zaman aşımı (None = sınırsız).

    Returns:
        DockResult — başarılıysa affinity_kcal_mol dolu, değilse error dolu.
    """
    ligand_file = Path(ligand_file)
    name = ligand_name or ligand_file.stem
    out_dir = Path(out_dir) if out_dir else ligand_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{mode}_docked.sdf"

    cmd = build_gnina_command(
        gnina_path, receptor, ligand_file, center, size,
        mode=mode, out_path=out_path, seed=seed, extra_args=extra_args,
    )

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return DockResult(name, mode, None, time.time() - t0, False, "zaman aşımı", None)
    except FileNotFoundError as e:
        return DockResult(name, mode, None, time.time() - t0, False,
                           f"GNINA çalıştırılamadı: {e}", None)
    elapsed = time.time() - t0

    aff = parse_affinity(out_path, proc.stdout)
    if aff is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        neden = " | ".join(tail[-2:]) if tail else "bilinmeyen hata"
        return DockResult(name, mode, None, elapsed, False, neden, str(out_path))
    return DockResult(name, mode, aff, elapsed, True, None, str(out_path))


def prepare_ligand_sdf(smiles, name, out_dir):
    """SMILES'ı GNINA'nın okuyacağı 3D SDF'e çevirir (ligand_prep.py ile aynı
    mantık: RDKit ile 3D konformasyon üret + MMFF ile optimize et).
    rdkit gerektirir — çağıran taraf ImportError'ı ele almalı."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf = out_dir / f"{name}.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(mol)
    w.close()
    return sdf


def select_top_candidates(fast_results, top_n=None, top_fraction=0.15):
    """Fast sonuçlarından en iyi (en negatif affinity) top_n / top_fraction'ı seçer.

    top_n verilmişse doğrudan kullanılır; verilmemişse
    ceil(len(basarili) * top_fraction) uygulanır (en az 1).
    """
    scored = [r for r in fast_results if r.success and r.affinity_kcal_mol is not None]
    scored.sort(key=lambda r: r.affinity_kcal_mol)  # en negatif = en güçlü, başta
    if not scored:
        return []
    if top_n is None:
        top_n = max(1, math.ceil(len(scored) * top_fraction))
    return scored[: max(0, top_n)]


def run_single_mode_screening(
    molecules,
    receptor,
    center,
    size,
    mode=MODE_FAST,
    gnina_path=DEFAULT_GNINA_PATH,
    out_dir=Path("gnina_out"),
    seed=42,
    log_fn=print,
    extra_args=None,
    timeout=None,
    prepare_fn=prepare_ligand_sdf,
):
    """Manuel tek-mod akışı: TÜM molekülleri yalnızca `mode` ile dockla (iki
    aşamalı otomatik akışın DIŞINDA — hızlı bir keşif için sadece fast, ya da
    tüm setin nihai kararı için sadece accurate çalıştırmak istendiğinde).

    Returns:
        rows: run_two_stage_screening ile AYNI sütun şemasında (fast_* /
        accurate_* sütunlarından yalnızca çalıştırılan mod dolu olur).
    """
    kaynak = KAYNAK_FAST if mode == MODE_FAST else KAYNAK_ACCURATE
    mode_dir = Path(out_dir) / mode
    log_fn(f"[{mode.upper()}] {len(molecules)} molekül docklanacak...")

    results: list[DockResult] = []
    for name, smiles in molecules:
        sdf = prepare_fn(smiles, name, mode_dir)
        if sdf is None:
            log_fn(f"  ❌ {name}: geçersiz SMILES / 3D üretilemedi — atlandı")
            results.append(DockResult(name, mode, None, 0.0, False, "3D üretilemedi"))
            continue
        r = dock_with_gnina(
            receptor, sdf, center, size, mode=mode, ligand_name=name,
            gnina_path=gnina_path, out_dir=mode_dir, seed=seed,
            extra_args=extra_args, timeout=timeout,
        )
        status = f"{r.affinity_kcal_mol:.2f} kcal/mol" if r.success else f"başarısız — {r.error}"
        log_fn(f"  {'✅' if r.success else '❌'} {name}: {status}  ({r.elapsed_seconds:.1f} sn)")
        results.append(r)

    rows = [{
        "ligand": r.ligand,
        "affinity_kcal_mol": r.affinity_kcal_mol,
        "skor_kaynagi": kaynak,
        "fast_affinity_kcal_mol": r.affinity_kcal_mol if mode == MODE_FAST else None,
        "accurate_affinity_kcal_mol": r.affinity_kcal_mol if mode == MODE_ACCURATE else None,
        "fast_seconds": round(r.elapsed_seconds, 3) if mode == MODE_FAST else None,
        "accurate_seconds": round(r.elapsed_seconds, 3) if mode == MODE_ACCURATE else None,
    } for r in results]
    return rows, {mode: results}


def run_two_stage_screening(
    molecules,
    receptor,
    center,
    size,
    gnina_path=DEFAULT_GNINA_PATH,
    out_dir=Path("gnina_out"),
    top_n=None,
    top_fraction=0.15,
    seed=42,
    log_fn=print,
    extra_args=None,
    timeout=None,
    prepare_fn=prepare_ligand_sdf,
):
    """Otomatik iki-aşamalı GNINA pipeline'ı.

    1) TÜM molekülleri FAST modda dockla (hızlı eleme).
    2) En iyi top_n / top_fraction'ı ACCURATE modda TEKRAR dockla.
    3) Nihai satırı üret: accurate ile yeniden docklanmış adaylarda
       affinity_kcal_mol = accurate skoru; diğerlerinde fast skoru
       (skor_kaynagi ile hangisi olduğu işaretlenir; nihai SIRALAMA yalnızca
       accurate skorlara güvenmelidir — fast skorlar sadece eleme amaçlıdır).

    Args:
        molecules: [(isim, smiles), ...]
        prepare_fn: SMILES → SDF dönüştürücü (varsayılan prepare_ligand_sdf;
            testlerde rdkit'siz çalışabilmek için mock'lanabilir).

    Returns:
        (rows, stage_info) — rows: rank_report.py'nin okuyabileceği dict listesi
        (ligand, affinity_kcal_mol, skor_kaynagi, fast_affinity_kcal_mol,
        accurate_affinity_kcal_mol, fast_seconds, accurate_seconds).
        stage_info: {"fast": [DockResult...], "accurate": [DockResult...],
                     "top_ligands": [isim, ...]}
    """
    out_dir = Path(out_dir)
    fast_dir = out_dir / "fast"
    accurate_dir = out_dir / "accurate"

    # --- AŞAMA 1: TÜM adayları FAST modda dockla ----------------------------
    log_fn(f"[AŞAMA 1/2] FAST screening — {len(molecules)} molekül...")
    fast_results: list[DockResult] = []
    for name, smiles in molecules:
        sdf = prepare_fn(smiles, name, fast_dir)
        if sdf is None:
            log_fn(f"  ❌ {name}: geçersiz SMILES / 3D üretilemedi — atlandı")
            fast_results.append(DockResult(name, MODE_FAST, None, 0.0, False, "3D üretilemedi"))
            continue
        r = dock_with_gnina(
            receptor, sdf, center, size, mode=MODE_FAST, ligand_name=name,
            gnina_path=gnina_path, out_dir=fast_dir, seed=seed,
            extra_args=extra_args, timeout=timeout,
        )
        status = f"{r.affinity_kcal_mol:.2f} kcal/mol" if r.success else f"başarısız — {r.error}"
        log_fn(f"  {'✅' if r.success else '❌'} {name}: {status}  ({r.elapsed_seconds:.1f} sn)")
        fast_results.append(r)

    # --- AŞAMA 2: En iyi top-N'i ACCURATE modda TEKRAR dockla ---------------
    top = select_top_candidates(fast_results, top_n=top_n, top_fraction=top_fraction)
    top_names = {r.ligand for r in top}
    smiles_by_name = {name: smi for name, smi in molecules}
    log_fn(f"\n[AŞAMA 2/2] ACCURATE re-dock — en iyi {len(top)}/{len(fast_results)} "
           f"aday (top_n={top_n}, top_fraction={top_fraction})...")

    accurate_results: list[DockResult] = []
    for r in top:
        smiles = smiles_by_name.get(r.ligand)
        sdf = prepare_fn(smiles, r.ligand, accurate_dir)
        if sdf is None:
            log_fn(f"  ❌ {r.ligand}: 3D üretilemedi (accurate) — atlandı")
            accurate_results.append(DockResult(r.ligand, MODE_ACCURATE, None, 0.0, False, "3D üretilemedi"))
            continue
        acc = dock_with_gnina(
            receptor, sdf, center, size, mode=MODE_ACCURATE, ligand_name=r.ligand,
            gnina_path=gnina_path, out_dir=accurate_dir, seed=seed,
            extra_args=extra_args, timeout=timeout,
        )
        status = f"{acc.affinity_kcal_mol:.2f} kcal/mol" if acc.success else f"başarısız — {acc.error}"
        log_fn(f"  {'✅' if acc.success else '❌'} {r.ligand}: fast={r.affinity_kcal_mol:.2f} → "
               f"accurate={status}  ({acc.elapsed_seconds:.1f} sn)")
        accurate_results.append(acc)

    accurate_by_name = {r.ligand: r for r in accurate_results}

    # --- Nihai satırları birleştir ------------------------------------------
    rows = []
    for fr in fast_results:
        acc = accurate_by_name.get(fr.ligand)
        if acc is not None and acc.success:
            final_aff = acc.affinity_kcal_mol
            kaynak = KAYNAK_ACCURATE
        else:
            final_aff = fr.affinity_kcal_mol
            kaynak = KAYNAK_FAST
        rows.append({
            "ligand": fr.ligand,
            "affinity_kcal_mol": final_aff,
            "skor_kaynagi": kaynak,
            "fast_affinity_kcal_mol": fr.affinity_kcal_mol,
            "accurate_affinity_kcal_mol": acc.affinity_kcal_mol if acc else None,
            "fast_seconds": round(fr.elapsed_seconds, 3),
            "accurate_seconds": round(acc.elapsed_seconds, 3) if acc else None,
        })

    stage_info = {"fast": fast_results, "accurate": accurate_results, "top_ligands": sorted(top_names)}
    return rows, stage_info


def benchmark_fast_vs_accurate(
    molecules,
    receptor,
    center,
    size,
    gnina_path=DEFAULT_GNINA_PATH,
    out_dir=Path("gnina_benchmark"),
    seed=42,
    log_fn=print,
    extra_args=None,
    timeout=None,
    prepare_fn=prepare_ligand_sdf,
):
    """AYNI molekül setini hem FAST hem ACCURATE modda dockler ve gerçek süre/skor
    farkını ölçer (bkz. görev #4 — 'gerçekten ölç').

    Returns:
        (rows, summary)
        rows: her molekül için {ligand, fast_affinity_kcal_mol, accurate_affinity_kcal_mol,
              skor_farki, fast_seconds, accurate_seconds, hiz_orani}
        summary: {n, fast_ortalama_sn, accurate_ortalama_sn, hiz_orani_ortalama,
                  skor_farki_ortalama, skor_farki_medyan, skor_farki_maksimum}
    """
    out_dir = Path(out_dir)
    rows = []
    for name, smiles in molecules:
        sdf_fast = prepare_fn(smiles, name, out_dir / "fast")
        sdf_acc = prepare_fn(smiles, name, out_dir / "accurate")
        if sdf_fast is None or sdf_acc is None:
            log_fn(f"  ❌ {name}: 3D üretilemedi — karşılaştırmaya dahil edilmedi")
            continue

        fast = dock_with_gnina(receptor, sdf_fast, center, size, mode=MODE_FAST,
                                ligand_name=name, gnina_path=gnina_path, out_dir=out_dir / "fast",
                                seed=seed, extra_args=extra_args, timeout=timeout)
        accurate = dock_with_gnina(receptor, sdf_acc, center, size, mode=MODE_ACCURATE,
                                    ligand_name=name, gnina_path=gnina_path, out_dir=out_dir / "accurate",
                                    seed=seed, extra_args=extra_args, timeout=timeout)

        skor_farki = (
            abs(fast.affinity_kcal_mol - accurate.affinity_kcal_mol)
            if fast.success and accurate.success else None
        )
        hiz_orani = (
            accurate.elapsed_seconds / fast.elapsed_seconds
            if fast.elapsed_seconds > 0 else None
        )
        log_fn(
            f"  {name}: fast={fast.affinity_kcal_mol} ({fast.elapsed_seconds:.1f} sn)  "
            f"accurate={accurate.affinity_kcal_mol} ({accurate.elapsed_seconds:.1f} sn)  "
            f"Δskor={skor_farki}  hız_oranı={hiz_orani}"
        )
        rows.append({
            "ligand": name,
            "fast_affinity_kcal_mol": fast.affinity_kcal_mol,
            "accurate_affinity_kcal_mol": accurate.affinity_kcal_mol,
            "skor_farki": round(skor_farki, 4) if skor_farki is not None else None,
            "fast_seconds": round(fast.elapsed_seconds, 3),
            "accurate_seconds": round(accurate.elapsed_seconds, 3),
            "hiz_orani": round(hiz_orani, 2) if hiz_orani is not None else None,
        })

    valid = [r for r in rows if r["skor_farki"] is not None]
    if valid:
        summary = {
            "n": len(rows),
            "n_karsilastirilabilir": len(valid),
            "fast_ortalama_sn": round(mean(r["fast_seconds"] for r in valid), 3),
            "accurate_ortalama_sn": round(mean(r["accurate_seconds"] for r in valid), 3),
            "hiz_orani_ortalama": round(mean(r["hiz_orani"] for r in valid), 2),
            "skor_farki_ortalama": round(mean(r["skor_farki"] for r in valid), 4),
            "skor_farki_medyan": round(median(r["skor_farki"] for r in valid), 4),
            "skor_farki_maksimum": round(max(r["skor_farki"] for r in valid), 4),
        }
    else:
        summary = {"n": len(rows), "n_karsilastirilabilir": 0}
    return rows, summary


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_smi_file(path):
    mols = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol_{len(mols) + 1}"
        mols.append((name, smiles))
    return mols


def main():
    parser = argparse.ArgumentParser(
        description="GNINA docking motoru — fast/accurate mod ve iki-aşamalı otomatik pipeline"
    )
    parser.add_argument("--receptor", required=True, help="Reseptör dosyası (.pdb/.pdbqt)")
    parser.add_argument("--ligands", required=True, help="Ligand .smi dosyası")
    parser.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, default=[20, 20, 20], metavar=("SX", "SY", "SZ"))
    parser.add_argument(
        "--mode", choices=["auto", "fast", "accurate", "compare"], default="auto",
        help="auto=iki-aşamalı otomatik akış (varsayılan) · fast/accurate=yalnızca o mod · "
             "compare=aynı seti iki modda da dockla ve farkı raporla (benchmark)",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Accurate re-dock için en iyi N aday")
    parser.add_argument("--top-fraction", type=float, default=0.15,
                         help="--top-n verilmezse kullanılacak oran (varsayılan 0.15 = top %%15)")
    parser.add_argument("--gnina-path", default=DEFAULT_GNINA_PATH)
    parser.add_argument("--out-dir", default="gnina_out")
    parser.add_argument("--output", default="results/docking_scores.csv")
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()

    molecules = _parse_smi_file(args.ligands)
    if args.max_molecules:
        molecules = molecules[: args.max_molecules]

    common = dict(
        receptor=args.receptor, center=args.center, size=args.size,
        gnina_path=args.gnina_path, out_dir=Path(args.out_dir),
    )

    if args.mode == "compare":
        rows, summary = benchmark_fast_vs_accurate(molecules, **common)
        _write_csv(args.output, rows, [
            "ligand", "fast_affinity_kcal_mol", "accurate_affinity_kcal_mol",
            "skor_farki", "fast_seconds", "accurate_seconds", "hiz_orani",
        ])
        print(f"\n[OK] Karşılaştırma sonuçları: {args.output}")
        print(summary)
        return

    if args.mode == "fast" or args.mode == "accurate":
        rows, _ = run_single_mode_screening(molecules, mode=args.mode, **common)
        _write_csv(args.output, rows, [
            "ligand", "affinity_kcal_mol", "skor_kaynagi",
            "fast_affinity_kcal_mol", "accurate_affinity_kcal_mol",
            "fast_seconds", "accurate_seconds",
        ])
        print(f"\n[OK] {args.mode} mod sonuçları: {args.output}")
        return

    # auto = iki aşamalı
    rows, stage_info = run_two_stage_screening(
        molecules, top_n=args.top_n, top_fraction=args.top_fraction, **common,
    )
    _write_csv(args.output, rows, [
        "ligand", "affinity_kcal_mol", "skor_kaynagi",
        "fast_affinity_kcal_mol", "accurate_affinity_kcal_mol",
        "fast_seconds", "accurate_seconds",
    ])
    print(f"\n[OK] İki-aşamalı sonuçlar: {args.output}")
    print(f"     {len(stage_info['fast'])} molekül fast tarandı, "
          f"{len(stage_info['top_ligands'])} tanesi accurate ile yeniden docklandı.")


if __name__ == "__main__":
    main()
