# Benchmark protocol (Phase 9)

This protocol defines how Remedia components are compared so results are fair,
repeatable, and publishable. It is implemented by `src/benchmark.py` and should
be run before changing any default (generator or pose engine).

> **Guiding rule:** never switch a default until benchmarks show the new
> component is **equal or better** on the metrics below. REINVENT4 and GNINA
> remain the baselines and are never removed.

## What is compared

1. **Generators** — REINVENT4 (baseline) vs MolMIM vs Hybrid (and optionally the
   heuristic generators).
2. **Pose engines** — GNINA (baseline) vs DiffDock vs Hybrid Validation.

Each comparison holds everything else fixed (same target, same seeds, same
candidate set for pose engines) so only the component under test varies.

## Metrics

### Generators
| Metric | Definition | Better |
|---|---|---|
| `runtime_seconds` | wall-clock for `generate(target, n)` | lower |
| `produced` | valid, de-duplicated molecules returned | higher (≥ n) |
| `uniqueness_ratio` | unique / produced | higher |
| `unique_scaffolds` | distinct Murcko scaffolds (absolute) | higher |
| `diversity_score` | unique scaffolds / molecules | higher |
| `admet_pass_rate` | fraction passing Lipinski/Veber | higher |

### Pose engines
| Metric | Definition | Better |
|---|---|---|
| `runtime_seconds` | wall-clock for `predict_pose(molecules)` | lower |
| `success_rate` | scored / total | higher |
| `best_affinity` | most-negative GNINA affinity (kcal/mol) | more negative |
| `mean_affinity` | mean over scored | more negative |
| `mean_confidence` | mean DiffDock confidence | higher |

`runtime` is the throughput axis (the bottleneck V3 targets); `diversity` and
`docking quality` are the scientific-quality axes; `ADMET pass rate` is the
developability axis.

## Procedure

1. **Fix inputs.** Choose a target (UniProt), fetch its structure and pocket,
   fetch known ligands as seeds. Record the commit SHA and seed (see
   `docs/reproducibility.md`).
2. **Generators.** Request the same `n` from each generator on the same seeds:
   ```python
   from generators import build_generator
   from benchmark import run_generator_benchmark
   gens = {name: build_generator(name) for name in ("reinvent4", "molmim", "hybrid")}
   report = run_generator_benchmark(gens, target=uniprot, n=50, seeds=seeds)
   report.export("Remedia_results/benchmark")
   ```
3. **Pose engines.** Dock the *same* molecule set with each engine:
   ```python
   from pose import build_pose_predictor
   from benchmark import run_pose_benchmark
   preds = {name: build_pose_predictor(name, gnina_path=gnina)
            for name in ("gnina", "diffdock", "hybrid")}
   report = run_pose_benchmark(preds, molecules, receptor=receptor,
                               center=center, size=(20, 20, 20))
   report.export("Remedia_results/benchmark")
   ```
4. **Repeat** across several targets and ≥3 seeds; report mean ± spread.
5. **Export.** `BenchmarkReport.export()` writes `benchmark_<kind>.csv/json/md`.
   Keep them with the run manifests.

## Fairness controls

- Identical seeds, target structure, pocket box, and molecule count across
  compared components.
- Same ADMET filter and diversity definition for all generators.
- Pose engines score the **identical** molecule set.
- A component that requires credentials (MolMIM/DiffDock) is benchmarked when
  configured; if unavailable it is recorded with an `error` and excluded from
  the winner selection rather than silently skewing results.
- Report runtime on comparable hardware (note GPU type).

## Interpreting results

- Prefer the component that is **equal or better** on quality metrics without a
  disproportionate runtime cost.
- A single strong docking score is not proof of binding; agreement between
  independent methods (Hybrid Validation "GÜÇLÜ ADAY") is stronger evidence.
- Always report the negative/limitation cases, not only the wins.
