# Remedia Architecture (V2 baseline → V3 modernization)

> **Phase 1 deliverable.** This document describes the *current* Remedia pipeline
> as it exists in the repository today. It is an analysis document only — it does
> **not** change any functionality. Later modernization phases (generator
> abstraction, MolMIM, DiffDock, composite ranking, richer result packages,
> benchmarking) are described here only as *replacement points*, so the team knows
> where new components will attach without removing anything that works.

## 1. High-level flow (as implemented today)

```
UniProt ID
    │
    ▼
fetch_structure.fetch_alphafold()      → receptor .pdb from AlphaFold DB (PDB fallback)
    │
    ▼
pocket_detection.best_druggable_pocket() → binding-pocket center (fpocket, cached)
    │
    ▼
known_ligands.fetch_known_ligands()    → seed SMILES from ChEMBL (PubChem fallback)
    │
    ▼
Generator (one of):                    → candidate SMILES
    • REINVENT4 sampling  (generative_model.generate_with_reinvent)   ["pretrained"]
    • fusion / genetic / brics / random (molecule_generator.*)        [seed-based]
    │
    ▼
GNINA docking                          → affinity (kcal/mol) per molecule
    gnina_engine.run_two_stage_screening() (default: fast pass → accurate pass)
    │
    ▼
ADMET filter                           → Lipinski/Veber drug-likeness
    admet_filter.lipinski_veber_filter()
    │
    ▼
Ranking                                → final_ranking.csv
    rank_report.py (ADMET-pass first, then affinity ascending)
    │
    ▼
Result package                         → annotated HTML report + CSV + manifest + ZIP
    result_report.build_result_package()
```

The V3 target keeps every box above and inserts/branches at two points:

```
Generator Layer   = REINVENT4 + NVIDIA MolMIM + Hybrid   (choice, not replacement)
Pose Layer        = DiffDock + GNINA + Hybrid Validation  (choice, not replacement)
Ranking           = composite "Final Remedia Score"       (adds to, not replaces, docking sort)
```

## 2. Where each concern lives (map of the code)

| Concern | Primary module(s) | Notes |
|---|---|---|
| **Orchestration** | `notebooks/remedia_modal.ipynb` → `_run_pipeline(settings)` | The real pipeline body. The Modal web worker executes this cell code, not a `src/` entrypoint. |
| **Web UI + Modal worker** | `modal/remedia_web_v2.py` | Single-page FastAPI app (`web`) + GPU job function (`run_job`). Progress via `_ProgressStream`. This is the current deployed surface. |
| **Earlier web variants** | `modal/remedia_web.py`, `modal/remedia_modal.py` | Older/alternate launchers kept for reference. `remedia_web_v2.py` is authoritative. |
| **Receptor fetch** | `src/fetch_structure.py` | `fetch_alphafold`, `fetch_pdb`. |
| **Pocket detection** | `src/pocket_detection.py` | `best_druggable_pocket` (fpocket + geometric fallback), cached in `pocket_cache.json`. |
| **Seed ligands** | `src/known_ligands.py` | ChEMBL REST, PubChem fallback. |
| **Generator — REINVENT4** | `src/generative_model.py` | `install_reinvent`, `generate_with_reinvent` (sampling from a prior; no fine-tuning/RL). |
| **Generator — heuristic** | `src/molecule_generator.py` | `fusion_generation`, `genetic_algorithm`, `brics_recombination`, `random_mutation`, `write_smi`. Large module; also contains a self-contained docking/validation path. |
| **Docking engine (GNINA)** | `src/gnina_engine.py` | `run_two_stage_screening`, `run_single_mode_screening`, `benchmark_fast_vs_accurate`, batch SDF prep, affinity parsing, accuracy profiles. |
| **ADMET** | `src/admet_filter.py` | `lipinski_veber_filter` (offline, RDKit), plus optional `admetlab_filter` (network API). |
| **Ranking** | `src/rank_report.py` | Merges docking + ADMET CSVs, sorts. |
| **Result package** | `src/result_report.py` | `build_result_package` → `00_REMEDIA_REPORT/` HTML + normalized candidate CSV + manifest; schema-tolerant (discovers files dynamically). |
| **DiffDock groundwork (already present)** | `src/merge_diffdock_results.py` | Merges a `diffdock_results.csv` with Vina/GNINA scores into a "genel güven" (overall-confidence) table. Not yet wired into `_run_pipeline`. |
| **Docking self-validation** | `src/validate_top_candidates.py`, `src/cross_validate_docking.py` | Re-dock top hits at higher exhaustiveness to catch low-exhaustiveness artifacts. |
| **Dashboards / reporting extras** | `src/generate_dashboard.py`, `src/dashboard_template.html`, `src/receptor_prep.py`, `src/ligand_prep.py`, `src/docking.py` | Supporting utilities. |
| **Deployment surfaces** | `modal/`, `runpod/`, `Dockerfile.local`, `scripts/setup_local.sh`, `environment.yml` | Modal (primary hosted), RunPod, local GPU, Colab. |
| **Tests** | `tests/test_gnina_engine.py`, `tests/test_gnina_failure_handling.py`, `tests/test_modal_assets.py`, `tests/test_local_assets.py` | GNINA logic + asset/syntax checks; no GPU required. |

## 3. Data contract between stages (the format everything downstream expects)

Keeping this stable is the core constraint for adding new generators and pose
engines. Downstream code must not need to know *which* generator or *which* pose
engine produced a row.

- **Molecule list**: `list[tuple[name, smiles]]`, e.g. `("mol_001", "CC(=O)...")`.
  Also persisted as a `.smi` file via `molecule_generator.write_smi`
  (`generated.smi` in the run directory).
- **Docking rows**: list of dicts written to `docking_scores.csv`. Ranking reads
  the `ligand` and `affinity_kcal_mol` columns (plus optional `skor_kaynagi`
  = score source).
- **ADMET rows**: dicts from `lipinski_veber_filter` → `admet_results.csv`
  (`ligand`, `pass`, `MW`, `LogP`, `violations`, …).
- **Ranking output**: `final_ranking.csv` (`ligand`, `affinity_kcal_mol`,
  `admet_pass`, …).
- **Run directory**: `Remedia_results/run_YYYYMMDD_HHMMSS/` on the persistent
  volume. `result_report.build_result_package` discovers files here dynamically,
  so extra columns/files are additive-safe.

**Implication for V3:** a new `MolMIMGenerator` or `DiffDockPredictor` is
"format-compatible" if it produces the same molecule list / docking-row / CSV
shapes above. That is the invariant to protect.

## 4. Progress & observability (current state → Phase 2 target)

Current progress is inferred by **string-matching stdout** inside
`modal/remedia_web_v2.py::_ProgressStream.write`. It maps log substrings
("pocket", "reinvent", "gnina] fast", "admet", …) to a percentage and one of 5
steps, committed to a per-job JSON on the Modal volume and polled by the UI.

Limitations this creates (the Phase 2 backlog):
- **Coupling to log wording.** Progress breaks silently if a log string changes,
  and it is Turkish-phrase specific.
- **Coarse stages.** The UI shows `step/5` and a heuristic percent, not
  "Generated 12/20 molecules" / "Docked 8/20". There are no true item counts.
- **GNINA percent is time-extrapolated**, not work-based (`_gnina_percent()`
  advances on elapsed seconds).
- **Errors** are captured (`{job_id}.error.txt` + `technical_excerpt`) and shown
  in the UI — this part is reasonably good and should be preserved/extended, not
  regressed. Stack traces are already surfaced rather than hidden.

Phase 2 replacement point: introduce structured progress events
(stage name + `done/total` counts) emitted by the pipeline itself, and have the
worker consume those instead of scraping stdout — while keeping the existing
stdout scraping as a fallback so nothing breaks during transition.

## 5. Bottlenecks (why V3 exists)

1. **GNINA is the throughput bottleneck.** Docking dominates runtime; the recent
   git history is almost entirely GNINA speed tuning (CPU counts, batch SDF,
   two-stage fast→accurate, throttled progress commits). This is the motivation
   for adding a **DiffDock** pose layer (fast, GPU deep-learning pose prediction)
   while keeping GNINA for physics-based confirmation.
2. **Single generator per run.** `_run_pipeline` picks exactly one `method`.
   There is no way to pool candidates from multiple generators — the motivation
   for the **Generator abstraction + Hybrid mode** (REINVENT4 + MolMIM).
3. **Ranking is docking-only.** `rank_report.py` sorts by ADMET-pass then
   affinity. Pose confidence, drug-likeness score, and diversity are not folded
   into a single score — the motivation for the **composite Final Remedia Score**.
4. **Orchestration lives in a notebook cell.** `_run_pipeline` is defined in
   `remedia_modal.ipynb` and `exec`'d by the web worker via `_load_pipeline()`.
   This makes the true entrypoint hard to test and hard to extend. A gentle,
   non-breaking direction is to move shared logic into importable `src/` modules
   that the notebook cell *calls*, so the notebook stays thin and the logic
   becomes unit-testable — without changing behavior.
5. **Progress is stdout-scraped** (see §4).

## 6. Replacement points for later phases (no code changes in Phase 1)

These are the seams where new components attach *alongside* the existing ones.
Nothing here is implemented yet; this section is the map for later PRs.

| Phase | New seam | Attaches at | Existing code preserved |
|---|---|---|---|
| 3 | `BaseGenerator.generate(target, n)` | Wraps the `method_value` branch in `_run_pipeline` | REINVENT4 becomes `ReinventGenerator`; heuristic methods stay callable |
| 4 | `MolMIMGenerator` (async, retry, timeout, logging) | New `BaseGenerator` impl; output via `write_smi` format | REINVENT4 untouched; Hybrid merges both pools |
| 5 | `BasePosePredictor.predict_pose()`, `DiffDockPredictor`, `GninaPredictor` | Wraps the `gnina_engine` call in `_run_pipeline` | `gnina_engine` kept as `GninaPredictor`; `merge_diffdock_results.py` feeds Hybrid Validation |
| 6 | Composite `Final Remedia Score` | Extends `rank_report.py` (new columns, new sort key) | Docking-based sort remains available as a component |
| 7 | `README_FIRST.txt`, `report.html`, `candidate_overview.csv`, `pipeline_log.txt`, `run_manifest.json` | Extends `result_report.build_result_package` | Current report is already schema-tolerant; additions are safe |
| 8 | Benchmark mode | Extends `gnina_engine.benchmark_fast_vs_accurate` pattern to generators + pose engines | Existing benchmark helper kept |
| 9 | `docs/benchmark_protocol.md`, `docs/reproducibility.md` | New docs; capture seeds/versions/params already partially available in manifest | — |

## 7. Invariants to protect during modernization

1. **REINVENT4 is never removed.** It becomes one `BaseGenerator` implementation
   and remains the benchmarking baseline.
2. **GNINA is never removed.** It becomes one `BasePosePredictor` implementation
   and the physics-based confirmation step in Hybrid Validation.
3. **The stage-to-stage data contract in §3 stays stable.** New generators/pose
   engines conform to it; downstream code stays generator/engine-agnostic.
4. **The pipeline stays runnable at every commit.** New engines are opt-in
   (a `settings` choice) and default to the current behavior until benchmarks
   justify changing defaults.
5. **Errors stay visible.** Never hide stack traces; keep and extend the existing
   `technical_log` / `technical_excerpt` mechanism.

## 8. Test & verification entry points

- `python -m unittest discover -s tests -v` (no GPU/GNINA binary required).
- `bash -n scripts/setup_local.sh` (shell syntax).
- New modules added in later phases should ship with unit tests that exercise the
  §3 data contract with fakes/mocks (no GPU), following the existing
  `test_gnina_engine.py` style.
