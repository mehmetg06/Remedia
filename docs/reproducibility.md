# Reproducibility (Phase 9)

Every Remedia run is designed to be **reproducible and auditable**. This page
explains what is recorded, where, and how to reproduce a result.

## What is recorded

Each run's `00_REMEDIA_REPORT/run_manifest.json` contains a `reproducibility`
block (produced by `src/reproducibility.py::capture_run_metadata`) with:

| Field | Meaning |
|---|---|
| `git.commit`, `git.branch`, `git.dirty` | Exact code version; `dirty=true` warns of uncommitted changes |
| `random_seed` | Pipeline random seed (default **42**; GNINA docking seed) |
| `seed_molecules` | Seed SMILES (known ligands) used for generation |
| `generator`, `pose_engine` | Which components ran (e.g. `reinvent4` / `gnina`) |
| `parameters` | The complete `settings` dict (molecule count, profile, box, top fraction, …) |
| `software` | Python + platform + versions of RDKit, torch, numpy, pandas, REINVENT, … |
| `tools.gnina` | GNINA binary version string |

The manifest also records the generator provenance (`generation_manifest`,
including per-source counts for Hybrid runs), pocket center, and the ranked top
candidates with their explanations.

## How to reproduce a run

1. **Check out the exact code**: `git checkout <git.commit>`. If `git.dirty` was
   `true`, the recorded commit does not fully capture the code — avoid dirty runs
   for anything you intend to publish.
2. **Recreate the environment**: install the versions listed under `software`
   (see `modal/requirements.txt` / `requirements.txt`). Use the same GNINA
   release (`tools.gnina`).
3. **Re-run with the same inputs**: use the same UniProt ID, `random_seed`, and
   the `parameters` from the manifest (generator, pose engine, molecule count,
   profile, box size, top fraction).

Given the same code, environment, seed, and target structure, the pipeline
produces the same candidate set and scores.

## Sources of variation to control

- **Random seeds.** GNINA uses `--seed 42`. REINVENT4 and MolMIM accept a
  `seed`; set it for bit-exact generation. Record any seed you change.
- **External data drift.** AlphaFold structures, ChEMBL known ligands, and the
  MolMIM/DiffDock hosted endpoints can change over time. The manifest timestamps
  the run; for publication, archive the fetched receptor `.pdb`, the
  `generated.smi`, and the `diffdock_results.csv` alongside the manifest.
- **GPU/library nondeterminism.** Deep-learning components (REINVENT, MolMIM,
  DiffDock) may vary slightly across GPU/driver/library versions; pin them and
  record versions (captured automatically).
- **Model versions.** MolMIM/DiffDock NIM model versions are part of the
  endpoint; note the endpoint URL (recorded in each generator/pose `metadata`).

## Minimum record to publish a result

Commit SHA · UniProt ID · random seed · pocket center · generator + pose engine ·
molecule count + profile + box + top fraction · software/tool versions ·
`generated.smi` · `remedia_ranking.csv` · `run_manifest.json`.

All of these are captured automatically in the result package; keep the ZIP.
