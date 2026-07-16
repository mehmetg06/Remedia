# DiffDock pose engine setup (Phase 5)

Remedia can predict binding poses with **DiffDock** (deep-learning) alongside
**GNINA** (physics-based docking). You choose the engine in the web UI:

```
Pose Engine
○ GNINA              (default — unchanged, always available)
○ DiffDock           (confidence-based pose prediction)
○ Hybrid Validation  (DiffDock screen → GNINA confirmation on top candidates)
```

> GNINA is the default and is **never removed**. DiffDock and Hybrid Validation
> are additive. If DiffDock is not available, Remedia falls back to GNINA so a
> run is never blocked.

## Reusing the existing groundwork

This integration **reuses the DiffDock code already in the repository**
(`src/merge_diffdock_results.py`) instead of rebuilding it. That module parses a
`diffdock_results.csv` (`ligand, diffdock_confidence, ...`) and merges DiffDock
confidence with GNINA/Vina affinity using the "genel güven" (overall-confidence)
rule. `DiffDockPredictor` turns that same CSV into engine-agnostic pose scores,
and `HybridValidationPredictor` applies the same overall-confidence merge.

## Providing DiffDock results

`DiffDockPredictor` obtains per-ligand confidences from the first available of:

1. **An injected runner** — a callable producing `{ligand: confidence}` (or a CSV
   path). Use this to plug in a live DiffDock backend.
2. **An existing `diffdock_results.csv`** — the repository's established flow
   (e.g. DiffDock run on Colab). Drop the file into the run directory
   (`Remedia_results/run_*/diffdock_results.csv`) and select DiffDock / Hybrid.
3. **The NVIDIA DiffDock NIM** — credential-gated (documented, not run by
   default):
   - Endpoint: `https://health.api.nvidia.com/v1/biology/mit/diffdock`
     (override with `DIFFDOCK_BASE_URL` for a self-hosted NIM).
   - Key from `DIFFDOCK_API_KEY` (falls back to `NVIDIA_API_KEY`, `NGC_API_KEY`,
     `NVCF_RUN_KEY`).

If none is available, `DiffDockPredictor.predict_pose` raises
`DiffDockUnavailable` and the pipeline falls back to GNINA.

## Hybrid Validation

```
DiffDock (all molecules)
        │  rank by confidence
        ▼
Top candidates (top_fraction, default 25%)
        │
        ▼
GNINA confirmation (physics score on the top subset)
        │  merge_diffdock_results.genel_guven()
        ▼
GÜÇLÜ ADAY / TEK YÖNTEMLE DESTEKLENİYOR / ZAYIF ADAY
```

A candidate that both DiffDock and GNINA rank strongly is labelled `GÜÇLÜ ADAY`
(strong). This two-method agreement is more trustworthy than either method alone.

## Programmatic use

```python
from pose import build_pose_predictor
pred = build_pose_predictor("hybrid", gnina_path="/usr/local/bin/gnina",
                            diffdock_results_csv="run_x/diffdock_results.csv")
result = pred.predict_pose(molecules, receptor=receptor, center=center, size=(20, 20, 20))
for s in result.scores:
    print(s.ligand, s.affinity_kcal_mol, s.confidence, s.extra.get("genel_guven_durumu"))
```

Reference: <https://build.nvidia.com/mit/diffdock>
