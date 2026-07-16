# NVIDIA MolMIM setup (Phase 4)

Remedia can generate molecules with **NVIDIA MolMIM** alongside REINVENT4.
MolMIM is served as an NVIDIA NIM (NVIDIA Inference Microservice) `/generate`
endpoint that explores chemical space around a seed molecule.

> MolMIM is **optional** and **additive**. REINVENT4 remains the default and the
> benchmarking baseline. If MolMIM is not configured, the `molmim` and `hybrid`
> generators fall back or raise a clear, non-fatal error — the existing pipeline
> keeps working unchanged.

## 1. Choose a deployment

### A. Hosted (build.nvidia.com)
- Endpoint: `https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate`
- Requires an API key (`nvapi-…`) from <https://build.nvidia.com> (create an
  account, open the MolMIM model, click **Get API Key**).
- The key is sent as `Authorization: Bearer <key>`.

### B. Self-hosted NIM container
- Pull and run the MolMIM NIM (needs an NVIDIA GPU + NGC access):
  ```bash
  docker run --rm --gpus all -p 8000:8000 \
    nvcr.io/nim/nvidia/molmim:latest
  ```
- Endpoint: `http://localhost:8000/generate` (no API key required).

## 2. Configure Remedia

Set environment variables (nothing is hard-coded in the repository):

| Variable | Purpose | Default |
|---|---|---|
| `MOLMIM_API_KEY` | API key for the hosted endpoint (falls back to `NVIDIA_API_KEY`, `NGC_API_KEY`, `NVCF_RUN_KEY`) | _unset_ |
| `MOLMIM_BASE_URL` | Override the endpoint (use this for a self-hosted NIM) | `https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate` |

Examples:

```bash
# Hosted
export MOLMIM_API_KEY="nvapi-xxxxxxxx"

# Self-hosted
export MOLMIM_BASE_URL="http://localhost:8000/generate"
```

On Modal, add `MOLMIM_API_KEY` as a Secret and expose it to the `run_job`
function (e.g. `modal.Secret.from_name("molmim")`).

## 3. Request parameters

The client (`src/generators/molmim_config.py`) exposes MolMIM's documented
parameters with safe defaults and range-clamping:

| Field | Default | Range | Meaning |
|---|---|---|---|
| `algorithm` | `CMA-ES` | `CMA-ES` / `none` | optimisation strategy |
| `property_name` | `QED` | `QED` / `plogP` | property being optimised |
| `minimize` | `false` | bool | minimise vs. maximise the property |
| `min_similarity` | `0.3` | 0.0–0.7 | keep results similar to the seed |
| `particles` | `30` | 2–1000 | CMA-ES particles |
| `iterations` | `10` | 1–1000 | CMA-ES iterations |
| `scaled_radius` | `1.0` | 0.0–2.0 | sampling radius |
| `num_molecules` | `n` | 1–100 | per-call cap |

## 4. Use it

In the web UI, pick the **Generator**: `REINVENT4`, `MolMIM`, or `Hybrid`
(50 % REINVENT4 + 50 % MolMIM, pools merged). Programmatically:

```python
from generators import build_generator
gen = build_generator("molmim")            # or "reinvent4" / "hybrid"
result = gen.generate(target="P00918", n=20, seeds=known_ligand_smiles)
print(result.smiles, result.per_source_counts())
```

## 5. Notes

- MolMIM is **seed-conditioned**: it needs at least one seed SMILES (Remedia uses
  the known ligands fetched from ChEMBL). If a target has no known ligands, use
  REINVENT4 (which samples seed-free) or Hybrid (which degrades to REINVENT4).
- The client is async-capable (`agenerate`), retries transient failures with
  exponential backoff, and enforces a per-request timeout.
- Reference: <https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html>
