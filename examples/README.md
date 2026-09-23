# Acceptance Artifacts

Examples use raw TTIR and strict PairSpec v3. Manifests and provenance distinguish
physical launches from store components. Counts below include both sides.

| Pair | Expected result | Launches / stores |
| --- | --- | --- |
| `add/pair.json` | `PROVED`, real fixed Add | 2 / 2 |
| `add/pair_parametric.json` | `PROVED`, bounded 2D/1D Add with `a*b=c` | 2 / 2 |
| `compute_mismatch/pair.json` | `DISPROVED(COMPUTE_MISMATCH)` | 2 / 2 |
| `mask_mismatch/pair.json` | `DISPROVED(MASK_MISMATCH)` | 2 / 2 |
| `address_mismatch/pair.json` | `DISPROVED(ADDRESS_MISMATCH)` | 2 / 2 |
| `multilaunch/pair.json` | `PROVED`, fused versus scale then increment | 3 / 3 |
| `unsupported/pair.json` | `UNKNOWN(TTIR_REGION_UNSUPPORTED)` | 2 / 2 |
| `real/argsort/pair.json` | `UNKNOWN(TTIR_REGION_UNSUPPORTED)` | 3 / 3 |
| `real/rms_norm/pair.json` | `UNKNOWN(TTIR_REGION_UNSUPPORTED)` | 3 / 3 |
| `real/batch_norm/pair.json` | `UNKNOWN(TTIR_REGION_UNSUPPORTED)` | 3 / 4 |
| `real/quantile/pair.json` | `UNKNOWN(TTIR_REGION_UNSUPPORTED)` | 3 / 4 |

Real multi-launch artifacts preserve original hashes, internal storage edges and
audited grouping. Unsupported computations are not approximated to improve results.
ABI and disjointness remain declared premises. `rules/fadd_zero.json` demonstrates
external rules; `generators/add.py` is an optional CUDA capture driver. Running
saved cases requires neither that driver nor its benchmark dependencies.
