# Complete ntops TTIR benchmark report

This report covers all collected test specializations from 76 ntops operators. LHS is the original Ninetoothed launch; RHS is the intercepted original reference callable/FX graph lowered automatically through TorchInductor.

## Numeric testing

- Operators completed: 76 / 76
- Tests completed: 2102 / 2102
- Tests actually executed: 1934
- Pass: 1822
- Fail: 112
- Skip: 168
- Operators with failures: 3 (addmv, conv2d, msort)

| Numeric failure class | Tests |
| --- | ---: |
| Ninetoothed/Triton mask broadcasting compile failure | 108 |
| Ninetoothed/Triton sort reshape compile failure | 2 |
| Ninetoothed/Triton tt.dot minimum-dimension compile failure | 2 |

## TTIR capture

- LHS statuses: `{"captured": 1773, "compile_failed": 110, "not_captured": 203, "not_reached": 16}`
- RHS statuses: `{"captured": 1782, "no_triton_kernel": 90, "not_reached": 16, "reference_compile_failed": 16, "reference_not_intercepted": 198}`
- Both sides captured: 1664 tests across 68 operators
- RHS compiled-vs-FX self-check on two-sided captures: `{"mismatch_or_unavailable": 27, "pass": 1637}`

## PairSpec generation

- Cases generated: 1011
- Cases unavailable: 1091
- Generated output observations: 1230
- Recovered-feature cases: `{"incomplete_pointer_mapping": 300, "inferred_single_store_output": 188, "multiple_references": 228, "multiple_references_fully_generated": 192, "multiple_stores": 146, "nonzero_endpoint_offset": 13, "tuple_output": 27, "unequal_pointer_abi": 231}`

| Why formal verification was not reached | Cases |
| --- | ---: |
| RHS reference has zero or multiple kernels/launches | 438 |
| LHS TTIR is absent or has multiple launches | 325 |
| stores cannot be associated with output leaves | 136 |
| RHS TTIR is absent | 101 |
| reference output has no verifiable tensor leaf | 48 |
| captured RHS failed compiled-vs-FX self-check | 27 |
| test did not reach the capture hook | 16 |

Partial cases may still have generated observations. Their failed reference attempts are counted separately:

| Failed reference-attempt class | Attempts |
| --- | ---: |
| stores cannot be associated with output leaves | 36 |

## Formal verification

- ETV invocations: 1230 output observations across 1011 tests
- Structured results: 1230
- Conclusive results: 7 across 7 tests
- Observation statuses: `{"DISPROVED": 4, "NOT_RUN": 1091, "PROVED": 3, "UNKNOWN": 1223}`
- DISPROVED observations: 4

| UNKNOWN/runner reason | Observations |
| --- | ---: |
| TTIR_REGION_SEMANTICS_UNSUPPORTED | 848 |
| TTIR_OP_UNSUPPORTED | 236 |
| TTIR_UNDEFINED_LOAD_LANE | 56 |
| TTIR_MEMORY_TYPE_UNSUPPORTED | 36 |
| MISSING_ROLE | 27 |
| CAST_EQUIVALENCE_NOT_REWRITTEN | 12 |
| TTIR_CONSTANT_UNSUPPORTED | 8 |

### DISPROVED observations

- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[10-0-dtype0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[10-0-dtype1]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[5-0-dtype0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[5-0-dtype1]::r000-e000-o000`: MASK_MISMATCH

All four are `diag_1d` with `n` equal to 5 or 10 and diagonal 0. The original CUDA tests passed. The LHS wrapper initializes the matrix with `torch.zeros` and its captured kernel writes only the diagonal, while the RHS kernel writes every element. The current kernel-only PairSpec does not model that host-side initialization, so these are real store-effect counterexamples under the PairSpec but not evidence that the original operator outputs differ.

## Per-operator results

Formal columns count output observations, so tuple outputs and multiple reference expressions can contribute more than one observation per test.

| Operator | Tests | Pass | Fail | Skip | Exact | Pair cases | Specs | P | D | U | E | T | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abs | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| acosh | 38 | 38 | 0 | 0 | 31 | 31 | 33 | 0 | 0 | 33 | 0 | 0 | 7 |
| adaptive_avg_pool2d | 6 | 6 | 0 | 0 | 6 | 6 | 12 | 0 | 0 | 12 | 0 | 0 | 0 |
| adaptive_max_pool2d | 4 | 4 | 0 | 0 | 4 | 4 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| add | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| addmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| addmv | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| alpha_dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| argsort | 60 | 60 | 0 | 0 | 60 | 35 | 35 | 0 | 0 | 35 | 0 | 0 | 25 |
| atan | 17 | 17 | 0 | 0 | 15 | 15 | 15 | 0 | 0 | 15 | 0 | 0 | 2 |
| avg_pool2d | 36 | 18 | 0 | 18 | 18 | 18 | 36 | 0 | 0 | 36 | 0 | 0 | 18 |
| batch_norm | 32 | 32 | 0 | 0 | 24 | 4 | 8 | 0 | 0 | 8 | 0 | 0 | 28 |
| bincount | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| bitwise_and | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bitwise_not | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bitwise_or | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| celu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| clamp | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| conv2d | 108 | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 108 |
| cos | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| cosh | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| diag | 60 | 60 | 0 | 0 | 46 | 46 | 46 | 3 | 4 | 39 | 0 | 0 | 14 |
| div | 24 | 16 | 0 | 8 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 8 |
| dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| eq | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| exp | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| fmax | 10 | 10 | 0 | 0 | 10 | 10 | 10 | 0 | 0 | 10 | 0 | 0 | 0 |
| ge | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| gelu | 16 | 8 | 0 | 8 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 8 |
| gt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| instance_norm | 384 | 384 | 0 | 0 | 384 | 84 | 132 | 0 | 0 | 132 | 0 | 0 | 300 |
| isinf | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| isnan | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| layer_norm | 96 | 96 | 0 | 0 | 96 | 84 | 166 | 0 | 0 | 166 | 0 | 0 | 12 |
| le | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| logsumexp | 52 | 52 | 0 | 0 | 46 | 46 | 48 | 0 | 0 | 48 | 0 | 0 | 6 |
| lp_pool1d | 12 | 12 | 0 | 0 | 11 | 11 | 20 | 0 | 0 | 20 | 0 | 0 | 1 |
| lp_pool2d | 24 | 24 | 0 | 0 | 24 | 24 | 41 | 0 | 0 | 41 | 0 | 0 | 0 |
| lp_pool3d | 24 | 24 | 0 | 0 | 24 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 8 |
| lt | 8 | 8 | 0 | 0 | 7 | 7 | 7 | 0 | 0 | 7 | 0 | 0 | 1 |
| matmul | 8 | 8 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| max | 24 | 24 | 0 | 0 | 24 | 23 | 38 | 0 | 0 | 38 | 0 | 0 | 1 |
| max_pool1d | 18 | 18 | 0 | 0 | 12 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 6 |
| max_pool2d | 108 | 54 | 0 | 54 | 54 | 54 | 54 | 0 | 0 | 54 | 0 | 0 | 54 |
| max_pool3d | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| maximum | 11 | 11 | 0 | 0 | 11 | 11 | 11 | 0 | 0 | 11 | 0 | 0 | 0 |
| mean | 25 | 25 | 0 | 0 | 25 | 24 | 24 | 0 | 0 | 24 | 0 | 0 | 1 |
| median | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| mm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| msort | 8 | 6 | 2 | 0 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 2 |
| mul | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| ne | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| neg | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| pow | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| quantile | 160 | 80 | 0 | 80 | 80 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 148 |
| relu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| rms_norm | 64 | 64 | 0 | 0 | 64 | 55 | 55 | 0 | 0 | 55 | 0 | 0 | 9 |
| rot90 | 32 | 32 | 0 | 0 | 32 | 32 | 32 | 0 | 0 | 32 | 0 | 0 | 0 |
| rotary_position_embedding | 128 | 128 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 128 |
| round | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| rsqrt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| scaled_dot_product_attention | 78 | 78 | 0 | 0 | 54 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 66 |
| select_copy | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| sgn | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| sigmoid | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| sign | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| signbit | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| silu | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| sin | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| softmax | 8 | 8 | 0 | 0 | 8 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 3 |
| sort | 36 | 36 | 0 | 0 | 28 | 12 | 24 | 0 | 0 | 24 | 0 | 0 | 24 |
| stack | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| sub | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| tanh | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 |
| threshold | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |

Individual node IDs, PairSpecs, pointer/alias diagnostics, raw reasons, and ETV reports remain in each case directory and the JSON summaries.
