# Complete ntops TTIR benchmark report

This report covers every collected test specialization from all 76 ntops operators. LHS is the original Ninetoothed launch; RHS is the original test reference expression lowered automatically through FX and TorchInductor.

## Numeric testing

- Operators completed: 76 / 76
- Tests classified: 2102 / 2102
- Tests actually executed (skip excluded): 1934
- Pass: 1822
- Fail: 112
- Skip: 168
- Operators with failures: 3 (addmv, conv2d, msort)

| Failure class | Tests |
| --- | ---: |
| Ninetoothed/Triton compile failure: incompatible mask broadcasting shapes (108 cases) | 108 |
| Ninetoothed/Triton compile failure: sort reshape receives a dtype without numel (2 cases) | 2 |
| Ninetoothed/Triton compile failure: tt.dot requires M, N, K >= 16 (2 cases) | 2 |

## TTIR capture

- LHS statuses: `{"captured": 1773, "compile_failed": 110, "not_captured": 203, "not_reached": 16}`
- RHS statuses: `{"captured": 1782, "no_triton_kernel": 90, "not_reached": 16, "reference_compile_failed": 16, "reference_not_intercepted": 198}`
- Both LHS and RHS TTIR captured: 1664 tests across 68 operators
- Exact-pair RHS compiled-vs-FX self-check: 1637 pass, 27 mismatch
- Operators without any two-sided TTIR pair: addmm, addmv, bincount, bmm, conv2d, median, mm, select_copy

RHS acquisition limits: 90 references completed without launching a Triton kernel; 16 failed reference compilation (6 bincount dynamic-output shapes, 6 max_pool1d cases whose reference is expected to raise, and 4 CausalBias snapshot reconstructions); 198 tests did not execute an interceptable reference assignment; and 16 setup-level skips did not reach the capture hook.

The self-check mismatches are retained rather than filtered: 8 dropout, 8 alpha_dropout, 6 argsort, 4 sort, and 1 rotary-position-embedding specialization. They reflect RNG stream differences, tie-index choices, or low-precision tolerance; they are not used by a generated PairSpec in this run.

## PairSpec generation

- Generated: 749
- Unavailable: 1353

| Why formal verification was not reached | Tests |
| --- | ---: |
| RHS lowered to multiple Triton kernels/launches | 442 |
| LHS is absent or has multiple launches | 325 |
| single-kernel IR has multiple stores | 309 |
| LHS/RHS pointer ABI counts differ | 128 |
| RHS TTIR is absent or multiple references were intercepted | 105 |
| reference output is tuple/empty/not a single tensor | 28 |
| test did not reach the capture hook | 16 |

## Formal verification

- ETV invoked: 749
- Structured result produced: 721
- Conclusive status: 6
- Status counts: `{"DISPROVED": 4, "NOT_RUN": 1353, "PROVED": 2, "RUNNER_ERROR": 28, "UNKNOWN": 715}`

| UNKNOWN reason | Tests |
| --- | ---: |
| TTIR_REGION_SEMANTICS_UNSUPPORTED | 367 |
| TTIR_OP_UNSUPPORTED | 236 |
| TTIR_UNDEFINED_LOAD_LANE | 56 |
| TTIR_MEMORY_TYPE_UNSUPPORTED | 36 |
| CAST_EQUIVALENCE_NOT_REWRITTEN | 12 |
| TTIR_CONSTANT_UNSUPPORTED | 8 |

The 28 RUNNER_ERROR cases are diag PairSpecs with an unbound compacted TTIR scalar (`arg3`). ETV was invoked but did not write report.json.

### Conclusive cases

- PROVED: diag 1D, n=1, diagonal=0, for float32 and float16.
- DISPROVED: four diag 2D cases with diagonal +4/-4, both dtypes.

The four DISPROVED results are not evidence of a numeric implementation bug. Every corresponding original test passed. Their LHS launch receives a tensor view whose pointer already includes storage_offset=4, while RHS receives the base pointer and adds 4 in TTIR. The current handwritten PairSpec maps both physical pointers to the same logical base and therefore produces a COMPUTE_MISMATCH (read offset 0 versus 4). The raw ETV result is preserved, but the pair predicate is incomplete.

## Per-operator results

`Exact` means both TTIR sides were captured. `Checked` means the RHS compiled-vs-FX self-check passed. Formal columns are P/D/U/E/N for PROVED/DISPROVED/UNKNOWN/RUNNER_ERROR/NOT_RUN.

| Operator | Tests | Pass | Fail | Skip | Exact | Checked | PairSpec | P | D | U | E | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abs | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| acosh | 38 | 38 | 0 | 0 | 31 | 31 | 29 | 0 | 0 | 29 | 0 | 9 |
| adaptive_avg_pool2d | 6 | 6 | 0 | 0 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 |
| adaptive_max_pool2d | 4 | 4 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 |
| add | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| addmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| addmv | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| alpha_dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| argsort | 60 | 60 | 0 | 0 | 60 | 54 | 0 | 0 | 0 | 0 | 0 | 60 |
| atan | 17 | 17 | 0 | 0 | 15 | 15 | 15 | 0 | 0 | 15 | 0 | 2 |
| avg_pool2d | 36 | 18 | 0 | 18 | 18 | 18 | 18 | 0 | 0 | 18 | 0 | 18 |
| batch_norm | 32 | 32 | 0 | 0 | 24 | 24 | 2 | 0 | 0 | 2 | 0 | 30 |
| bincount | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| bitwise_and | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| bitwise_not | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| bitwise_or | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| bmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| celu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| clamp | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| conv2d | 108 | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 108 |
| cos | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| cosh | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| diag | 60 | 60 | 0 | 0 | 46 | 46 | 46 | 2 | 4 | 12 | 28 | 14 |
| div | 24 | 16 | 0 | 8 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 8 |
| dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| eq | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| exp | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| fmax | 10 | 10 | 0 | 0 | 10 | 10 | 10 | 0 | 0 | 10 | 0 | 0 |
| ge | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| gelu | 16 | 8 | 0 | 8 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 8 |
| gt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| instance_norm | 384 | 384 | 0 | 0 | 384 | 384 | 0 | 0 | 0 | 0 | 0 | 384 |
| isinf | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| isnan | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| layer_norm | 96 | 96 | 0 | 0 | 96 | 96 | 21 | 0 | 0 | 21 | 0 | 75 |
| le | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| logsumexp | 52 | 52 | 0 | 0 | 46 | 46 | 44 | 0 | 0 | 44 | 0 | 8 |
| lp_pool1d | 12 | 12 | 0 | 0 | 11 | 11 | 11 | 0 | 0 | 11 | 0 | 1 |
| lp_pool2d | 24 | 24 | 0 | 0 | 24 | 24 | 24 | 0 | 0 | 24 | 0 | 0 |
| lp_pool3d | 24 | 24 | 0 | 0 | 24 | 24 | 16 | 0 | 0 | 16 | 0 | 8 |
| lt | 8 | 8 | 0 | 0 | 7 | 7 | 7 | 0 | 0 | 7 | 0 | 1 |
| matmul | 8 | 8 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8 |
| max | 24 | 24 | 0 | 0 | 24 | 24 | 8 | 0 | 0 | 8 | 0 | 16 |
| max_pool1d | 18 | 18 | 0 | 0 | 12 | 12 | 12 | 0 | 0 | 12 | 0 | 6 |
| max_pool2d | 108 | 54 | 0 | 54 | 54 | 54 | 54 | 0 | 0 | 54 | 0 | 54 |
| max_pool3d | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| maximum | 11 | 11 | 0 | 0 | 11 | 11 | 11 | 0 | 0 | 11 | 0 | 0 |
| mean | 25 | 25 | 0 | 0 | 25 | 25 | 24 | 0 | 0 | 24 | 0 | 1 |
| median | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| mm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| msort | 8 | 6 | 2 | 0 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 2 |
| mul | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| ne | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| neg | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| pow | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| quantile | 160 | 80 | 0 | 80 | 80 | 80 | 4 | 0 | 0 | 4 | 0 | 156 |
| relu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| rms_norm | 64 | 64 | 0 | 0 | 64 | 64 | 28 | 0 | 0 | 28 | 0 | 36 |
| rot90 | 32 | 32 | 0 | 0 | 32 | 32 | 32 | 0 | 0 | 32 | 0 | 0 |
| rotary_position_embedding | 128 | 128 | 0 | 0 | 128 | 127 | 0 | 0 | 0 | 0 | 0 | 128 |
| round | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| rsqrt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| scaled_dot_product_attention | 78 | 78 | 0 | 0 | 54 | 54 | 0 | 0 | 0 | 0 | 0 | 78 |
| select_copy | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| sgn | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 |
| sigmoid | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| sign | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| signbit | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| silu | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| sin | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| softmax | 8 | 8 | 0 | 0 | 8 | 8 | 5 | 0 | 0 | 5 | 0 | 3 |
| sort | 36 | 36 | 0 | 0 | 28 | 24 | 0 | 0 | 0 | 0 | 0 | 36 |
| stack | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| sub | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |
| tanh | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 4 |
| threshold | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 |

All individual node IDs, paths, raw reasons, PairSpecs, and ETV reports remain in the case directories and in collection-summary.json, pairspec-summary.json, and formal-summary.json.
