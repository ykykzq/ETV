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

- LHS statuses: `{"captured": 1776, "compile_failed": 110, "not_captured": 200, "not_reached": 16}`
- RHS statuses: `{"captured": 1783, "no_triton_kernel": 90, "not_reached": 16, "reference_compile_failed": 16, "reference_not_intercepted": 197}`
- Both sides captured: 1667 tests across 67 operators
- RHS compiled-vs-FX self-check on two-sided captures: `{"mismatch_or_unavailable": 27, "pass": 1640}`

## PairSpec generation

- Cases generated: 1068
- Cases unavailable: 1034
- Generated output observations: 1099
- Recovered-feature cases: `{"incomplete_pointer_mapping": 320, "inferred_single_store_output": 0, "multiple_references": 4, "multiple_references_fully_generated": 4, "multiple_stores": 199, "nonzero_endpoint_offset": 12, "tuple_output": 27, "unequal_pointer_abi": 284}`

| Why formal verification was not reached | Cases |
| --- | ---: |
| LHS TTIR is absent or has multiple launches | 322 |
| runtime launches cannot be associated with selected TTIR | 192 |
| stores cannot be associated with output leaves | 175 |
| TTIR/runtime pointer ABI cannot be aligned | 102 |
| RHS TTIR is absent | 101 |
| multi-launch final output store is ambiguous | 81 |
| captured RHS failed compiled-vs-FX self-check | 27 |
| multiple runtime launches were captured, but only one launch contributes to this observed output leaf | 18 |
| test did not reach the capture hook | 16 |

## Formal verification

- ETV invocations: 1099 output observations across 1068 tests
- Structured results: 1099
- Conclusive results: 115 across 115 tests
- Observation statuses: `{"DISPROVED": 113, "NOT_RUN": 1034, "PROVED": 2, "UNKNOWN": 984}`
- DISPROVED observations: 113

| UNKNOWN/runner reason | Observations |
| --- | ---: |
| TTIR_REGION_SEMANTICS_UNSUPPORTED | 706 |
| MISSING_ROLE | 72 |
| TTIR_OP_UNSUPPORTED | 68 |
| TTIR_MEMORY_TYPE_UNSUPPORTED | 46 |
| TTIR_UNDEFINED_LOAD_LANE | 44 |
| CAST_EQUIVALENCE_NOT_REWRITTEN | 32 |
| TTIR_CONSTANT_UNSUPPORTED | 8 |
| TTIR_EXTERN_SYMBOL_UNSUPPORTED | 8 |

### DISPROVED observations

- `build/upstream/ntops/tests/test_abs.py::test_abs[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_abs.py::test_abs[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_clamp.py::test_clamp[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_clamp.py::test_clamp[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cos.py::test_cos[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cos.py::test_cos[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cos.py::test_cos[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_cosh.py::test_cosh[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[10-0-dtype0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[10-0-dtype1]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[5-0-dtype0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_diag.py::test_diag_1d[5-0-dtype1]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_div.py::test_div[shape2-dtype2-cuda-0.001-0.001-None]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_div.py::test_div[shape2-dtype2-cuda-0.001-0.001-floor]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_div.py::test_div[shape4-dtype4-cuda-0.001-0.001-None]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_div.py::test_div[shape4-dtype4-cuda-0.001-0.001-floor]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_exp.py::test_exp[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_exp.py::test_exp[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_exp.py::test_exp[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_exp.py::test_exp[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_fmax.py::test_fmax_elementwise[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_fmax.py::test_fmax_elementwise[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape0-dtype0-cuda-0.001-0.001-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape1-dtype1-cuda-0.01-0.01-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape2-dtype2-cuda-0.001-0.001-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape3-dtype3-cuda-0.01-0.01-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape4-dtype4-cuda-0.001-0.001-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape5-dtype5-cuda-0.01-0.01-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape6-dtype6-cuda-0.001-0.001-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_gelu.py::test_gelu[shape7-dtype7-cuda-0.01-0.01-none]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_maximum.py::test_maximum_elementwise[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_maximum.py::test_maximum_elementwise[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_maximum.py::test_maximum_elementwise[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_mul.py::test_mul[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_mul.py::test_mul[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_mul.py::test_mul[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_mul.py::test_mul[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_neg.py::test_neg[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_pow.py::test_pow[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape0-dtype0-cuda-0.001-0.001-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape0-dtype0-cuda-0.001-0.001-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape1-dtype1-cuda-0.01-0.01-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape1-dtype1-cuda-0.01-0.01-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape2-dtype2-cuda-0.001-0.001-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape2-dtype2-cuda-0.001-0.001-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape3-dtype3-cuda-0.01-0.01-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape3-dtype3-cuda-0.01-0.01-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape4-dtype4-cuda-0.001-0.001-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape4-dtype4-cuda-0.001-0.001-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape5-dtype5-cuda-0.01-0.01-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape5-dtype5-cuda-0.01-0.01-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape6-dtype6-cuda-0.001-0.001-False]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_relu.py::test_relu[shape6-dtype6-cuda-0.001-0.001-True]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape0-dtype0-cuda-0.001-0.001-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape1-dtype1-cuda-0.01-0.01-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape2-dtype2-cuda-0.001-0.001-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape3-dtype3-cuda-0.01-0.01-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape4-dtype4-cuda-0.001-0.001-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape5-dtype5-cuda-0.01-0.01-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rot90.py::test_rot90[shape6-dtype6-cuda-0.001-0.001-0]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_round.py::test_round[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_rsqrt.py::test_rsqrt[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sigmoid.py::test_sigmoid[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sigmoid.py::test_sigmoid[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sigmoid.py::test_sigmoid[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sigmoid.py::test_sigmoid[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape1-dtype1-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape3-dtype3-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape5-dtype5-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_silu.py::test_silu[shape7-dtype7-cuda-0.01-0.01]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sin.py::test_sin[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sin.py::test_sin[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sin.py::test_sin[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_sin.py::test_sin[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_tanh.py::test_tanh[shape0-dtype0-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_tanh.py::test_tanh[shape2-dtype2-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_tanh.py::test_tanh[shape4-dtype4-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH
- `build/upstream/ntops/tests/test_tanh.py::test_tanh[shape6-dtype6-cuda-0.001-0.001]::r000-e000-o000`: MASK_MISMATCH

## Per-operator results

Formal columns count output observations, so tuple outputs and multiple reference expressions can contribute more than one observation per test.

| Operator | Tests | Pass | Fail | Skip | Exact | Pair cases | Specs | P | D | U | E | T | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abs | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |
| acosh | 38 | 38 | 0 | 0 | 31 | 31 | 33 | 0 | 0 | 33 | 0 | 0 | 7 |
| adaptive_avg_pool2d | 6 | 6 | 0 | 0 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 0 |
| adaptive_max_pool2d | 4 | 4 | 0 | 0 | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| add | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| addmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| addmv | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| alpha_dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| argsort | 60 | 60 | 0 | 0 | 60 | 54 | 54 | 0 | 0 | 54 | 0 | 0 | 6 |
| atan | 17 | 17 | 0 | 0 | 15 | 15 | 15 | 0 | 0 | 15 | 0 | 0 | 2 |
| avg_pool2d | 36 | 18 | 0 | 18 | 18 | 18 | 18 | 0 | 0 | 18 | 0 | 0 | 18 |
| batch_norm | 32 | 32 | 0 | 0 | 24 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 20 |
| bincount | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| bitwise_and | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bitwise_not | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bitwise_or | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| bmm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| celu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| clamp | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 2 | 6 | 0 | 0 | 0 |
| conv2d | 108 | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 108 |
| cos | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 3 | 1 | 0 | 0 | 4 |
| cosh | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 0 | 0 |
| diag | 60 | 60 | 0 | 0 | 46 | 46 | 46 | 2 | 4 | 40 | 0 | 0 | 14 |
| div | 24 | 16 | 0 | 8 | 16 | 16 | 16 | 0 | 4 | 12 | 0 | 0 | 8 |
| dropout | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| eq | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| exp | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 4 |
| fmax | 10 | 10 | 0 | 0 | 10 | 10 | 10 | 0 | 2 | 8 | 0 | 0 | 0 |
| ge | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| gelu | 16 | 8 | 0 | 8 | 8 | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 8 |
| gt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| instance_norm | 384 | 384 | 0 | 0 | 384 | 84 | 84 | 0 | 0 | 84 | 0 | 0 | 300 |
| isinf | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| isnan | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| layer_norm | 96 | 96 | 0 | 0 | 96 | 87 | 87 | 0 | 0 | 87 | 0 | 0 | 9 |
| le | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| logsumexp | 52 | 52 | 0 | 0 | 48 | 48 | 50 | 0 | 0 | 50 | 0 | 0 | 4 |
| lp_pool1d | 12 | 12 | 0 | 0 | 12 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 0 |
| lp_pool2d | 24 | 24 | 0 | 0 | 24 | 24 | 24 | 0 | 0 | 24 | 0 | 0 | 0 |
| lp_pool3d | 24 | 24 | 0 | 0 | 24 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 8 |
| lt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| matmul | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| max | 24 | 24 | 0 | 0 | 24 | 23 | 38 | 0 | 0 | 38 | 0 | 0 | 1 |
| max_pool1d | 18 | 18 | 0 | 0 | 12 | 12 | 12 | 0 | 0 | 12 | 0 | 0 | 6 |
| max_pool2d | 108 | 54 | 0 | 54 | 54 | 54 | 54 | 0 | 0 | 54 | 0 | 0 | 54 |
| max_pool3d | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| maximum | 11 | 11 | 0 | 0 | 11 | 11 | 11 | 0 | 3 | 8 | 0 | 0 | 0 |
| mean | 25 | 25 | 0 | 0 | 25 | 24 | 24 | 0 | 0 | 24 | 0 | 0 | 1 |
| median | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| mm | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| msort | 8 | 6 | 2 | 0 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 2 |
| mul | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 4 | 4 | 0 | 0 | 0 |
| ne | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| neg | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 0 | 0 |
| pow | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 1 | 3 | 0 | 0 | 4 |
| quantile | 160 | 80 | 0 | 80 | 80 | 23 | 23 | 0 | 0 | 23 | 0 | 0 | 137 |
| relu | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 14 | 2 | 0 | 0 | 0 |
| rms_norm | 64 | 64 | 0 | 0 | 64 | 58 | 58 | 0 | 0 | 58 | 0 | 0 | 6 |
| rot90 | 32 | 32 | 0 | 0 | 32 | 32 | 32 | 0 | 7 | 25 | 0 | 0 | 0 |
| rotary_position_embedding | 128 | 128 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 128 |
| round | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |
| rsqrt | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 7 | 1 | 0 | 0 | 0 |
| scaled_dot_product_attention | 78 | 78 | 0 | 0 | 54 | 21 | 21 | 0 | 0 | 21 | 0 | 0 | 57 |
| select_copy | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| sgn | 16 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 0 | 16 | 0 | 0 | 0 |
| sigmoid | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 4 |
| sign | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| signbit | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| silu | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |
| sin | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 4 |
| softmax | 8 | 8 | 0 | 0 | 8 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 3 |
| sort | 36 | 36 | 0 | 0 | 28 | 12 | 24 | 0 | 0 | 24 | 0 | 0 | 24 |
| stack | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| sub | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |
| tanh | 8 | 8 | 0 | 0 | 4 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 4 |
| threshold | 8 | 8 | 0 | 0 | 8 | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 |

Individual node IDs, PairSpecs, pointer/alias diagnostics, raw reasons, and ETV reports remain in each case directory and the JSON summaries.
