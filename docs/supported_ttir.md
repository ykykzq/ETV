# Supported TTIR

Official Triton 3.7.1 parsing and verification precede lifting. This is the modeled
acyclic subset, not everything Triton can parse.

| Operations | Interpretation |
| --- | --- |
| `tt.get_program_id`, `tt.make_range` | Axis 0 and static one-dimensional lane ranges. |
| `tt.splat/broadcast/expand_dims/reshape` | Static index maps preserving element order. |
| `tt.addptr` | Element offsets retaining storage identity. |
| `tt.load/store` | Float memory, masks, defaults/undefined. |
| `tt.clampf` | Real clamp with `propagateNan = none`. |
| `arith.constant` | Bool, arbitrary-width integer, finite rational float, static dense constants. |
| `arith.addi/subi/muli/divsi/remsi` | Finite signed arithmetic and definedness. |
| `arith.andi/ori/xori` | `i1` only. |
| `arith.cmpi` | `eq/ne/slt/sle/sgt/sge`. |
| `arith.addf/subf/mulf/divf/negf` | Real arithmetic. |
| `arith.cmpf` | Real interpretation, no NaN domain. |
| `arith.select/maxnumf/minnumf` | Typed selection and real min/max. |
| `arith.extsi/extui/trunci/index_cast/index_castui` | Explicit finite-width conversion. |
| `arith.sitofp/uitofp`, `arith.extf/truncf` | Typed integer-to-real and same-value abstract float conversions. |
| `math.absf/ceil/cos/erf/exp/exp2/floor/fma/log/rsqrt/sin/sqrt` | Pure real functions and required domains. |

`tt.extern_elementwise` requires `pure=true` and one of these symbols, optionally
prefixed with `__nv_`: `acosh/acoshf`, `atan/atanf`, `ceilf`, `coshf`, `erff`,
`expm1f`, `floorf`, `nearbyintf`, `powf`, `rsqrtf`, `sqrtf`, `tanhf`. These denote
abstract mathematical functions, not bit-level CUDA libdevice equivalence.

Exact replay supports rational arithmetic, integer operations/casts, booleans,
comparisons/select, abs, min/max, floor/ceil, ties-to-even nearbyint and fma. Other
pure functions can participate in structural/congruence proofs; no approximate
floating-point counterexample path exists.

Unsupported: nested computation regions, loops, reductions, atomics, barriers,
shared memory, bitcasts, non-boolean bitwise ops, integer/bool memory, annotated
signed/unsigned MLIR integer types, non-default pointer address spaces, dynamic
shapes, TTGIR encodings, multidimensional grids and cross-stream/event semantics.
Unsupported integer overflow flags are rejected. See [verification](verification.md)
for fixed and parameterized writer restrictions.
