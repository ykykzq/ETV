# Supported TTIR

Official Triton 3.7.1 parsing and verification precede lifting. This is the modeled
acyclic subset, not everything Triton can parse.

| Operations | Interpretation |
| --- | --- |
| `tt.get_program_id`, `tt.make_range` | Axis 0 and static one-dimensional lane ranges. |
| `tt.splat/broadcast/expand_dims/reshape` | Static index maps preserving element order. |
| `tt.gather` | Tensor coordinate selection with static source shapes; out-of-bounds indices are undefined. |
| `tt.addptr` | Element offsets retaining storage identity. |
| `tt.load/store` | Float, bool and finite integer memory, masks, defaults/undefined. |
| `tt.reduce` | Static single-input associative reduction, extent up to 4096; add/mul/min/max and integer add/mul/and/or/xor. |
| `tt.bitcast` | Bool-pointer to byte-pointer canonical boolean stores; loads additionally require `target.boolean_storage: canonical_i8`. Other reinterpretations rejected. |
| `tt.clampf` | Real clamp with `propagateNan = none`. |
| `arith.constant` | Bool, arbitrary-width integer, finite rational float, static dense constants. |
| `arith.addi/subi/muli/divsi/remsi` | Finite signed arithmetic and definedness. |
| `arith.minsi/maxsi` | Signed finite-width min/max, including signed i1 values. |
| `arith.shrui`, `tt.mulhiui` | Logical right shift and high half of an unsigned double-width product. Shift counts must be below the bit width. |
| `arith.andi/ori/xori` | Booleans and finite-width integers. |
| `arith.cmpi` | `eq/ne/slt/sle/sgt/sge/ult/ule/ugt/uge`. |
| `arith.addf/subf/mulf/divf/negf` | Real arithmetic. |
| `arith.cmpf` | Real interpretation, no NaN domain. |
| `arith.select/maxnumf/minnumf` | Typed selection and real min/max. |
| `arith.extsi/extui/trunci/index_cast/index_castui` | Explicit finite-width conversion. |
| `arith.sitofp/uitofp`, `arith.extf/truncf` | Typed integer-to-real and same-value abstract float conversions. |
| `arith.fptosi/fptoui` | Real-to-integer conversion truncating toward zero; the result must fit the signed/unsigned target range. |
| `math.absf/ceil/cos/erf/exp/exp2/floor/fma/log/rsqrt/sin/sqrt` | Pure real functions and required domains. |

`tt.extern_elementwise` requires `pure=true` and one of these symbols, optionally
prefixed with `__nv_`: `acosh/acoshf`, `atan/atanf`, `ceilf`, `coshf`, `erff`,
`expm1f`, `floorf`, `nearbyintf`, `powf`, `rsqrtf`, `sqrtf`, `tanhf`, `isnan/isnanf`. These denote
abstract mathematical functions, not bit-level CUDA libdevice equivalence.
`isnan` is false in the finite real domain. All 16 MLIR `cmpf` predicate encodings
are tested against that domain, including ordered/unordered and constant predicates.

Exact replay supports rational arithmetic, integer operations/casts, booleans,
comparisons/select, abs, min/max, floor/ceil, ties-to-even nearbyint and fma. Other
pure functions can participate in structural/congruence proofs; no approximate
floating-point counterexample path exists.

Unsupported: other computation regions, loops, multi-input/non-associative reductions,
non-finite constants, atomics, barriers, shared memory, general bitcasts, annotated
signed/unsigned MLIR integer types, non-default pointer address spaces, dynamic
shapes, TTGIR encodings, multidimensional grids and cross-stream/event semantics.
Unsupported integer overflow and exact-shift flags are rejected. Data-dependent
memory addresses remain outside the modeled gather subset. See [verification](verification.md)
for fixed and parameterized writer restrictions.
