# Supported TTIR

Official Triton 3.7.1 parsing and verification precede lifting. The modeled subset includes straight-line code and the structured regions below.

| Operations | Interpretation |
| --- | --- |
| `tt.get_program_id`, `tt.make_range` | Static three-axis launch grids and static one-dimensional lane ranges. A grid with only a program count has unit extents on Y and Z. |
| `tt.splat/broadcast/expand_dims/reshape` | Static index maps preserving element order. |
| `tt.gather` | Tensor coordinate selection with static source shapes; out-of-bounds indices are undefined. |
| `tt.addptr` | Element offsets retaining storage identity. |
| `tt.load/store` | Float, bool and finite integer memory, masks, defaults/undefined. |
| `scf.if/yield` | Scalar predicates and typed value merges. Branch stores with statically resolvable masks are checked for full output coverage and disjoint writes. |
| `scf.for/yield` | Signed loops with concrete bounds and positive step, simultaneous carried-value updates, and fixed-width integer recurrence summaries. Other supported bodies use exact expansion within a 64-trip per-loop limit and the instance budget. |
| `tt.reduce` | Static associative reductions up to 4096 elements. Supports single-input add/mul/min/max and integer add/mul/and/or/xor. Also recognizes compare/select min/max and lexicographic value/index pairs that select an extremum with an integer tie-break. |
| `tt.bitcast` | Bool-pointer to byte-pointer canonical boolean storage. Loads require `target.boolean_storage: canonical_i8`. Same-format float-to-integer-to-float round trips support static XOR lane routing when each output reduces to an original bit pattern. |
| `tt.clampf` | Real clamp with `propagateNan = none`. |
| `arith.constant` | Bool, arbitrary-width integer, finite rational float, signed infinity, NaN and static dense constants. |
| `arith.addi/subi/muli/divsi/remsi` | Finite signed arithmetic and definedness. |
| `arith.minsi/maxsi` | Signed finite-width min/max, including signed i1 values. |
| `arith.shrui`, `tt.mulhiui` | Logical right shift and high half of an unsigned double-width product. Shift counts must be below the bit width. |
| `arith.andi/ori/xori` | Booleans and finite-width integers. |
| `arith.cmpi` | `eq/ne/slt/sle/sgt/sge/ult/ule/ugt/uge`. |
| `arith.addf/subf/mulf/divf/negf` | Real arithmetic. |
| `arith.cmpf` | Extended-real comparisons with ordered and unordered predicates for NaN. |
| `arith.select/maxnumf/minnumf` | Typed selection and real min/max. |
| `arith.extsi/extui/trunci/index_cast/index_castui` | Explicit finite-width conversion. |
| `arith.sitofp/uitofp`, `arith.extf/truncf` | Typed integer-to-real and same-value abstract float conversions. |
| `arith.fptosi/fptoui` | Real-to-integer conversion truncating toward zero; the result must fit the signed/unsigned target range. |
| `math.absf/ceil/cos/erf/exp/exp2/floor/fma/log/rsqrt/sin/sqrt` | Pure real functions and required domains. |

`tt.extern_elementwise` requires `pure=true` and one of these symbols, optionally
prefixed with `__nv_`: `acosh/acoshf`, `atan/atanf`, `ceilf`, `coshf`, `erff`,
`exp`, `expm1f`, `floorf`, `nearbyintf`, `powf`, `rsqrtf`, `sqrtf`, `tanhf`, `isnan/isnanf`. These denote
abstract mathematical functions, not bit-level CUDA libdevice equivalence.
`isnan` is false for finite values and signed infinities. In the abstract model,
`acosh` uses the real identity `log(x + sqrt((x - 1)(x + 1)))` for `x > 1`,
returns zero at one and returns NaN below one.

Exact replay supports rational arithmetic, integer operations/casts, booleans,
comparisons/select, abs, min/max, floor/ceil, ties-to-even nearbyint and fma. Other
pure functions can participate in structural/congruence proofs; no approximate
floating-point counterexample path exists.

Unsupported: `scf.while`, dynamic or unsigned loop bounds, induction overflow,
control-dependent writer sets that cannot be resolved, and repeated writes that
require reasoning about execution order. Other limits include general tuple or
non-associative reductions, NaN payloads, atomics, synchronization inside
structured regions, shared memory, general bitcasts, and annotated
signed/unsigned MLIR integer types, non-default pointer address spaces, dynamic
shapes, TTGIR encodings, dynamic multidimensional grids and cross-stream/event semantics.
Unsupported integer overflow and exact-shift flags are rejected. Data-dependent
memory addresses remain outside the modeled gather subset. See [verification](verification.md)
for fixed and parameterized writer restrictions.
