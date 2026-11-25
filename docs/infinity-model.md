# Floating-point special values

The TTIR-only verifier uses the partial value domain `R ∪ {−∞, +∞, NaN}` for floating intermediates. Z3 uses a tagged datatype with `Finite(Real)`, `PositiveInfinity`, `NegativeInfinity` and `NaN`; the concrete evaluator uses exact rational finite values and explicit special-value tags. No largest finite number or epsilon is substituted for infinity.

External floating buffer values and scalar variables remain **finite, unbounded mathematical reals**. Thus `isinf(input)` is false under this declared input-domain premise. This does not prove correctness for arbitrary IEEE infinity/NaN input buffers. TTIR constants and modeled intermediate values can carry any of the tags.

| Operation | Supported behavior |
| --- | --- |
| Constants | Official MLIR/APFloat classification; scalar and dense f16/bf16/f32/f64 infinities preserve sign. NaN constants preserve the NaN tag. |
| Comparisons, min/max, select | Extended-real ordering; all finite values lie strictly between −Inf and +Inf. Ordered comparisons with NaN are false and unordered comparisons are true. Min/max retain infinite reduction identities. Observation equality compares NaN by class. |
| Negation, absolute value, float casts | Preserve or transform infinity tags. Float casts retain the existing no-rounding abstraction. |
| Addition, subtraction, multiplication, division, fma | Exact finite-real arithmetic and defined signed-infinity combinations; opposite-infinity addition, same-infinity subtraction, zero times infinity and infinity divided by infinity fail definedness. Basic arithmetic propagates an explicit NaN. Division by zero still requires definedness. |
| exp / exp2 / expm1 | Exact infinite limits, including exp(−Inf)=0. Finite transcendental calls retain the existing uninterpreted overapproximation. |
| sqrt / rsqrt | Negative inputs and NaN return NaN. At an external input zero, rsqrt returns infinity with that input's sign. A computed zero in rsqrt retains a definedness obligation. |
| log | Supported +Inf limit and positive finite domain. |
| pow | Exact for constant integer exponents 1 through 4 in the modeled domain. Zero raised to a positive finite exponent is zero. Positive finite bases with other exponents retain an uninterpreted positive result. |
| acosh | Uses the real formula on x > 1, zero at x = 1 and NaN below one. |
| cosh / tanh / erf | Supported infinite limits. |
| floor / ceil / nearbyint | Infinite values remain infinite. |
| libdevice isinff / isinfd | Accepted only when every SSA use checks equality/inequality against literal zero. Internally normalized as a predicate; raw integer observations remain unsupported because the contract guarantees nonzero, not integer 1. |
| Other transcendental cases | No guessed limits: unsupported infinite cases remain UNKNOWN. |

Mandatory definedness checks precede output equality. The selected branch controls select definedness; comparison/isnan operands retain dependencies. Operations with an unmodeled exceptional result retain their definedness obligations even when both outputs have identical syntax. Generic rewrite candidates are checked over all tagged extended-real values with equal definedness, so finite-only algebra is not silently generalized to infinity.

The verifier excludes IEEE rounding, finite-width floating overflow and NaN payloads. The bitcast sign predicate and rsqrt can observe the sign of external zero values in SMT. The model shares that sign per logical buffer element and treats exact float widening as preserving it. Ordinary real-valued observations still identify the two zeros. Integer semantics retain their existing finite-width model. No host initialization, padding, sorting or other Python/Torch operation is introduced; missing TTIR-only producers remain a separate pairing limitation. The same model applies within supported SCF regions.

Reference: [MLIR arithmetic operations](https://mlir.llvm.org/docs/Dialects/ArithOps/) specify floating comparisons and min/max NaN behavior.

Nonzero MLIR fastmath flags are rejected until their exceptional-value premises are modeled. Scalar reduction results bind their sole lane to zero before broadcasting; program IDs remain unchanged. DAG caches are implementation optimizations: private hash/dependency memos are omitted from serialization, and memory-resolution caches never cross state/binding snapshots.

Reference: [NVIDIA libdevice isinff](https://docs.nvidia.com/cuda/archive/12.1.0/libdevice-users-guide/__nv_isinff.html) specifies the zero/nonzero contract.

For fixed large outputs that use a symbolic all-elements proof, an SMT unknown/timeout now retains the actual unresolved obligation and returns formal UNKNOWN. It does not restart full enumeration after that solver budget is exhausted. Unsupported symbolic layouts still permit exact enumeration fallback. This changes resource strategy only; no incomplete obligation is counted as proved.
