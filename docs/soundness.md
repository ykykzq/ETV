# Semantics and Trust

`PROVED` states equality of observed logical memory under the reported PairSpec
premises and `abstract_float` semantics. It does not establish arbitrary host or
bit-level GPU behavior.

Floating intermediates denote extended mathematical reals: finite reals, positive
infinity and negative infinity are distinct tagged values. Finite constants remain
exact rationals from official IR attributes; infinity is never replaced by a large
finite sentinel. External floating inputs remain finite mathematical reals, which
is explicitly reported as a model premise. This extension does not establish
correctness for arbitrary NaN/infinity input buffers.

Comparisons, min/max, selection and supported arithmetic preserve infinity tags.
NaN is not a value in this model: reachable operations such as infinity minus the
same infinity or zero times infinity must fail the definedness obligation. Rules
that can match floating intermediates are checked over the extended domain; a rule
proved only for finite real variables cannot silently be applied to infinity.
Float conversions keep source/target metadata. Rounding, signed-zero distinctions,
overflow from finite IEEE inputs, NaN payloads and numerical tolerance are outside
the abstraction. See `infinity-model.md` for the implemented operation boundaries.

Signless `iN` uses a signed finite-width interpretation; add/sub/mul wrap modulo
width. Signed division/remainder truncate toward zero and exclude zero divisors
and minimum-integer divided by minus one. Unsigned conversions reinterpret source
bits explicitly. Index casts require an explicit index width; ordinary address
indices are mathematical integers.

`fdiv`, `sqrt`, `rsqrt`, `log`, `acosh` and general real `pow` have explicit domains.
Undefined loads must be unreachable from active outputs. Select definedness follows
the selected branch. Boolean and/or simplification does not erase undefined
dependencies. Transcendental UF encodings overapproximate values for obligations
and cannot establish an exact numerical counterexample.

External buffers are total maps over nonnegative offsets. Allocation sizes/device
pointer validity are host premises, not inferred bounds. Internal/output storage
starts uninitialized and becomes defined through prior steps. Explicit separation
is required between distinct roles interacting with writers. In-place aliasing is
outside this subset. A role both loaded and stored in one execution step returns
`UNKNOWN(INPLACE_MEMORY_DEPENDENCY_UNPROVED)`, including previously initialized
internal storage and dependencies between same-step launches. Snapshot reads are
not evidence that overlapping accesses are safe across programs or lanes. This
conservative check may reject independent address regions until their access
independence or ordering is proved; read-only aliases remain supported.

## Evidence Levels

`formal_under_declared_predicates` means mandatory obligations closed and egglog
roots merged using formally admitted rules, conditional on listed input predicates.
ABI, host schedule, bindings, domains and disjointness remain supplied premises.
Custom trusted predicates are listed as trusted axioms.

`conditional_on_unverified_rewrites` means an admitted unverified rule matched,
including in successful partition checks. Its ID/source stays visible. Tracking
all matches is conservative: some may not be needed for the final equality.

There is no independent proof certificate. The trusted implementation includes
the parser/verifier, native adapter, lifting/composition, rule validator, Z3, egglog
and pinned bindings. Tests exercise that implementation; they do not replace the
stated theorem.

Unsupported regions/operations remain `UNKNOWN` even for identical files. Missing
premises, partial domains, uncertain accesses and solver/resource failures cannot
produce `PROVED`. LLM output and partitions are proposals; invalid/unavailable
proposals fall back to whole-goal verification. Generated rules remain explicitly
trusted. Numerical benchmark checks cannot change a verifier verdict.
