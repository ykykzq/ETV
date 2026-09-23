# Semantics and Trust

`PROVED` states equality of observed logical memory under the reported PairSpec
premises and `abstract_float` semantics. It does not establish arbitrary host or
bit-level GPU behavior.

Floats denote mathematical reals. Constants are exact rationals from official IR
attributes. Real comparisons/min/max exclude NaN, infinity and signed-zero
distinctions. Float conversions keep source/target metadata and are eliminated
only by admitted rules for this abstraction. There is no rounding/tolerance model.

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
outside this subset.

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
