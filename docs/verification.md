# Verification and Reports

The orchestrator freezes the strict PairSpec and input manifest, parses/verifies
distinct modules, selects exact functions, lifts physical launches and validates
ABI types. It composes both step sequences and closes mandatory memory obligations
before rule admission and equality saturation.

## Memory and Computation

Fixed verification enumerates all declared program/lane/store instances. Same-step
launches read the prior state and commit together. Duplicate active writes produce
`WRITE_RACE`. Intermediate loads require prior writes at the same storage/offset.
Missing masked-load defaults remain explicit undefined branches. Final visible
output addresses determine observation roots.

Parameterized verification constructs `program = k / block_size` and
`lane = k % block_size`. It proves domains, ranges, coverage, active addresses,
masks, canonical linear uniqueness and load alignment. Its mathematical-integer
optimization requires proved ranges for every intermediate finite integer before
being used for equivalence. One launch/store per side is currently supported.

A symbolic memory model is specialized into the original typed program and
exhaustively replayed. Only reproduced mismatches become `DISPROVED`; inconclusive
or too-large replay remains `UNKNOWN`.

Rule admission is local to each run. Optional partitions prove boundary equalities
in dependency order. Saturation runs in an isolated worker with wall-clock,
iteration and e-node limits. Final root equality must hold in egglog. If unresolved,
an exact Z3 model can be replayed using the independent rational/integer evaluator.
Uninterpreted transcendental encodings cannot support `DISPROVED`.

## Output Contract

`etv verify PAIR --out DIRECTORY [--json]` writes `report.json` and `report.md`.
JSON schema version 1 contains `tool`, `run_id`, `pair_id`, `status`, `reason`,
`inputs`, `semantics`, `predicates`, `obligations`, `rewrites`, `proof`,
`counterexample`, `unsupported`, `soundness`, launch/store counts and diagnostics.
Markdown renders the same facts without additional conclusions.

Inputs include hashes and IDs. Obligations reference premises/frontend facts;
root equalities reference obligations, used rules and proved partitions. Rules
retain expressions, source hash, validation/admission, gates and match counts.
Matches conservatively track use, not a minimal extracted derivation. Counterexamples
include index, offsets/masks, values when applicable, assignments and replay guidance.

Logs go to stderr. `--log-format jsonl` emits UTC time, level, event, run ID, pair
ID, phase and fields. `--log-file PATH` also persists them. JSON stdout remains a
single object.

## Reasons

`OBSERVABLE_MEMORY_EQUIVALENT` means `PROVED`. Replayed `DISPROVED` reasons are
`COVERAGE_MISMATCH`, `WRITE_RACE`, `ADDRESS_MISMATCH`, `MASK_MISMATCH`,
`COMPUTE_MISMATCH`.

Common `UNKNOWN` reasons: `INVALID_PAIRSPEC`, `INPUT_NOT_FOUND`, `FILE_HASH_MISMATCH`,
`TTIR_VERSION_MISMATCH`, `TTIR_PARSE_FAILED`, `TTIR_VERIFY_FAILED`,
`TTIR_OP_UNSUPPORTED`, `TTIR_REGION_UNSUPPORTED`, `ABI_ROLE_MISSING`,
`NO_ALIAS_NOT_DECLARED`, `UNDEFINED_LOAD_REACHABLE`, `FLOAT_DEFINEDNESS_NOT_PROVED`,
`INTEGER_DEFINEDNESS_NOT_PROVED`, `CAST_EQUIVALENCE_NOT_REWRITTEN`, `SMT_UNKNOWN`,
`RESOURCE_LIMIT`, `NO_EQUIVALENCE_PROOF`, `INTERNAL_ERROR`. More specific unsupported
codes preserve this interpretation. Unexpected exceptions log an error and produce
an `UNKNOWN(INTERNAL_ERROR)` report.
