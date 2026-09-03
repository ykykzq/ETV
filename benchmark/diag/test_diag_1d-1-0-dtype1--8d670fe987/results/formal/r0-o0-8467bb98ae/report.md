# ETV verification report: build/upstream/ntops/tests/test_diag.py::test_diag_1d[1-0-dtype1]::r000-e000-o000

- Status: **PROVED**
- Reason: `OBSERVABLE_MEMORY_EQUIVALENT`
- Semantic mode: `abstract_float`
- Scope: fixed-specialization single-store bounded translation validation
- Soundness: `formal_under_declared_predicates`

## Assumptions

- `TRUSTED_AXIOM`: X = 1 (PairSpec.predicates.bindings)
- `TRUSTED_AXIOM`: disjoint(Input0, Output) (PairSpec.predicates.disjoint)

## LLM-only context

- Both kernels were captured from the same specialized pytest node.
- Input roles are aligned by captured tensor/storage provenance when available, otherwise by exact runtime tensor signatures; endpoint offsets normalize tensor views to their logical storage bases.
- Selected-store dependencies without a reliable counterpart are left unmapped and must prevent a proof if they affect the observation.
- This PairSpec observes output leaf 'compiled_output' only.
- Proof relevance: `informational_only`.

## Declared formal predicates

- `abi.Input0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Output`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.X`: builtin / assumed (`TRUSTED_AXIOM`)
- `disjoint.0`: builtin / assumed (`TRUSTED_AXIOM`)

## Rewrite registry

- Builtin: `validated_builtin_rules`

## Proof blocks

| Block | Status | Evidence | Summary |
| --- | --- | --- | --- |
| `FRONTEND` | **PROVED** | STRUCTURAL | both raw TTIR modules were parsed and verified by pinned libtriton before lifting into the common semantic IR |
| `ABI` | **PROVED** | TRUSTED_AXIOM | physical parameters are aligned to explicit logical roles |
| `INDEX` | **PROVED** | BOUNDED_EXHAUSTIVE | all launch programs and lanes were exhaustively enumerated |
| `MASK` | **PROVED** | BOUNDED_EXHAUSTIVE | both store masks select the same logical output domain |
| `COVERAGE` | **PROVED** | BOUNDED_EXHAUSTIVE | the active lanes cover every contracted output element exactly once |
| `RACE_FREEDOM` | **PROVED** | BOUNDED_EXHAUSTIVE | active store addresses are injective on both sides |
| `ADDRESS` | **PROVED** | BOUNDED_EXHAUSTIVE | all corresponding logical elements write the same Output offsets |
| `LOAD` | **PROVED** | BOUNDED_EXHAUSTIVE, TRUSTED_AXIOM | all output values depend on the same logical input/scalar leaves |
| `DEFINEDNESS` | **PROVED** | STRUCTURAL | all partial floating operations have structurally proved constant domains |
| `COMPUTE` | **PROVED** | CONGRUENCE | every required pair of symbolic output values belongs to the same e-class |
| `STORE` | **PROVED** | BOUNDED_EXHAUSTIVE, CONGRUENCE | same domain, address, value, and frame condition imply equal final Output memory |

## Proof summary

Finite domain: 1 active outputs, 128 lhs lanes, 1 rhs lanes.
E-graph: `egglog 13.2.0`, 2 e-nodes, 2 e-classes, 0 iterations, stop reason `ROOTS_ALREADY_CONGRUENT`.
Initial unmatched roots: 0; rewrite phase: `NO_REWRITE_REQUIRED`; unmatched after predicate-derived rewrites: 0.
Matched rules: none

## Trust boundary

- Soundness level: `formal_under_declared_predicates`
- Conditional on: declared predicate: abi.Input0
- Conditional on: declared predicate: abi.Output
- Conditional on: declared predicate: binding.X
- Conditional on: declared predicate: disjoint.0
- PairSpec role correspondence
- PairSpec fixed shape/stride/launch bindings
- PairSpec no-alias declarations
- ABSTRACT_FLOAT interprets floating operations over exact mathematical values
- ETV IR evaluator and memory-token model
- ETV TTIR-to-semantic-IR lifting implementation

This result establishes: equal final Output memory for all abstract input values in the enumerated launch domain

It does not prove:

- IEEE-754 or bitwise GPU equality
- formal correctness of the raw-TTIR-to-ETV-IR lifting implementation
- parametric shapes or unbounded loops
- concurrent/shared-memory semantics
- allocated-buffer bounds or GPU memory safety
