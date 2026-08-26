# ETV verification report: build/upstream/ntops/tests/test_diag.py::test_diag_1d[5--1-dtype0]

- Status: **UNKNOWN**
- Reason: `CAST_EQUIVALENCE_NOT_REWRITTEN`
- Semantic mode: `abstract_float`
- Scope: fixed-specialization single-store bounded translation validation
- Soundness: `formal_under_declared_predicates`

## Assumptions

- `TRUSTED_AXIOM`: X = 36 (PairSpec.predicates.bindings)
- `TRUSTED_AXIOM`: lhs.arg1 = 5 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: rhs.arg2 = 36 (PairSpec.predicates.side_bindings.rhs)
- `TRUSTED_AXIOM`: disjoint(Input0, Output) (PairSpec.predicates.disjoint)

## LLM-only context

- This pair was captured from the same specialized ntops pytest node; pointer roles are mapped by one-store dataflow and remaining ABI order.
- Proof relevance: `informational_only`.

## Declared formal predicates

- `abi.Input0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Output`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.X`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg1`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.rhs.arg2`: builtin / assumed (`TRUSTED_AXIOM`)
- `disjoint.0`: builtin / assumed (`TRUSTED_AXIOM`)

## Rewrite registry

- Builtin: `validated_builtin_rules`

## Proof blocks

| Block | Status | Evidence | Summary |
| --- | --- | --- | --- |
| `FRONTEND` | **PROVED** | STRUCTURAL | both raw TTIR modules were parsed and verified by pinned libtriton before lifting into the common semantic IR |
| `ABI` | **PROVED** | TRUSTED_AXIOM | physical parameters are aligned to explicit logical roles |
| `CAST` | **UNKNOWN** | - | integer cast differences were preserved but no admitted rewrite established their equivalence |

## Proof summary


## Trust boundary

- Soundness level: `formal_under_declared_predicates`
- Conditional on: declared predicate: abi.Input0
- Conditional on: declared predicate: abi.Output
- Conditional on: declared predicate: binding.X
- Conditional on: declared predicate: binding.lhs.arg1
- Conditional on: declared predicate: binding.rhs.arg2
- Conditional on: declared predicate: disjoint.0
- PairSpec role correspondence
- PairSpec fixed shape/stride/launch bindings
- PairSpec no-alias declarations
- ABSTRACT_FLOAT interprets floating operations over exact mathematical values
- ETV IR evaluator and memory-token model
- ETV TTIR-to-semantic-IR lifting implementation

This result establishes: no equivalence or inequivalence conclusion; the reason identifies the first unmet obligation

It does not prove:

- IEEE-754 or bitwise GPU equality
- formal correctness of the raw-TTIR-to-ETV-IR lifting implementation
- parametric shapes or unbounded loops
- concurrent/shared-memory semantics
- allocated-buffer bounds or GPU memory safety
