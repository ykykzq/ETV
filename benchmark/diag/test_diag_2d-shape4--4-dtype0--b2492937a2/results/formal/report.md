# ETV verification report: build/upstream/ntops/tests/test_diag.py::test_diag_2d[shape4--4-dtype0]

- Status: **DISPROVED**
- Reason: `COMPUTE_MISMATCH`
- Semantic mode: `abstract_float`
- Scope: fixed-specialization single-store bounded translation validation
- Soundness: `formal_under_declared_predicates`

## Assumptions

- `TRUSTED_AXIOM`: X = 1 (PairSpec.predicates.bindings)
- `TRUSTED_AXIOM`: lhs.arg1 = 5 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: disjoint(Input0, Output) (PairSpec.predicates.disjoint)

## LLM-only context

- This pair was captured from the same specialized ntops pytest node; pointer roles are mapped by one-store dataflow and remaining ABI order.
- Proof relevance: `informational_only`.

## Declared formal predicates

- `abi.Input0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Output`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.X`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg1`: builtin / assumed (`TRUSTED_AXIOM`)
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
| `LOAD` | **UNKNOWN** | - | input dependency frontiers differ; compute proof will decide observability |
| `DEFINEDNESS` | **PROVED** | STRUCTURAL | all partial floating operations have structurally proved constant domains |
| `COMPUTE` | **DISPROVED** | - | a concrete abstract-value assignment makes the computations differ |

## Counterexample

```json
{
  "assignment": {
    "read(Input0,0)": "1",
    "read(Input0,20)": "0"
  },
  "kind": "ABSTRACT_VALUE_MODEL",
  "lhs_expression": "read(Input0, 0)",
  "lhs_value": "1",
  "logical_index": 0,
  "rhs_expression": "read(Input0, 20)",
  "rhs_value": "0"
}
```

## Proof summary

Finite domain: 1 active outputs, 128 lhs lanes, 1 rhs lanes.
E-graph: `egglog 13.2.0`, 4 e-nodes, 4 e-classes, 1 iterations, stop reason `SATURATED`.
Initial unmatched roots: 1; rewrite phase: `ALGEBRAIC_REWRITES`; unmatched after predicate-derived rewrites: 1.
Matched rules: none

### E-graph Iterations

| Phase/iteration | Before | After | Updated | Matched rules |
| --- | --- | --- | --- | --- |
| ALGEBRAIC_REWRITES #1 | 4/4 | 4/4 | False | none |

## Trust boundary

- Soundness level: `formal_under_declared_predicates`
- Conditional on: declared predicate: abi.Input0
- Conditional on: declared predicate: abi.Output
- Conditional on: declared predicate: binding.X
- Conditional on: declared predicate: binding.lhs.arg1
- Conditional on: declared predicate: disjoint.0
- PairSpec role correspondence
- PairSpec fixed shape/stride/launch bindings
- PairSpec no-alias declarations
- ABSTRACT_FLOAT interprets floating operations over exact mathematical values
- ETV IR evaluator and memory-token model
- ETV TTIR-to-semantic-IR lifting implementation

This result establishes: the reported concrete witness violates observable equivalence in ABSTRACT_FLOAT semantics

It does not prove:

- IEEE-754 or bitwise GPU equality
- formal correctness of the raw-TTIR-to-ETV-IR lifting implementation
- parametric shapes or unbounded loops
- concurrent/shared-memory semantics
- allocated-buffer bounds or GPU memory safety
