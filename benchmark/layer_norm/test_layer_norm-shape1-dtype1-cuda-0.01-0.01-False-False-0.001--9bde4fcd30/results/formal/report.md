# ETV verification report: build/upstream/ntops/tests/test_layer_norm.py::test_layer_norm[shape1-dtype1-cuda-0.01-0.01-False-False-0.001]

- Status: **UNKNOWN**
- Reason: `TTIR_REGION_SEMANTICS_UNSUPPORTED`
- Semantic mode: `abstract_float`
- Scope: fixed-specialization single-store bounded translation validation
- Soundness: `formal_under_declared_predicates`

## Assumptions

- `TRUSTED_AXIOM`: X = 417 (PairSpec.predicates.bindings)
- `TRUSTED_AXIOM`: lhs.arg3 = 1 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg5 = 417 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg8 = 417 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: rhs.arg4 = 1 (PairSpec.predicates.side_bindings.rhs)
- `TRUSTED_AXIOM`: disjoint(Input0, Input1, Input2, Output) (PairSpec.predicates.disjoint)

## LLM-only context

- This pair was captured from the same specialized ntops pytest node; pointer roles are mapped by one-store dataflow and remaining ABI order.
- Proof relevance: `informational_only`.

## Declared formal predicates

- `abi.Input0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Input1`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Input2`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Output`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.X`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg3`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg5`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg8`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.rhs.arg4`: builtin / assumed (`TRUSTED_AXIOM`)
- `disjoint.0`: builtin / assumed (`TRUSTED_AXIOM`)

## Rewrite registry

- Builtin: `validated_builtin_rules`

## Proof blocks

| Block | Status | Evidence | Summary |
| --- | --- | --- | --- |
| `FRONTEND` | **UNKNOWN** | - | region operation scf.for is parsed but not yet lifted |

## Trust boundary

- Soundness level: `formal_under_declared_predicates`
- Conditional on: declared predicate: abi.Input0
- Conditional on: declared predicate: abi.Input1
- Conditional on: declared predicate: abi.Input2
- Conditional on: declared predicate: abi.Output
- Conditional on: declared predicate: binding.X
- Conditional on: declared predicate: binding.lhs.arg3
- Conditional on: declared predicate: binding.lhs.arg5
- Conditional on: declared predicate: binding.lhs.arg8
- Conditional on: declared predicate: binding.rhs.arg4
- Conditional on: declared predicate: disjoint.0
- PairSpec role correspondence
- PairSpec fixed shape/stride/launch bindings
- PairSpec no-alias declarations
- ABSTRACT_FLOAT interprets floating operations over exact mathematical values
- ETV IR evaluator and memory-token model

This result establishes: no equivalence or inequivalence conclusion; the reason identifies the first unmet obligation

It does not prove:

- IEEE-754 or bitwise GPU equality
- formal correctness of the raw-TTIR-to-ETV-IR lifting implementation
- parametric shapes or unbounded loops
- concurrent/shared-memory semantics
- allocated-buffer bounds or GPU memory safety
