# ETV verification report: build/upstream/ntops/tests/test_argsort.py::test_argsort[True-shape8-0-dtype8-cuda-0.001-0.001]::r000-e000-o000

- Status: **UNKNOWN**
- Reason: `TTIR_REGION_SEMANTICS_UNSUPPORTED`
- Semantic mode: `abstract_float`
- Scope: fixed-specialization ordered multi-launch translation validation
- Soundness: `formal_under_declared_predicates`

## Assumptions

- `TRUSTED_AXIOM`: X = 258 (PairSpec.predicates.bindings)
- `TRUSTED_AXIOM`: lhs.arg1 = 6 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg10 = 43 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg11 = 1 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg2 = 6 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg3 = 43 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg5 = 43 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg6 = 1 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg7 = 6 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: lhs.arg9 = 43 (PairSpec.predicates.side_bindings.lhs)
- `TRUSTED_AXIOM`: disjoint(Input0, Output) (PairSpec.predicates.disjoint)
- `TRUSTED_AXIOM`: rhs.launch000.store000.arg2 = 43 (PairSpec.metadata.launches.rhs.bindings)
- `TRUSTED_AXIOM`: rhs.launch000.store000.arg3 = 6 (PairSpec.metadata.launches.rhs.bindings)
- `TRUSTED_AXIOM`: rhs.launch001.store000.arg2 = 258 (PairSpec.metadata.launches.rhs.bindings)

## LLM-only context

- The RHS launches were captured in execution order from one compiled reference call.
- RHS launch boundaries are fixed; the LLM may only select corresponding LHS expression paths.
- Intermediate storage edges come from capture-v2 storage provenance and exact element offsets.
- This PairSpec observes output leaf 'compiled_output' only.
- Proof relevance: `informational_only`.

## Declared formal predicates

- `abi.Input0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.Output`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.X`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg1`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg10`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg11`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg2`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg3`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg5`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg6`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg7`: builtin / assumed (`TRUSTED_AXIOM`)
- `binding.lhs.arg9`: builtin / assumed (`TRUSTED_AXIOM`)
- `disjoint.0`: builtin / assumed (`TRUSTED_AXIOM`)
- `abi.rhs.launch000.store000.Input0`: launch_abi / assumed (`TRUSTED_AXIOM`)
- `abi.rhs.launch000.store000.Internal0`: launch_abi / assumed (`TRUSTED_AXIOM`)
- `abi.rhs.launch001.store000.Internal0`: launch_abi / assumed (`TRUSTED_AXIOM`)
- `abi.rhs.launch001.store000.Output`: launch_abi / assumed (`TRUSTED_AXIOM`)
- `launch_order.rhs.launch000.store000.step`: launch_order / assumed (`TRUSTED_AXIOM`)
- `launch_order.rhs.launch001.store000.step`: launch_order / assumed (`TRUSTED_AXIOM`)

## Rewrite registry

- Builtin: `validated_builtin_rules`

## Proof blocks

| Block | Status | Evidence | Summary |
| --- | --- | --- | --- |
| `FRONTEND` | **UNKNOWN** | - | region operation tt.reduce is parsed but not yet lifted |

## Trust boundary

- Soundness level: `formal_under_declared_predicates`
- Conditional on: declared predicate: abi.Input0
- Conditional on: declared predicate: abi.Output
- Conditional on: declared predicate: binding.X
- Conditional on: declared predicate: binding.lhs.arg1
- Conditional on: declared predicate: binding.lhs.arg10
- Conditional on: declared predicate: binding.lhs.arg11
- Conditional on: declared predicate: binding.lhs.arg2
- Conditional on: declared predicate: binding.lhs.arg3
- Conditional on: declared predicate: binding.lhs.arg5
- Conditional on: declared predicate: binding.lhs.arg6
- Conditional on: declared predicate: binding.lhs.arg7
- Conditional on: declared predicate: binding.lhs.arg9
- Conditional on: declared predicate: disjoint.0
- Conditional on: declared predicate: abi.rhs.launch000.store000.Input0
- Conditional on: declared predicate: abi.rhs.launch000.store000.Internal0
- Conditional on: declared predicate: abi.rhs.launch001.store000.Internal0
- Conditional on: declared predicate: abi.rhs.launch001.store000.Output
- Conditional on: declared predicate: launch_order.rhs.launch000.store000.step
- Conditional on: declared predicate: launch_order.rhs.launch001.store000.step
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
