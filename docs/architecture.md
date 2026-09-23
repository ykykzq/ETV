# Architecture

There is one immutable typed semantic IR. Frontend snapshots, Z3 terms and egglog
terms adapt that IR; they are not competing program models.

| Modules | Responsibility |
| --- | --- |
| `pairspec`, `jsonio` | Strict contracts, typed configuration expressions and hashes. |
| `frontend`, `native/snapshot.cpp` | Official parse/verify, object traversal and snapshots. |
| `ir`, `lift` | Types, storage references, expressions, stores and static lane maps. |
| `semantics`, `smt` | Exact evaluation, finite integers/casts and proof queries. |
| `memory` | Step composition, writers, coverage, addresses and definedness. |
| `rewrites`, `eqsat` | Per-run admission and bounded official egglog equality. |
| `partition`, `partition_proof`, `heuristics` | PDG proposals, boundary checks and fallback. |
| `counterexample` | Independent exact replay of value models. |
| `verify`, `result`, `report`, `logging`, `cli` | State machine and public outputs. |

Snapshots retain module/function symbols, argument order/types, operation ordinals,
SSA IDs, operands/results, attributes, blocks/regions, locations, assembly and
SHA-256. Assembly is diagnostic; regex does not reconstruct semantics. A thin C++
adapter fills Python API gaps using official MLIR objects in the same verified
module. Where Triton pointer accessor symbols are hidden, structural subelement
traversal and official type-object comparison recognize the default address space.

Tensor values are static shapes plus scalar expressions indexed by flattened lane.
Storage retains side, launch, physical argument, logical role and view offset.
Composition resolves intermediate loads from earlier steps. Corresponding external
reads are shared only after their memory alignment obligations close.

## Proof Responsibilities

Z3 closes memory/definedness queries and validates exact algebraic rules. Egglog
establishes computation equality by congruence and admitted rules. An exact Z3
value query may provide a replayed counterexample; `unsat` alone does not replace
egglog equality. Heuristics cannot discharge mandatory obligations.

The PDG covers composed acyclic scalar observations. Edges record data dependencies,
select/load-mask control and read-address dependencies. Proposed boundaries name
known nodes and previously proved dependencies. Each boundary equality is checked
by egglog and can supply a ground equality to subsequent goals. Whole observation
roots are checked at the end. This is not a general loop/interprocedural PDG.

## References

Object traversal follows [MLIR's language reference](https://mlir.llvm.org/docs/LangRef/)
and [bindings](https://mlir.llvm.org/docs/Bindings/Python/), with pinned
[Triton](https://github.com/triton-lang/triton/tree/v3.7.1) dialect registration.
The separation of lifting, obligations and verdicts follows translation validation
as exemplified by [Alive2](https://github.com/AliveToolkit/alive2). Query interpretation
uses [Z3](https://github.com/Z3Prover/z3) and its [guide](https://github.com/microsoft/z3guide).
Congruence and bounded saturation use [egglog-python](https://github.com/egraphs-good/egglog-python),
[Better Together](https://arxiv.org/abs/2304.04332) and [egg](https://arxiv.org/abs/2004.03082).

[Verified Lifting of Deep Learning Operators](https://arxiv.org/abs/2412.20992),
[Verify Implementation Equivalence of Large Models](https://arxiv.org/abs/2603.21851)
and [Gimlet's verification description](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels)
motivate common scalar semantics and separating generated proposals from evidence.
Numerical tests and model output do not become formal validation in ETV.
