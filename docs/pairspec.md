# PairSpec v3

Only `etv-pair-v3` is accepted. Duplicate JSON keys and unknown fields are errors.
The files under `examples/` provide complete, runnable inputs.

| Field | Contract |
| --- | --- |
| `metadata` | `pair_id`, `semantic_mode: abstract_float`, `limits`; optional `target`, `partition`, `llm`. |
| `programs` | Nonempty `lhs` and `rhs` physical-launch arrays. |
| `assumptions` | `for_llm` strings, with no formal meaning. |
| `predicates` | `roles`, `bindings`, `parameters`, `constraints`, `disjoint`, `custom`. |
| `observation` | `role`, integer expression `numel`, `require_full_coverage`, `require_disjoint`. |
| `rewrites` | File references, optionally with `expected_sha256`. |

## Launches and ABI

Launch fields are `id`, `file`, exact `function`, nonnegative `step`,
`grid: {"programs": EXPR}`, `stores`, `abi`, `bindings`. Steps are ordered within
a side. Same-step launches read the prior state and commit together. A multi-store
kernel is one launch with multiple store entries. Store indices count `tt.store`
operations in the selected acyclic function; all effects, including internal
writes, must be declared.

ABI maps logical roles to physical arguments `arg0`, `arg1`, etc. A physical
argument cannot have multiple endpoints or also have a binding. Roles declare
`kind: buffer|scalar` and `element: abstract_float|bool|iN|index`; modeled memory
is floating point only. Endpoint examples:

```json
{
  "Input": {"kind": "block", "name": "arg0", "offset": 0},
  "Alpha": {"kind": "scalar_block", "name": "arg1", "index": 0},
  "Beta": {"kind": "scalar", "name": "arg2"}
}
```

`block` is a buffer view with an element offset; `scalar` is passed by value;
`scalar_block` identifies one pointer element as a logical scalar. Loads from it
must address exactly `offset + index`. Disjoint premises must also cover scalar
blocks that could alias writes. Store roles must match their pointer storage.
Bindings are integer expressions keyed by logical names or physical argument names.
Cycles and shadowing are rejected. Bind runtime integers explicitly to establish
their fixed or bounded symbolic domains.

## Expressions and Premises

```text
EXPR := integer | boolean | {"var": NAME} | {"op": OP, "args": [EXPR, ...]}
OP   := add | sub | mul | div | rem | ceildiv
      | eq | ne | lt | le | gt | ge | and | or | not | select
```

Arity/sorts are checked. Configuration integers are mathematical indices; TTIR
fixed-width arithmetic is separate. Division truncates toward zero; `ceildiv`
requires a positive divisor. Expressions must be defined throughout the domain.

```json
{
  "parameters": {
    "a": {"type": "i32", "min": 1, "max": 16},
    "b": {"type": "i32", "min": 1, "max": 16},
    "c": {"type": "i32", "min": 1, "max": 256}
  },
  "constraints": [{"id": "shape.product", "expr": {"op": "eq", "args": [
    {"op": "mul", "args": [{"var": "a"}, {"var": "b"}]}, {"var": "c"}
  ]}}]
}
```

Bounds must fit the parameter's signed width. Constraints and custom predicates
with `encoder: z3` require boolean expressions. Custom `encoder: trusted` predicates
are listed as trusted axioms; natural language is never translated into a solver
constraint. The domain must be nonempty.

All are supplied premises, not facts proved about the host. Stable IDs are
`abi.<launch>.<role>`, `binding.<scope>.<name>`, `parameter.<name>`, sorted
`disjoint.<role>...`, and explicit constraint/custom IDs. IDs are globally unique;
rewrite gates can name only existing IDs.

## Paths and Options

Files resolve within the PairSpec directory; escapes via `..` or symlinks are
rejected unless `--allow-external-paths` explicitly permits external read-only
inputs. Actual hashes are frozen and rechecked. Expected rewrite hashes must match.

Required positive limits are `smt_timeout_ms`, `egraph_timeout_ms`, `max_iterations`,
`max_enodes`. Optional `max_instances` defaults to 100000 and bounds static shapes
and complete enumeration. Exhaustion is `UNKNOWN(RESOURCE_LIMIT)`, never sampling.
`target.index_bits` is 32 or 64, required for actual index casts.

Partition options are `enabled`, `provider` (`none`, `deterministic`, `llm`) and
`max_parts`. LLM defaults to disabled; enabling requires `provider: openai_compatible`,
`model`, `base_url_env`, `api_key_env`, `timeout_ms`. Credentials are read from
environment variables, not artifact contents. See [rewrites](rewrites.md).
