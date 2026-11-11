# PairSpec v3

Only `etv-pair-v3` is accepted. Duplicate JSON keys and unknown fields are errors.
The files under `examples/` provide complete, runnable inputs.

| Field | Contract |
| --- | --- |
| `metadata` | `pair_id`, `semantic_mode: abstract_float`, `limits`; optional `execution_scope`, `target`, `partition`, `llm`. |
| `programs` | Nonempty `lhs` and `rhs` arrays of captured TTIR launches; host steps require an explicit alternate scope. |
| `assumptions` | `for_llm` strings, with no formal meaning. |
| `predicates` | `roles`, `bindings`, `parameters`, `constraints`, `disjoint`, `custom`. |
| `observation` | `role`, integer expression `numel`, `require_full_coverage`, `require_disjoint`; optional `layouts`, `value_representation`. |
| `rewrites` | File references, optionally with `expected_sha256`. |

`metadata.execution_scope` is `ttir-only` by default, including for older files
that omit the field. The benchmark generator always writes it explicitly.
This scope rejects every `host_op` step with `HOST_STEP_OUTSIDE_TTIR_ONLY`.
It does not add host initialization, bind host-produced buffers to constants,
substitute final observations, or weaken full-output coverage. Captured host
effects may remain as provenance, but are not premises or executable proof steps.

The separate `ttir-with-host-ops` scope must be chosen explicitly. Its verdict is
about the composition of TTIR with a trusted model of the captured host operations,
not TTIR alone. Reports retain the scope so the two experiments cannot be combined
as one TTIR-only result. Neither scope proves the capture process, PyTorch's host
implementation, or its CUDA library kernels correct.

## Launches and ABI

Launch fields are `id`, `file`, exact `function`, nonnegative `step`,
`grid: {"programs": EXPR}`, `stores`, `abi`, `bindings`. Steps are ordered within
a side. Same-step launches read the prior state and commit together only after
the verifier rules out shared read/write storage roles within that step; possible
cross-program/lane or inter-launch dependencies return
`UNKNOWN(INPLACE_MEMORY_DEPENDENCY_UNPROVED)`. A multi-store
kernel is one launch with multiple store entries. Store indices count `tt.store`
operations in the selected acyclic function; all effects, including internal
writes, must be declared.

ABI maps logical roles to physical arguments `arg0`, `arg1`, etc. Each role maps
to one endpoint object, or a nonempty list of endpoint objects for views of the
same storage. Every endpoint retains its own element `offset`; listing two
endpoints does not declare those views disjoint. A physical
argument cannot have multiple endpoints or also have a binding. Roles declare
`kind: buffer|scalar` and `element: abstract_float|bool|iN|index`; modeled memory
supports floating point, boolean and finite integer elements. Endpoint examples:

```json
{
  "Input": {"kind": "block", "name": "arg0", "offset": 0},
  "Alpha": {"kind": "scalar_block", "name": "arg1", "index": 0},
  "Beta": {"kind": "scalar", "name": "arg2"}
}
```

For example, `"Input": [{"kind":"block","name":"arg0","offset":0},
{"kind":"block","name":"arg1","offset":4}]` identifies two actual pointer
arguments with the same underlying logical storage and different element offsets.
Duplicate physical arguments are rejected both within and across roles. Merely
declaring a view does not initialize storage written by the program: unsupported
read-before-write or in-place dependencies remain UNKNOWN. This includes internal
storage initialized by an earlier launch: prior initialization does not prove
that a later in-place update has no cross-program/lane read/write conflict.
Read-only aliases and separate, ordered producer/consumer launches remain supported.

`block` is a buffer view with an element offset; `scalar` is passed by value;
`scalar_block` identifies one pointer element as a logical scalar. Loads from it
must address exactly `offset + index`. Disjoint premises must also cover scalar
blocks that could alias writes. Store roles must match their pointer storage.
Bindings are integer expressions keyed by logical names or physical argument names.
Cycles and shadowing are rejected. Bind runtime integers explicitly to establish
their fixed or bounded symbolic domains.

Physical launch bindings also accept exact finite real constants, for example
`"arg3": {"constant": "13421773/134217728", "type": "abstract_float"}`.
Runtime Python floats are first rounded to the recorded fp32/fp64 scalar ABI.
Shape/configuration expressions and global bindings remain integer/boolean expressions.

## Captured host steps and output views

A host step is accepted only when metadata explicitly contains
`"execution_scope": "ttir-with-host-ops"`. It references an actual event in a
SHA-256-pinned runtime capture:

```json
{
  "kind": "host_op", "id": "lhs.host.0", "file": "capture.json",
  "expected_sha256": "SHA256_OF_CAPTURE", "event": 0, "step": 0,
  "abi": {"Internal": {"kind": "block", "name": "arg0", "offset": 0}},
  "stores": [{"index": 0, "role": "Internal"}]
}
```

`event` indexes `host_operations` in an `etv-host-operations-v1` capture extension.
Arguments are event inputs in order, followed by its output. The verifier checks
event identity, operation whitelist, tensor views, scalar conversion and ABI;
it then lowers the event to the same memory IR as TTIR. This is modeled host
semantics, not compiler-generated TTIR or a proof of the PyTorch implementation.
Its implementation is part of the verifier's trusted semantic model; an empty
`trusted_axioms` list refers to user rewrite axioms and does not remove this trust
boundary. Reports preserve that distinction.
Supported effects are fixed scalar initialization, copy (including broadcast
views), scalar division, and constant padding. Nonfinite values and unsupported
casts or aliasing remain UNKNOWN. Arbitrary tensor contents are never frozen.

Host steps and TTIR launches share ordered `step` values. Both sides may contain
multiple steps under complete bounded enumeration; symbolic multi-launch proofs
remain unsupported. Reading an internal result before its producer commits is
not silently converted to an external input.

Optional `observation.layouts` supplies both `lhs` and `rhs` descriptors, each
with `shape`, `stride` and element `offset`. It maps row-major logical output
indices to each side's physical storage. Thus a transposed or strided output is
compared by logical element, with coverage and overlap checks preserved. The
default remains a contiguous physical range. Layouts with unsupported or
excessive enumeration return UNKNOWN, never a sampled proof.
Output-layout descriptions are available in either execution scope: they identify
the same logical observation without supplying missing writes or initialized values.

## Explicit Boolean/numeric output observation

The default compares identically typed observable values in a common buffer role.
An opt-in `value_representation` permits a Boolean output and a numeric output to
be compared as logical predicate values, while keeping their physical types distinct:

```json
{
  "role": "Output",
  "numel": 1024,
  "require_full_coverage": true,
  "require_disjoint": [],
  "value_representation": {
    "kind": "boolean_numeric_01",
    "lhs": {"role": "Output_lhs", "encoding": "identity"},
    "rhs": {"role": "Output_rhs", "encoding": "bool_to_numeric_01"}
  }
}
```

Here `predicates.roles.Output_lhs` is a buffer with element `abstract_float`,
and `Output_rhs` is a buffer with element `bool`. Launch ABI and stores use the
physical roles. `Output` names the common logical observation and need not be a
physical role. The two sides may be reversed. Exactly one side must have each
encoding, with those exact physical element types; other conversions are rejected.

The Boolean injection maps `false` to exact numeric 0 and `true` to exact numeric
1. It is injective because 0 and 1 are distinct. The numeric side is unchanged:
2 versus `true` is unequal, and no premise assumes a numeric output is already
0 or 1. The verifier applies this mapping only to final comparison roots after
checking writers, masks, addresses, definedness and the full original output
domain. This mode requires `require_full_coverage: true`; it creates no TTIR or
host steps and supplies no initialization or missing writes. Optional output
layouts continue to map the same full logical index domain to physical storage.
Within-side read/write separation premises are still required; distinct output
role names alone do not require cross-side physical separation.

Reports record `comparison_contract.kind: logical-value-under-encoding`, each
side's physical role, element type and encoding, and successful proofs use
`LOGICAL_BOOLEAN_OUTPUT_EQUIVALENT`. The contract is an explicit definition of
the comparison, visible as `observation.value_representation`, rather than a
trusted equality assumption. Other proofs use `typed-observable-values` and
`OBSERVABLE_MEMORY_EQUIVALENT`. Neither result claims byte-identical memory or
output dtype equality: `abstract_float` already abstracts floating-point widths
and does not establish IEEE rounding, NaN or signed-zero behavior. Explicit TTIR
infinities use the partial extended-real model in [infinity-model.md](infinity-model.md);
external floating inputs still have the declared finite-real domain.

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
`abi.<launch>.<role>` (with `.argN` appended when that role has multiple endpoints),
`binding.<scope>.<name>`, `parameter.<name>`, sorted
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

Optional `target.boolean_storage: "canonical_i8"` declares that boolean storage
bytes are exactly 0 or 1. This enables loads through a bool-to-byte pointer cast
and is recorded as the supplied premise `target.boolean_storage`; it is not a
fact established about host memory. Other values are rejected. Benchmark pairing
adds this premise when a capture contains boolean memory roles.

Partition options are `enabled`, `provider` (`none`, `deterministic`, `llm`) and
`max_parts`. LLM defaults to disabled; enabling requires `provider: openai_compatible`,
`model`, `base_url_env`, `api_key_env`, `timeout_ms`. Credentials are read from
environment variables, not artifact contents. See [rewrites](rewrites.md).
