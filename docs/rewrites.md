# Rewrites and Proposals

Rules use strict `etv-rewrite-v1` JSON. Fields are `id`, `kind` (`algebraic`,
`layout`, `relation`), `lhs`, `rhs`, `requires`, and `validation: {"method":"z3"}`
or `{"method":"trusted"}`. A requested method is not evidence. Patterns are typed
`pvar`, `const`, or `op` with `args`. Variables agree across sides, sorts/arity are
checked, and an unrestricted variable LHS is rejected.

See [the complete example](../examples/rules/fadd_zero.json). Reference files with
`"rewrites": [{"file":"rules.json","expected_sha256":"..."}]`. Actual hashes
are always computed. Gates name stable PairSpec predicate IDs; unknown gates and
duplicate rule IDs are errors. Registries never leak between runs.

Z3 algebraic admission refutes equality and matching definedness for all pattern
inputs. Only `unsat` under an exact encoder is `formal`. Incorrect, unsupported
or unresolved algebraic rules are rejected. Gated layout/relation rules may be
`admitted_unverified`; generated rules are explicitly unverified too. A match
makes the soundness level conditional and lists the rule as a trusted axiom.
Reports retain expressions, source, hash, validator, admission and match counts.

## Optional LLM

```json
{
  "partition": {"enabled": true, "provider": "llm", "max_parts": 8},
  "llm": {
    "enabled": true, "provider": "openai_compatible", "model": "MODEL_NAME",
    "base_url_env": "ETV_LLM_BASE_URL", "api_key_env": "ETV_LLM_API_KEY",
    "timeout_ms": 10000
  }
}
```

The base URL includes its API prefix (such as `/v1`); ETV appends
`/chat/completions`. Two bounded requests first select node pairs, then propose
rewrites/partitions. The payload contains semantic expressions, opaque node IDs,
predicate IDs and heuristic assumptions. Credentials never enter reports. Responses
cannot specify paths or commands.

Invalid JSON/schema, unknown nodes/gates, inconsistent dependencies, network failure
or missing credentials cause fallback. Partition nodes must exist, types must
agree, dependencies must be acyclic and inside the selected subgraphs. Boundary
equality still requires proof. The deterministic provider proposes shared
expressions without a network request. LLM assistance is disabled by default.
