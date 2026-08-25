# ETV

[中文](README_cn.md)

ETV verifies whether two raw TT IR kernels have identical observable output
memory under the assumptions declared in a PairSpec. Both inputs must be
`.ttir` or `.mlir` files:

```text
lhs.ttir -> libtriton parse/verify -> TTIR snapshot -> ETV IR --+
                                                               +-> SMT + egglog -> verdict
rhs.ttir -> libtriton parse/verify -> TTIR snapshot -> ETV IR --+
```

ETV IR is a verification-oriented semantic representation of launches,
per-lane addresses, masks, loads, arithmetic, and stores. It is not an e-graph.
Only the proof stage encodes ETV IR expressions into an egglog e-graph and uses
rewrites plus congruence closure to determine whether the two `observe_store`
roots belong to the same e-class.

## Results

- `PROVED`: observable Output memory is equivalent under all PairSpec premises;
- `DISPROVED`: a replayable address, mask, or abstract-value counterexample was found;
- `UNKNOWN`: the input is outside the semantic subset, or facts, domains, rules,
  or resources are insufficient.

The current `ABSTRACT_FLOAT` mode uses exact mathematical values. It does not
represent IEEE-754, tolerance-based, or GPU bitwise equivalence. The current MVP
targets fixed-rank, single-kernel, single-store, acyclic pointwise programs.

Integer casts remain explicit IR operators carrying source and target types
after TT IR lifting. The verifier has no default `cast(x) = x` rule. A cast
structure mismatch must be eliminated by an admitted rewrite that actually
fires; otherwise the result is `UNKNOWN(CAST_EQUIVALENCE_NOT_REWRITTEN)`. An
undeclared target-dependent `index` width likewise produces `UNKNOWN`.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,ttir]'
.venv/bin/python -m pytest
```

The `ttir` extra can be installed directly on Linux. Building Triton 3.7.1 from
source is required on macOS; see [Dependencies and Environment](docs/dependencies.md).

## Usage

Verify the real Add program pair with fixed shape `[8,16]`:

```bash
.venv/bin/python -m etv check examples/add/pair.json --out build/add
```

The left-hand side comes from ntops/ninetoothed and the right-hand side from
PyTorch/TorchInductor. Both are genuinely generated TT IR. Expected result:

```text
PROVED ntops_add_vs_torch_inductor_add: OBSERVABLE_MEMORY_EQUIVALENT
```

Verify a parameterized two-dimensional 256-lane TT IR program against a
one-dimensional 128-lane TT IR program:

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

This example proves address, mask, coverage, and computation equivalence under
`a > 0`, `b > 0`, `c > 0`, and `a * b = c`.

Parse a single TT IR file and export a stable snapshot:

```bash
.venv/bin/python -m etv parse examples/add/ttir/ntops_add.ttir \
  --out build/ntops_add.snapshot.json --no-assembly
.venv/bin/python -m etv inspect examples/add/ttir/torch_inductor_add.ttir
```

`check` and `inspect` do not accept Semantic JSON or Prims. JSON files under
`tests/fixtures/semantic` are proof-kernel unit-test fixtures that enter through
the Python-only `verify_internal_spec` API; they are not part of the user input
protocol. Raw TT IR integration fixtures live under `tests/fixtures/ttir`.

## PairSpec

PairSpec v2 strictly separates configuration, natural-language context, formal
predicates, observation targets, and rewrite libraries:

```json
{
  "format": "etv-pair-v2",
  "metadata": {
    "pair_id": "lhs_vs_rhs",
    "lhs": "lhs.ttir",
    "rhs": "rhs.ttir",
    "semantic_mode": "abstract_float",
    "frontends": {
      "lhs": {"kind": "ttir", "function": "lhs_kernel", "programs": 1},
      "rhs": {"kind": "ttir", "function": "rhs_kernel", "programs": 1}
    }
  },
  "assumptions": {"for_llm": []},
  "predicates": {
    "abi": {
      "Output": {
        "lhs": {"kind": "block", "name": "arg0"},
        "rhs": {"kind": "block", "name": "arg0"}
      }
    },
    "bindings": {"n": 128},
    "disjoint": [],
    "custom": []
  },
  "observation": {
    "output_role": "Output",
    "output_numel": {"var": "n"},
    "require_full_coverage": true,
    "require_disjoint": []
  },
  "rewrites": []
}
```

TT IR does not contain the host launch grid, so `programs` must be supplied
explicitly. It may also be a symbolic expression such as `ceildiv(c, 256)`.
`assumptions.for_llm` never participates in formal proof. ABI, shape,
cross-program relations, and no-alias information must be encoded as
`predicates`. See [TT IR Input Format](docs/ttir_input.md) and
[Complete Verification Process](docs/verification_process.md).

## Equality Saturation and LLM Assistance

Built-in floating-point algebraic rules are proved over Z3 Reals before use;
explicit-width integer cast rules are checked with Z3 integer formulas. ETV
derives layout, address, mask, and scalar ABI relation rules from PairSpec
predicates and proves their applicability with parameterized SMT queries. An
equivalence goal is closed only when a rule actually matches in egglog and
merges its roots. User rules live in a separate `etv-rewrite-v1` file and are
tagged with a `user:<path>` source and the file's SHA-256 digest. A new
non-algebraic user or LLM rule may be admitted as trusted under the current
policy, but if it fires the report marks it `admitted_unverified` and lowers
`soundness.level` to `conditional_on_unverified_rewrites`.

Optional subgraph partitioning lets an LLM scan the complete left and right
programs and propose corresponding boundaries. ETV itself checks root coverage,
path uniqueness, left/right dependency topology, acyclicity, and types before
verifying partitions in dependency order. The LLM never decides equivalence.

## Repository Layout

```text
etv/ir.py                     Common semantic IR lifted from TT IR
etv/casts.py                  Integer cast type metadata and finite-width semantics
etv/ttir/libtriton.py         Pinned libtriton parsing, verification, and snapshots
etv/ttir/lift.py              Semantic lifting from TT IR to ETV IR
etv/schema.py                 TT IR PairSpec and internal test-fixture schemas
etv/parametric.py             Parameterized SMT obligations and predicate-derived rules
etv/rules.py                  Built-in rule library and predicate requirements
etv/egraph.py                 egglog encoding, saturation, and application logs
etv/partition.py              LLM paired-subgraph proposals and machine validation
etv/verify.py                 End-to-end proof orchestration
tools/extract_add_pair.py     ntops/TorchInductor dual-TT-IR extraction
examples/add/                 Program pair, PairSpec, sources, and provenance hashes
```

Further reading: [Architecture](docs/architecture.md),
[Complete Verification Process](docs/verification_process.md),
[Implementation Status](docs/implementation_status.md), and
[Real Add Verification](docs/add_validation.md).
