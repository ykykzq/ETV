# ETV

ETV (Equivalence for TTIR Verification) is a minimum viable translation
validator for fixed, specialized Triton-kernel semantics. It accepts raw TTIR
through pinned Triton/libtriton 3.7.1 or strictly typed Semantic TTIR JSON, and
returns exactly one of:

- `PROVED`: observable Output memory is equal under all declared assumptions.
- `DISPROVED`: a replayable address, mask, or abstract-value counterexample exists.
- `UNKNOWN`: an assumption, semantic model, rule, or resource obligation is missing.

The current MVP machine-checks the `add` proof described in the project research
archive: a stride-aware 2-D ntops implementation is compared with an Inductor
linear implementation, including the scalar-argument versus 0-D-tensor ABI
difference. It also parses and proves a raw-TTIR mul+add versus `math.fma` pair,
proves an FMA Semantic JSON variant, and rejects three deliberate mutations.

## What `PROVED` means

For a fixed `PairSpec`, ETV exhaustively enumerates every launched program/lane,
proves mask, coverage, address injectivity, and output-address correspondence,
then uses a joint `egglog==13.2.0` e-graph to prove symbolic output values equal
for arbitrary logical input values. Every admitted algebraic rule is replayed
through Z3 by checking that `lhs != rhs` is unsatisfiable over mathematical reals.
All admitted rules participate in one unified saturation; rules are no longer
selected only from operations observed in the unmatched roots.

The result is conditional on the role, shape, stride, launch, and no-alias facts
in the spec. Floating-point operations use `ABSTRACT_FLOAT`; a proof is not an
IEEE-754, tolerance, or bitwise-GPU result. Raw syntax and IR validity are checked
by libtriton; ETV's TTIR-to-Semantic-TTIR lifting remains part of the trusted
implementation boundary.
Logical blocks are modeled as total maps, so allocated-buffer bounds and GPU
memory safety are also outside the current claim.

PairSpec may additionally declare fact-gated trusted rewrites. Their gates are
checked, but the equalities themselves are intentionally not verified yet. A
matched trusted rewrite is reported as `TRUSTED_AXIOM` and can make an incorrect
`PROVED` result possible if the declaration is unsound.

See [Complete verification process](docs/verification_process.md) for the precise
claim, input contract, frontend boundary, rule admission, and result semantics.

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

Run the positive add proof and create both machine and human reports:

```bash
.venv/bin/python -m etv check \
  examples/specs/add_proved.json \
  --out build/add_proved
```

Expected summary:

```text
PROVED add_ntops_2d_vs_inductor_linear: OBSERVABLE_MEMORY_EQUIVALENT
```

Exercise all three outcomes:

```bash
.venv/bin/python -m etv check examples/specs/add_bad_compute.json
.venv/bin/python -m etv check examples/specs/add_missing_alias.json
.venv/bin/python -m etv validate-rule fma_def
```

On Linux with Python 3.11+, install `.[dev,ttir]` and exercise the raw frontend:

```bash
.venv/bin/python -m etv parse examples/ttir/add_mul.ttir \
  --out build/add_mul.snapshot.json
.venv/bin/python -m etv check examples/specs/add_raw_ttir_proved.json
```

PyPI does not publish macOS Triton wheels. The tested source-build procedure is
documented in [Dependencies and environment](docs/dependencies.md).

`check` exit codes are `0` for `PROVED`, `1` for `DISPROVED`, and `2` for
`UNKNOWN` or invalid input. `--out` writes deterministic `report.json` and
`report.md` artifacts.

## Commands

```text
etv check SPEC [--out DIR] [--json]
etv inspect PROGRAM
etv parse INPUT [--function NAME] [--out FILE] [--no-assembly]
etv rules [--json]
etv validate-rule [RULE_ID]
etv explain REPORT_JSON
```

## Repository map

```text
etv/                    verifier implementation
examples/programs/      Semantic TTIR programs
examples/ttir/          raw TTIR programs parsed by libtriton
examples/specs/         positive, negative, and UNKNOWN contracts
tests/                  unit, property, CLI, and integration tests
docs/architecture.md     components and proof flow
docs/verification_process.md
                         inputs, frontend, semantics, proof, saturation, and trust
docs/implementation_status.md
docs/dependencies.md     pinned dependencies, installation, and environments
```

The implementation is intentionally a verifier, not an optimizer: it does not
extract a cheapest e-graph term. Its success condition is that corresponding
observable roots become members of the same egglog e-class. See the
[complete verification process](docs/verification_process.md) for rule admission,
fact gates, audit logs, and the LLM provenance boundary.
