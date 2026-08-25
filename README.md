# ETV

ETV (Equivalence for TTIR Verification) is a minimum viable translation
validator for fixed, specialized Triton-kernel semantics. It compares two
strictly typed Semantic TTIR programs and returns exactly one of:

- `PROVED`: observable Output memory is equal under all declared assumptions.
- `DISPROVED`: a replayable address, mask, or abstract-value counterexample exists.
- `UNKNOWN`: an assumption, semantic model, rule, or resource obligation is missing.

The current MVP machine-checks the `add` proof described in the project research
archive: a stride-aware 2-D ntops implementation is compared with an Inductor
linear implementation, including the scalar-argument versus 0-D-tensor ABI
difference. It also proves an FMA variant and rejects three deliberate mutations.

## What `PROVED` means

For a fixed `PairSpec`, ETV exhaustively enumerates every launched program/lane,
proves mask, coverage, address injectivity, and output-address correspondence,
then uses a joint e-graph to prove symbolic output values equal for arbitrary
logical input values. Every admitted algebraic rule is replayed through Z3 by
checking that `lhs != rhs` is unsatisfiable over mathematical reals.

The result is conditional on the role, shape, stride, launch, and no-alias facts
in the spec. Floating-point operations use `ABSTRACT_FLOAT`; a proof is not an
IEEE-754, tolerance, or bitwise-GPU result. Raw TTIR-to-Semantic-TTIR lifting is
also outside this MVP and therefore remains part of the trusted input boundary.
Logical blocks are modeled as total maps, so allocated-buffer bounds and GPU
memory safety are also outside the current claim.

See [Semantics and trust](docs/SEMANTICS_AND_TRUST.md) for the precise claim.

## Quick start

```bash
python3 -m venv .venv
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

`check` exit codes are `0` for `PROVED`, `1` for `DISPROVED`, and `2` for
`UNKNOWN` or invalid input. `--out` writes deterministic `report.json` and
`report.md` artifacts.

## Commands

```text
etv check SPEC [--out DIR] [--json]
etv inspect PROGRAM
etv rules [--json]
etv validate-rule [RULE_ID]
etv explain REPORT_JSON
```

## Repository map

```text
etv/                    verifier implementation
examples/programs/      Semantic TTIR programs
examples/specs/         positive, negative, and UNKNOWN contracts
tests/                  unit, property, CLI, and integration tests
docs/ARCHITECTURE.md     components and proof flow
docs/INPUT_FORMAT.md     strict JSON formats and expression language
docs/SEMANTICS_AND_TRUST.md
docs/RESEARCH_NOTES.md   research synthesis and design decisions
docs/IMPLEMENTATION_STATUS.md
docs/DEPENDENCIES.md
```

The implementation is intentionally a verifier, not an optimizer: it does not
extract a cheapest e-graph term. Its success condition is that corresponding
observable roots become members of the same e-class.
