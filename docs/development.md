# Development

Core checks require no CUDA or Triton:

```bash
uv sync --locked --extra dev
uv run --no-sync pytest tests/unit
uv run --no-sync ruff check src tests tools examples/generators
uv run --no-sync ruff format --check src tests tools examples/generators
uv run --no-sync mypy
uv run --no-sync python -m build
```

For full tests, install `ttir`, build the native adapter, then run
`uv run --no-sync pytest` and `uv run --no-sync python tools/smoke.py`. Generic wheels
do not contain a portable native extension; rebuild it against the test environment.

## Extending Semantics

Add arity/sort validation, exact evaluation where supported, solver encoding and
definedness conditions together. Lift tests use verified raw TTIR; property tests
compare finite integers/casts with Z3. Unsupported encodings remain explicit.
Uninterpreted math symbols must not enable approximate counterexamples.

Memory regressions cover complete enumeration, ranges, coverage, masks, addresses,
duplicate writers, undefined loads, no-alias premises and step visibility. Rule
tests cover rejection, gates, hashes, per-run isolation, matches and trust levels.
Partition/LLM failures must preserve whole-goal fallback.

## Acceptance and CI

The small corpus includes real fixed/parameterized Add, deliberate compute/mask/
address mismatches, a supported single-side multi-launch pair, an unsupported loop
and four real multi-launch sequences. Proof tests inspect premises, matches,
soundness and replay evidence in addition to verdicts. Benchmark tests cover slot
alignment, compiler metadata, hashes, review immutability and subprocess reports.

`Core` CI has contract/build and Linux TTIR jobs. The latter installs Triton 3.7.1
and matching LLVM headers. `Optional CUDA Capture` is manual and requires a
self-hosted CUDA runner; it cannot block the no-CUDA core job.

Schema and reason codes are public API. New reasons can be added; changing meanings
or field names needs a schema/version decision. For fixed inputs/dependencies with
LLM disabled, facts should be deterministic apart from run IDs, durations and
environment paths.
