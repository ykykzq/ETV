# Benchmark Artifacts

The benchmark package uses public verifier CLI/report contracts, not verifier
internals. Heavy dependencies are isolated in the `benchmark` extra.

## Contract

A case contains `manifest.json` (`etv-benchmark-case-v1`), raw TTIR, strict
`pair.json`, capture evidence and review entries in `provenance.json`, and
`expected.json`. The latter is a regression oracle, never a verifier input.
Generated cases initially have an unspecified oracle until it is assigned.

Manifests record ID, generation mode/version, source repository/commit/test ID,
environment versions, per-artifact SHA-256, side, physical launch ordinal/step,
and relative paths. Status is `captured`, `needs_review`, `ready`, `unsupported`
or `failed`; `ready` says nothing about equivalence.

## Capture and Pair

Use the separate environments in [installation](installation.md). The supplied
CUDA driver compiles and warms both sides before capturing:

```bash
.venv-benchmark/bin/etv-bench collect examples.generators.add:collect \
  --out build/captured_add --repository https://github.com/ykykzq/ETV \
  --commit YOUR_CHECKED_OUT_COMMIT --test-id generated_add
.venv-benchmark/bin/etv-bench pair build/captured_add/capture.json \
  --verifier .venv-verifier/bin/etv
.venv-benchmark/bin/etv-bench run build/captured_add/manifest.json \
  --out build/captured_add/result --verifier .venv-verifier/bin/etv
```

An entry is a Python `module:function` accepting `Capture`. Declare corresponding
inputs with `capture.inputs`, bracket actual executions with `capture.side`, and
identify final tensors with `capture.outputs(side, {"Output": tensor})`. Tensor
contents are not serialized.

Python profiling observes selected Triton JIT runs and selected TorchInductor
Python launchers, excluding tuning candidates. Records contain actual TTIR, grid,
function, compiler signature, constexpr metadata and runtime slots. Buffer evidence
retains device/storage identity, allocation base, element offset, dtype, shape and
stride. Storage references stay alive to prevent allocation reuse from merging
unrelated buffers. Integers equal to 1 remain slots unless the compiler signature
classifies them as constexpr.

Pairing invokes `etv parse`, traces structured pointer owners, checks TTIR arity
against compiler slots, assigns output/internal roles, and creates one launch with
all its stores. External inputs need explicit cross-side role correspondence.
Allocation identities supply disjointness premises, not proof. Ambiguous mappings,
unsupported scalars, missing roles or slot mismatches become `needs_review`.
Complete candidates run through the strict verifier before being marked ready.

CPU tests exercise these contracts using real TTIR and recorded frame shapes;
they do not establish successful CUDA capture. Static C++ launchers, CUDA graphs,
nested signatures, multiple final outputs and stream/event ordering require reviewed
artifacts. Optional CUDA CI needs a suitable self-hosted runner and is independent
of core verification.

## Review and Run

After correcting a PairSpec, append review evidence:

```bash
etv-bench review build/captured_add --note 'Checked runtime slots and storage views' \
  --verifier .venv-verifier/bin/etv
etv-bench summarize build/captured_add/result/report.json
```

Review preserves capture records, stores the pair hash/validation result and updates
readiness. It never upgrades proof strength. `run` checks containment, artifact
hashes and the recorded pair hash, launches a subprocess, reads a fresh report and
rejects stale reports or inconsistent exit codes. `summarize` retains exact statuses
and reasons, including UNKNOWN.

Manual artifacts share this contract. `tools/build_examples.py` regenerates small
synthetic fixtures. `etv_bench.migrate.migrate_v2` combines explicitly mapped store
components belonging to one physical launch, preserves TTIR hashes and records
before/after evidence. Multi-launch conversion requires explicit store-role evidence.
`tools/import_real_cases.py` converts the four retained real fixtures using their
captured pointer inventories. Fixed upstream sources are in
`tools/upstream_revisions.json`; large corpora are not required by the core.
