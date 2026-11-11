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
  --verifier .venv-verifier/bin/etv --execution-scope ttir-only
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

The pytest collector additionally enables `etv-host-operations-v1` events. A
Torch dispatch mode records actual pre/post-processing effects; a function mode
records known Python scalar constructors without inspecting arbitrary tensor
contents. Reference-side modes wrap the compiled backend's runtime callable,
so they do not cause Dynamo to replace the compiled reference with eager code.
The final tensor views are recorded separately for logical-output observation.
Unsupported effects are explicit diagnostics; stochastic states are not assumed
to correspond merely because both sides use a seed.

Pair generation defaults to `--execution-scope ttir-only`. Host events remain in
the capture and provenance files, but are excluded from proof steps and internal
buffer producers. A TTIR load that depends on a host-produced or host-modified
buffer, or a final observation produced by a host post-operation, is diagnosed as
`host dependency outside ttir-only execution scope`. The generator does not add
zero-initialization premises, promote host constants to inputs, replace the final
observed buffer, or reduce full-coverage requirements. Output shape/stride/offset
descriptions remain available to identify the logical observation.

To run the distinct composition experiment, opt in explicitly:

```bash
.venv-benchmark/bin/etv-bench pair build/captured_add/capture.json \
  --verifier .venv-verifier/bin/etv --execution-scope ttir-with-host-ops
```

Both scopes are written to `pair.json` metadata and provenance. Composition uses
the verifier's trusted host-operation semantics; it does not verify PyTorch's
implementation or turn those operations into captured TTIR. In particular, a
`diag` composition can prove equivalent after a captured zero-fill, while the
original TTIR-only pair still fails full-output write coverage. Such verdicts
must be reported under their respective execution scopes.

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

## Frozen representative experiments

`etv_bench.pytest_capture` freezes the full collection with `--etv-freeze` and
selects one representative per test function. `tools/run_representatives.py`
checks the upstream commit and test-source hashes, collects the entire test tree
in each isolated process, compares selected parameters to the lock, then executes
only that representative. Full collection order matters because upstream modules
can import and consume each other's random parametrization generators.

`tools/freeze_representatives.py` copies the TTIR, runtime mappings, original test
outcomes, process failures and logs into a new versioned directory. It refuses to
overwrite an existing version and records SHA-256 for every frozen file.
`tools/analyze_representatives.py` verifies the hashes both before and after the
batch, invokes the public pairing/verification CLI, and retains all failed cases
in the summary denominator. `tools/report_representatives.py` renders the complete
case matrix, output dtype differences and explicitly separate synthetic controls.
Reports and raw experiment corpora are generated locally and are excluded from
the code-only experiment branch.

## Complete-capture formal suite

`tools/filter_captures.py` requires a frozen inventory with every test selected
and attempted. It retains complete two-sided runtime captures and excludes failed
or incomplete capture records. It checks identities, source revisions, selected
launches, file containment and TTIR hashes. Numerical-test outcomes, output dtype
differences, PairSpec validity and formal verdicts do not filter the retained set.
Both retained and excluded identities are recorded; the source captures remain
unchanged. The destination must be fresh.

Run these commands from the repository root, using the verifier environment from
[installation](installation.md). The paths below refer to locally captured data;
the repository does not include the full experiment corpus.

```bash
.venv-verifier/bin/python tools/filter_captures.py build/full-suite.lock.json \
  --captures build/full-capture --out build/filtered-v1
.venv-verifier/bin/python tools/run_formal_suite.py \
  --frozen build/filtered-v1 --out build/formal-v1 \
  --python .venv-verifier/bin/python --workers 16
```

The runner copies the frozen corpus into a fresh derived workspace and writes
`run.json`, `progress.json`, `completed.json`, per-case reports and the complete
`analysis/summary.json`. It checks implementation hashes and the complete selected
identity set; the analyzer checks frozen artifact hashes before and after the run.
`NOT_VERIFIED` denotes pipeline/contract failures without a formal verdict and
must remain separate from verifier `UNKNOWN` in the detailed counts.

The 2026-11-04 run retained 1,365 of 2,102 cases (3,350 TTIR files): 223 `PROVED`,
462 `UNKNOWN`, 664 `NOT_VERIFIED` and 16 `DISPROVED`. All 223 proofs were formal
under declared predicates, with no unverified trusted axioms. Branches and loops
remain unsupported. The 16 replayed mismatches concern output-write masks in
captured `diag_1d` fragments: host-side zero initialization is outside those TTIR
fragments. They do not establish 16 incorrect full operators. Conclusions use
the finite-real abstraction, not IEEE floating-point numerical equivalence.

## Host-operation recapture iteration

`tools/prepare_host_recapture.py prepare` selects the 77 missing-input and 24
missing-output cases from the audited baseline, validates the frozen corpus,
and writes a targeted selection lock. `--include-diag` also selects the 16
initialization-omission cases. Run that lock with `run_representatives.py` in
the pinned benchmark environment, then use `prepare_host_recapture.py merge`
to build a new immutable corpus. The original 1,365 identities remain unchanged.
Every attempted recapture and old/new capture hash is audited. Failed recaptures
retain an explicitly marked original record; they never count as enriched.

Run `run_formal_suite.py` on the new corpus with an explicit execution scope;
the default is the strict `ttir-only` experiment. A composition run requires
`--execution-scope ttir-with-host-ops` and a separate output directory.
`export_host_iteration.py` requires
a completed actual run and its recapture audit; it exports per-case transitions,
host-step provenance and fresh proof auditing for local reports. This workflow
does not infer new verification results from unit tests or from source inspection.
