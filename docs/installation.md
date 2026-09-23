# Installation

Use Python 3.12, `uv`, and a C++17 compiler. Core dependencies are egglog 13.2.0
and z3-solver 4.16.0.0. TTIR verification also needs official Triton 3.7.1 and the
local snapshot adapter. Core checks require only `uv sync --locked --extra dev`
and `uv run --no-sync pytest tests/unit`.

## Linux

```bash
uv sync --locked --extra dev --extra ttir
uv run --no-sync python tools/fetch_llvm_headers.py
uv run --no-sync python tools/build_adapter.py --llvm-include build/llvm/llvm-1f126a6d-ubuntu-x64-1/include
uv run --no-sync python tools/smoke.py
```

The header tool uses Triton's official LLVM storage, pinned to revision
`1f126a6dea50d185c0781743a667390037ae88bd`, build 1. It extracts headers with
path-safe tar extraction. Reading the archive can require several hundred MB of
traffic; output is cached in `build/llvm`. ARM64 uses `ubuntu-arm64` in the path.

The extension links to the installed `libtriton` and shares its registered MLIR
types. Rebuild it after changing environments. A generic Python wheel contains
Python packages, not a portable native adapter. Use an editable source checkout
for TTIR verification and build the adapter there.

## macOS

PyPI's Triton wheel is Linux-only. Install official `v3.7.1` from source in a Python
3.12 environment with Triton's CMake/Ninja prerequisites and matching LLVM headers.
Disable Proton when the platform does not support it. Then run:

```bash
python tools/build_adapter.py --llvm-include /path/to/llvm-1f126a6d-macos-arm64-1/include
python -m pytest
```

The adapter has been exercised locally against an official 3.7.1 source build on
Apple Silicon. This enables IR inspection, not CUDA execution on macOS. Do not
substitute another LLVM revision merely because its headers compile.

## Benchmark Environment

```bash
UV_PROJECT_ENVIRONMENT=.venv-verifier uv sync --locked --extra dev --extra ttir
UV_PROJECT_ENVIRONMENT=.venv-benchmark uv sync --locked --extra benchmark
```

Build the adapter with `.venv-verifier/bin/python`. Benchmark dependencies pin
PyTorch 2.8.0, ninetoothed 0.26.0 and NumPy 2.2.6. PyTorch's selected Triton can
differ; uv explicitly forbids combining `ttir` and `benchmark` extras. The two
environments exchange files through public CLI commands.

Missing/mismatched frontend components return `UNKNOWN(TTIR_VERSION_MISMATCH)`.
Malformed IR produces parser/verifier reasons; valid unsupported IR produces
operation/region/type reasons. Linker failures are installation errors, not a
reason to bypass the module verifier.
