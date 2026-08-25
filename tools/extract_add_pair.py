#!/usr/bin/env python3
"""Reproduce the real ninetoothed-TTIR/Torch-Prims Add pair used by ETV.

The script intentionally requires local checkouts at pinned commits. It does not
clone repositories or execute a CUDA kernel. Only the ninetoothed side is lowered
through Triton; the Torch side is captured as an explicit Prims graph.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from importlib.metadata import version
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

NTOPS_COMMIT = "9ae4166ad342e4745f0eed13a5a20d069e994fc0"
NINETOOTHED_COMMIT = "efe519d1b12a820e7aa605d775af3d52c8b0d605"
NTOPS_ADD_SHA256 = "a8f938399835b59385fb3142c7f96382e7855cc6afc7025ccfaa495dd2422d50"
NTOPS_ELEMENTWISE_SHA256 = "a3f2d7a1bac443bf256e591c8910b125ebc994b97da7c5b8b60d66ac31284d72"
EXPECTED_VERSIONS = {
    "ninetoothed": "0.26.0",
    "torch": "2.8.0",
    "triton": "3.7.1",
}
SHAPE = (8, 16)
NUMEL = 128
NTOPS_BLOCK = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_checkout(repo: Path, expected: str, name: str) -> None:
    if _git_head(repo) != expected:
        raise RuntimeError(f"{name} must be checked out at {expected}")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"{name} checkout has modified tracked files")


def _require_version(package: str, expected: str) -> None:
    installed = version(package)
    if installed.split("+")[0] != expected:
        raise RuntimeError(f"{package} must be {expected}, found {installed}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generate_ntops_source(ntops_repo: Path, ninetoothed_repo: Path, sources: Path) -> Path:
    import ninetoothed
    from ninetoothed.compiler.driver import lower

    installed_root = Path(ninetoothed.__file__).resolve().parents[2]
    if installed_root != ninetoothed_repo.resolve():
        raise RuntimeError(
            "ninetoothed must be installed editable from the pinned checkout: "
            f"expected {ninetoothed_repo.resolve()}, found {installed_root}"
        )

    source_root = ntops_repo / "src"
    ntops_package = types.ModuleType("ntops")
    ntops_package.__path__ = [str(source_root / "ntops")]
    sys.modules["ntops"] = ntops_package
    kernels_package = types.ModuleType("ntops.kernels")
    kernels_package.__path__ = [str(source_root / "ntops" / "kernels")]
    sys.modules["ntops.kernels"] = kernels_package
    _load_module(
        "ntops.kernels.element_wise",
        source_root / "ntops" / "kernels" / "element_wise.py",
    )
    add = _load_module(
        "ntops.kernels.add",
        source_root / "ntops" / "kernels" / "add.py",
    )
    artifact = lower(
        *add.premake(2, dtype=ninetoothed.float32, block_size=128),
        backend="triton",
        caller="torch",
        kernel_name="ntops_add",
    )
    written = artifact.write_to(sources)
    path = Path(written[0])
    if path.name != "ntops_add.triton.py":
        raise RuntimeError(f"unexpected ninetoothed artifact name: {path}")
    return path


def _prims_program() -> dict[str, Any]:
    dimension = lambda name: {"var": name}
    shape = [dimension("torch_dim0"), dimension("torch_dim1")]
    return {
        "format": "etv-prims-program-v1",
        "name": "torch_add_prims",
        "inputs": [
            {"name": "input", "dtype": "float32", "shape": shape},
            {"name": "alpha", "dtype": "float32", "shape": []},
            {"name": "other", "dtype": "float32", "shape": shape},
        ],
        "nodes": [
            {
                "name": "broadcast_alpha",
                "op": "prims.broadcast_in_dim",
                "args": ["alpha"],
                "shape": shape,
                "broadcast_dimensions": [],
            },
            {
                "name": "scaled_other",
                "op": "prims.mul",
                "args": ["broadcast_alpha", "other"],
            },
            {"name": "result", "op": "prims.add", "args": ["input", "scaled_other"]},
        ],
        "output": {"value": "result", "block": "output"},
    }


def _generate_prims_sources(sources: Path, prims: Path) -> tuple[Path, Path]:
    import torch
    import torch._dynamo  # Initialize make_fx support before entering TorchRefsMode.
    from torch._prims.context import TorchRefsMode
    from torch.fx.experimental.proxy_tensor import make_fx

    def reference(input_tensor, alpha, other):
        return input_tensor + alpha * other

    input_tensor = torch.empty(SHAPE, device="meta", dtype=torch.float32)
    alpha = torch.empty((), device="meta", dtype=torch.float32)
    other = torch.empty(SHAPE, device="meta", dtype=torch.float32)
    with TorchRefsMode():
        graph = make_fx(reference)(input_tensor, alpha, other)

    operations = [
        str(node.target)
        for node in graph.graph.nodes
        if node.op == "call_function"
    ]
    expected = [
        "prims.broadcast_in_dim.default",
        "prims.mul.default",
        "prims.add.default",
    ]
    if operations != expected:
        raise RuntimeError(f"unexpected Torch Prims graph: {operations}")

    graph_path = sources / "torch_prims_add.fx.txt"
    graph_path.write_text(str(graph.graph).rstrip() + "\n", encoding="utf-8")
    prims_path = prims / "torch_add.prims.json"
    _write_json(prims_path, _prims_program())
    return prims_path, graph_path


def _compile_ttir(
    source_path: Path,
    function_name: str,
    signature: dict[str, str],
    constexprs: dict[str, Any],
    output_path: Path,
    canonical_location: str,
) -> None:
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    module = _load_module(f"etv_extract_{function_name}", source_path)
    function = getattr(module, function_name)
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    compiled = ASTSource(function, signature, constexprs=constexprs).make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiled = backend.make_ttir(compiled, {}, options, 80)
    text = str(compiled).replace(str(source_path.resolve()), canonical_location)
    output_path.write_text(text, encoding="utf-8")


def _ntops_signature() -> dict[str, str]:
    signature = {
        "input": "*fp32",
        "other": "*fp32",
        "alpha": "fp64",
        "output": "*fp32",
    }
    for tensor in (0, 1, 3):
        prefix = f"ninetoothed_ninetoothed_tensor_{tensor}"
        signature[f"{prefix}_size_0"] = "i32"
        signature[f"{prefix}_size_1"] = "i32"
        signature[f"{prefix}_stride_0"] = "i32"
        signature[f"{prefix}_stride_1"] = "i32"
    signature["BLOCK"] = "constexpr"
    return signature


def _pair_spec() -> dict[str, Any]:
    endpoint = lambda kind, name, **extra: {"kind": kind, "name": name, **extra}
    return {
        "format": "etv-pair-v1",
        "pair_id": "ntops_add_vs_torch_prims_add",
        "lhs": "ttir/ntops_add.ttir",
        "rhs": "prims/torch_add.prims.json",
        "frontends": {
            "lhs": {
                "kind": "ttir",
                "function": "ntops_add_kernel",
                "programs": {"op": "ceildiv", "args": [{"var": "c"}, NTOPS_BLOCK]},
            },
            "rhs": {"kind": "prims"},
        },
        "semantic_mode": "abstract_float",
        "roles": {
            "Alpha": {
                "lhs": endpoint("scalar", "arg2"),
                "rhs": endpoint("scalar_block", "alpha", index=0),
            },
            "Input": {"lhs": endpoint("block", "arg0"), "rhs": endpoint("block", "input")},
            "Other": {"lhs": endpoint("block", "arg1"), "rhs": endpoint("block", "other")},
            "Output": {"lhs": endpoint("block", "arg3"), "rhs": endpoint("block", "output")},
        },
        "facts": {
            "bindings": {},
            "side_bindings": {
                "lhs": {
                    "arg4": {"var": "a"},
                    "arg5": {"var": "b"},
                    "arg6": 16,
                    "arg7": 1,
                    "arg8": 8,
                    "arg9": 16,
                    "arg10": 16,
                    "arg11": 1,
                    "arg12": 8,
                    "arg13": 16,
                    "arg14": 16,
                    "arg15": 1,
                },
                "rhs": {
                    "torch_dim0": {"var": "a"},
                    "torch_dim1": {"var": "b"},
                },
            },
            "parameters": {
                "a": {"min": SHAPE[0], "max": SHAPE[0]},
                "b": {"min": SHAPE[1], "max": SHAPE[1]},
                "c": {"min": NUMEL, "max": NUMEL},
            },
            "constraints": [
                {
                    "op": "eq",
                    "args": [
                        {"op": "imul", "args": [{"var": "a"}, {"var": "b"}]},
                        {"var": "c"},
                    ],
                }
            ],
            "assumptions": [],
            "disjoint": [["Input", "Other", "Output"]],
        },
        "contract": {
            "output_role": "Output",
            "output_numel": {"var": "c"},
            "require_full_coverage": True,
            "require_disjoint": ["Input", "Other", "Output"],
        },
        "limits": {"max_iterations": 8, "max_enodes": 20000, "timeout_ms": 10000},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntops-repo", type=Path, required=True)
    parser.add_argument("--ninetoothed-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("examples/add"))
    args = parser.parse_args()

    ntops_repo = args.ntops_repo.resolve()
    ninetoothed_repo = args.ninetoothed_repo.resolve()
    output = args.output
    sources = output / "sources"
    prims = output / "prims"
    ttir = output / "ttir"
    sources.mkdir(parents=True, exist_ok=True)
    prims.mkdir(parents=True, exist_ok=True)
    ttir.mkdir(parents=True, exist_ok=True)

    _require_checkout(ntops_repo, NTOPS_COMMIT, "ntops")
    _require_checkout(ninetoothed_repo, NINETOOTHED_COMMIT, "ninetoothed")
    for package, expected in EXPECTED_VERSIONS.items():
        _require_version(package, expected)
    add_source = ntops_repo / "src/ntops/kernels/add.py"
    elementwise_source = ntops_repo / "src/ntops/kernels/element_wise.py"
    if _sha256(add_source) != NTOPS_ADD_SHA256:
        raise RuntimeError("pinned ntops add.py hash mismatch")
    if _sha256(elementwise_source) != NTOPS_ELEMENTWISE_SHA256:
        raise RuntimeError("pinned ntops element_wise.py hash mismatch")

    ntops_source = _generate_ntops_source(ntops_repo, ninetoothed_repo, sources)
    ntops_metadata = sources / "ntops_add.triton.json"
    if not ntops_metadata.is_file():
        raise RuntimeError("ninetoothed did not emit ntops_add.triton.json")
    prims_program, fx_graph = _generate_prims_sources(sources, prims)
    ntops_ttir = ttir / "ntops_add.ttir"
    _compile_ttir(
        ntops_source,
        "ntops_add_kernel",
        _ntops_signature(),
        {"BLOCK": NTOPS_BLOCK},
        ntops_ttir,
        "examples/add/sources/ntops_add.triton.py",
    )
    spec_path = output / "pair.json"
    _write_json(spec_path, _pair_spec())
    provenance = {
        "format": "etv-extracted-pair-provenance-v1",
        "pair_id": "ntops_add_vs_torch_prims_add",
        "operator": "add(input, other, alpha) = input + alpha * other",
        "specialization": {
            "shape": list(SHAPE),
            "dtype": "float32",
            "numel": NUMEL,
            "target": "cuda:sm80",
            "ntops_block": NTOPS_BLOCK,
        },
        "upstreams": {
            "ntops": {
                "url": "https://github.com/InfiniTensor/ntops",
                "commit": NTOPS_COMMIT,
                "files": {
                    "src/ntops/kernels/add.py": NTOPS_ADD_SHA256,
                    "src/ntops/kernels/element_wise.py": NTOPS_ELEMENTWISE_SHA256,
                },
            },
            "ninetoothed": {
                "url": "https://github.com/InfiniTensor/ninetoothed",
                "commit": NINETOOTHED_COMMIT,
            },
        },
        "toolchain": {name: version(name) for name in EXPECTED_VERSIONS},
        "generation": {
            "lhs": "ntops Add -> ninetoothed SSA lowering -> Triton AST -> optimized TTIR",
            "rhs": (
                "PyTorch tensor expression -> TorchRefsMode/make_fx -> explicit "
                "Torch Prims graph"
            ),
            "rhs_fx_expression": "input_tensor + alpha_tensor * other",
            "ntops_test_reference": "torch.add(input, other, alpha=python_scalar)",
            "compatibility_shims": [],
            "cuda_execution": False,
            "prims_export": (
                "The fixed-rank FX trace is checked for broadcast_in_dim, mul, add "
                "and serialized with symbolic dimension slots."
            ),
        },
        "artifacts": {
            str(path.relative_to(output)): _sha256(path)
            for path in (
                ntops_source,
                ntops_metadata,
                prims_program,
                fx_graph,
                ntops_ttir,
                spec_path,
            )
        },
    }
    _write_json(output / "provenance.json", provenance)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
