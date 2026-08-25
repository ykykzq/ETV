#!/usr/bin/env python3
"""Reproduce the real ntops/TorchInductor Add pair used by ETV.

The script intentionally requires local checkouts at pinned commits. It does not
clone repositories or execute a CUDA kernel. Both TTIR modules are produced by
Triton's AST frontend and TTIR optimization pipeline for a CUDA sm80 target.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
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
INDUCTOR_XBLOCK = 128


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


def _fake_a100_properties() -> SimpleNamespace:
    return SimpleNamespace(
        name="NVIDIA A100-SXM4-40GB",
        major=8,
        minor=0,
        multi_processor_count=108,
        regs_per_multiprocessor=65536,
        max_threads_per_multi_processor=2048,
        total_memory=40 * 1024**3,
        warp_size=32,
        gcnArchName="",
    )


def _extract_inductor_kernel(wrapper: str, name: str) -> str:
    marker = f"async_compile.triton('{name}', '''\n"
    start = wrapper.find(marker)
    if start < 0:
        raise RuntimeError(f"TorchInductor wrapper omitted {name}")
    start += len(marker)
    end = wrapper.find("\n''', device_str='cuda')", start)
    if end < 0:
        raise RuntimeError(f"cannot locate the end of {name}")
    return "\n".join(line.rstrip() for line in wrapper[start:end].splitlines()) + "\n"


def _jit_compile_adapter(generated: str, name: str) -> str:
    tree = ast.parse(generated)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None or function.end_lineno is None:
        raise RuntimeError(f"generated source omitted function {name}")
    lines = generated.splitlines()
    definition = function.lineno - 1
    jit_line = next(
        (index for index in range(definition - 1, -1, -1) if lines[index].strip() == "@triton.jit"),
        None,
    )
    if jit_line is None:
        raise RuntimeError(f"generated function {name} has no @triton.jit decorator")
    body = "\n".join(lines[jit_line : function.end_lineno])
    return "import triton\nimport triton.language as tl\n\n\n" + body + "\n"


def _generate_inductor_sources(sources: Path) -> tuple[Path, Path, Path]:
    import torch
    import torch.utils._triton as torch_triton
    import triton.compiler.compiler as triton_compiler
    from torch._dynamo.device_interface import CudaInterface
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.virtualized import V
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils._triton import has_triton, has_triton_package

    properties = _fake_a100_properties()
    torch.cuda.get_device_properties = lambda device=None: properties
    torch.cuda.get_device_capability = lambda device=None: (8, 0)
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1

    def reference(input_tensor, alpha, other):
        return (input_tensor + alpha * other,)

    mode = FakeTensorMode()
    with mode:
        input_tensor = torch.empty(SHAPE, device="cuda", dtype=torch.float32)
        alpha = torch.empty((), device="cuda", dtype=torch.float32)
        other = torch.empty(SHAPE, device="cuda", dtype=torch.float32)
        graph = make_fx(reference, tracing_mode="fake")(input_tensor, alpha, other)

    torch.cuda.is_available = lambda: True
    triton_compiler.triton_key = lambda: "triton-3.7.1"
    CudaInterface.is_available = staticmethod(lambda: True)
    CudaInterface.Worker.current_device = staticmethod(lambda: 0)
    CudaInterface.Worker.get_device_properties = staticmethod(lambda device=None: properties)
    CudaInterface.get_device_properties = staticmethod(lambda device=None: properties)
    has_triton_package.cache_clear()
    has_triton.cache_clear()
    if not has_triton():
        raise RuntimeError("TorchInductor did not recognize the pinned Triton installation")

    # This value is report metadata only. Avoid initializing a CUDA driver on
    # the offline host while leaving Inductor scheduling and codegen unchanged.
    torch_triton.triton_hash_with_backend = lambda: "triton-3.7.1-cuda-sm80-offline"

    lowering = GraphLowering(
        graph,
        example_inputs=[input_tensor, alpha, other],
        is_inference=True,
    )
    lowering.freeze_runtime_asserts()
    with DebugContext(), V.set_fake_mode(mode), V.set_graph_handler(lowering):
        lowering.run(input_tensor, alpha, other)
        wrapper, _kernels = lowering.codegen()

    kernel_name = "triton_poi_fused_0"
    generated = _extract_inductor_kernel(wrapper.value, kernel_name)
    raw_path = sources / "torch_inductor_add.generated.py"
    raw_path.write_text(generated, encoding="utf-8")
    adapter_path = sources / "torch_inductor_add.triton.py"
    adapter_path.write_text(_jit_compile_adapter(generated, kernel_name), encoding="utf-8")
    graph_path = sources / "torch_inductor_add.fx.txt"
    graph_path.write_text(str(graph.graph).rstrip() + "\n", encoding="utf-8")
    return raw_path, adapter_path, graph_path


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
        "pair_id": "ntops_add_vs_torch_inductor_add",
        "lhs": "ttir/ntops_add.ttir",
        "rhs": "ttir/torch_inductor_add.ttir",
        "frontends": {
            "lhs": {"kind": "ttir", "function": "ntops_add_kernel", "programs": 1},
            "rhs": {"kind": "ttir", "function": "triton_poi_fused_0", "programs": 1},
        },
        "semantic_mode": "abstract_float",
        "roles": {
            "Alpha": {
                "lhs": endpoint("scalar", "arg2"),
                "rhs": endpoint("scalar_block", "arg1", index=0),
            },
            "Input": {"lhs": endpoint("block", "arg0"), "rhs": endpoint("block", "arg0")},
            "Other": {"lhs": endpoint("block", "arg1"), "rhs": endpoint("block", "arg2")},
            "Output": {"lhs": endpoint("block", "arg3"), "rhs": endpoint("block", "arg3")},
        },
        "facts": {
            "bindings": {"X": NUMEL},
            "side_bindings": {
                "lhs": {
                    "arg4": 8,
                    "arg5": 16,
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
                "rhs": {"arg4": NUMEL},
            },
            "assumptions": [],
            "disjoint": [["Input", "Other", "Output"]],
        },
        "contract": {
            "output_role": "Output",
            "output_numel": {"var": "X"},
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
    ttir = output / "ttir"
    sources.mkdir(parents=True, exist_ok=True)
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
    inductor_raw, inductor_adapter, fx_graph = _generate_inductor_sources(sources)
    ntops_ttir = ttir / "ntops_add.ttir"
    inductor_ttir = ttir / "torch_inductor_add.ttir"
    _compile_ttir(
        ntops_source,
        "ntops_add_kernel",
        _ntops_signature(),
        {"BLOCK": NTOPS_BLOCK},
        ntops_ttir,
        "examples/add/sources/ntops_add.triton.py",
    )
    _compile_ttir(
        inductor_adapter,
        "triton_poi_fused_0",
        {
            "in_ptr0": "*fp32",
            "in_ptr1": "*fp32",
            "in_ptr2": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        {"XBLOCK": INDUCTOR_XBLOCK},
        inductor_ttir,
        "examples/add/sources/torch_inductor_add.triton.py",
    )

    spec_path = output / "pair.json"
    _write_json(spec_path, _pair_spec())
    provenance = {
        "format": "etv-extracted-pair-provenance-v1",
        "pair_id": "ntops_add_vs_torch_inductor_add",
        "operator": "add(input, other, alpha) = input + alpha * other",
        "specialization": {
            "shape": list(SHAPE),
            "dtype": "float32",
            "numel": NUMEL,
            "target": "cuda:sm80",
            "ntops_block": NTOPS_BLOCK,
            "torch_inductor_xblock": INDUCTOR_XBLOCK,
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
            "rhs": "PyTorch tensor expression -> FakeTensor FX -> TorchInductor Triton -> Triton AST -> optimized TTIR",
            "rhs_fx_expression": "input_tensor + alpha_tensor * other",
            "ntops_test_reference": "torch.add(input, other, alpha=python_scalar)",
            "compatibility_shims": [
                "Expose triton_key() so PyTorch 2.8 recognizes the separately pinned Triton 3.7.1 package.",
                "Return an offline backend hash instead of initializing the CUDA runtime driver.",
                "Provide fixed A100/sm80 device properties to TorchInductor scheduling.",
            ],
            "cuda_execution": False,
            "inductor_adapter": (
                "Removed only the pointwise launch decorator and retained the exact @triton.jit "
                "function text for offline AST compilation."
            ),
            "source_normalization": "Removed trailing whitespace from TorchInductor source lines.",
        },
        "artifacts": {
            str(path.relative_to(output)): _sha256(path)
            for path in (
                ntops_source,
                ntops_metadata,
                inductor_raw,
                inductor_adapter,
                fx_graph,
                ntops_ttir,
                inductor_ttir,
                spec_path,
            )
        },
    }
    _write_json(output / "provenance.json", provenance)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
