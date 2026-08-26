#!/usr/bin/env python3
"""Probe the Torch reference side of every pinned ntops operator test.

The upstream tests require a CUDA device for their numerical comparison.  This
tool instead traces one representative specialization per test module with
FakeTensor, lowers it through TorchInductor for an offline CUDA sm80 target,
and compiles every emitted Triton kernel to optimized TTIR.  It never executes
a CUDA kernel, so a successful result is a frontend/code-generation result,
not a numerical correctness result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from extract_add_pair import (
    NTOPS_COMMIT,
    _compile_ttir,
    _extract_inductor_kernel,
    _fake_a100_properties,
    _jit_compile_adapter,
    _require_checkout,
)

sys.dont_write_bytecode = True


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class Case:
    reference: str
    function: Callable[..., Any]
    inputs: tuple[TensorSpec, ...]


class ProbeFailure(RuntimeError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.original = error


def _cases(torch: Any) -> dict[str, Case]:
    functional = torch.nn.functional
    unary = TensorSpec((8, 16))
    binary = (TensorSpec((8, 16)), TensorSpec((8, 16)))
    integer_binary = (TensorSpec((8, 16), "int32"), TensorSpec((8, 16), "int32"))

    def rotary(input_tensor: Any, cos: Any, sin: Any) -> Any:
        first, second = input_tensor.chunk(2, dim=-1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )

    cases = {
        "abs": Case("torch.abs", torch.abs, (unary,)),
        "acosh": Case("torch.acosh", torch.acosh, (unary,)),
        "adaptive_avg_pool2d": Case(
            "torch.nn.functional.adaptive_avg_pool2d",
            lambda value: functional.adaptive_avg_pool2d(value, (2, 2)),
            (TensorSpec((2, 3, 8, 8)),),
        ),
        "adaptive_max_pool2d": Case(
            "torch.nn.functional.adaptive_max_pool2d",
            lambda value: functional.adaptive_max_pool2d(value, (2, 2)),
            (TensorSpec((2, 3, 8, 8)),),
        ),
        "add": Case(
            "torch.add", lambda left, right: torch.add(left, right, alpha=0.5), binary
        ),
        "addmm": Case(
            "torch.addmm",
            lambda bias, left, right: torch.addmm(
                bias, left, right, beta=0.5, alpha=1.5
            ),
            (TensorSpec((8, 4)), TensorSpec((8, 16)), TensorSpec((16, 4))),
        ),
        "addmv": Case(
            "torch.addmv",
            lambda bias, matrix, vector: torch.addmv(
                bias, matrix, vector, beta=0.5, alpha=1.5
            ),
            (TensorSpec((8,)), TensorSpec((8, 16)), TensorSpec((16,))),
        ),
        "alpha_dropout": Case(
            "torch.nn.functional.alpha_dropout",
            lambda value: functional.alpha_dropout(value, p=0.2, training=True),
            (unary,),
        ),
        "argsort": Case(
            "torch.argsort", lambda value: torch.argsort(value, dim=-1), (unary,)
        ),
        "atan": Case("torch.atan", torch.atan, (unary,)),
        "avg_pool2d": Case(
            "torch.nn.functional.avg_pool2d",
            lambda value: functional.avg_pool2d(
                value, kernel_size=3, stride=2, padding=1
            ),
            (TensorSpec((2, 3, 8, 8)),),
        ),
        "batch_norm": Case(
            "torch.nn.functional.batch_norm",
            lambda value, mean, variance, weight, bias: functional.batch_norm(
                value, mean, variance, weight, bias, training=False
            ),
            (
                TensorSpec((2, 4, 8, 8)),
                TensorSpec((4,)),
                TensorSpec((4,)),
                TensorSpec((4,)),
                TensorSpec((4,)),
            ),
        ),
        "bincount": Case(
            "torch.bincount",
            lambda value: torch.bincount(value, minlength=16),
            (TensorSpec((32,), "int64"),),
        ),
        "bitwise_and": Case("torch.bitwise_and", torch.bitwise_and, integer_binary),
        "bitwise_not": Case(
            "torch.bitwise_not", torch.bitwise_not, (integer_binary[0],)
        ),
        "bitwise_or": Case("torch.bitwise_or", torch.bitwise_or, integer_binary),
        "bmm": Case(
            "torch.bmm", torch.bmm, (TensorSpec((2, 8, 16)), TensorSpec((2, 16, 4)))
        ),
        "celu": Case(
            "torch.nn.functional.celu",
            lambda value: functional.celu(value, alpha=1.2),
            (unary,),
        ),
        "clamp": Case(
            "torch.clamp", lambda value: torch.clamp(value, -0.5, 0.5), (unary,)
        ),
        "conv2d": Case(
            "torch.nn.functional.conv2d",
            lambda value, weight: functional.conv2d(value, weight, padding=1),
            (TensorSpec((1, 3, 8, 8)), TensorSpec((4, 3, 3, 3))),
        ),
        "cos": Case("torch.cos", torch.cos, (unary,)),
        "cosh": Case("torch.cosh", torch.cosh, (unary,)),
        "diag": Case(
            "torch.diag",
            lambda value: torch.diag(value, diagonal=1),
            (TensorSpec((16,)),),
        ),
        "div": Case("torch.div", torch.div, binary),
        "dropout": Case(
            "torch.nn.functional.dropout",
            lambda value: functional.dropout(value, p=0.2, training=True),
            (unary,),
        ),
        "eq": Case("torch.eq", torch.eq, binary),
        "exp": Case("torch.exp", torch.exp, (unary,)),
        "fmax": Case("torch.maximum", torch.maximum, binary),
        "ge": Case("torch.ge", torch.ge, binary),
        "gelu": Case("torch.nn.functional.gelu", functional.gelu, (unary,)),
        "gt": Case("torch.gt", torch.gt, binary),
        "instance_norm": Case(
            "torch.nn.functional.instance_norm",
            lambda value, weight, bias: functional.instance_norm(
                value, weight=weight, bias=bias
            ),
            (TensorSpec((2, 4, 8, 8)), TensorSpec((4,)), TensorSpec((4,))),
        ),
        "isinf": Case("torch.isinf", torch.isinf, (unary,)),
        "isnan": Case("torch.isnan", torch.isnan, (unary,)),
        "layer_norm": Case(
            "torch.nn.functional.layer_norm",
            lambda value, weight, bias: functional.layer_norm(
                value, (16,), weight, bias
            ),
            (unary, TensorSpec((16,)), TensorSpec((16,))),
        ),
        "le": Case("torch.le", torch.le, binary),
        "logsumexp": Case(
            "torch.logsumexp", lambda value: torch.logsumexp(value, dim=-1), (unary,)
        ),
        "lp_pool1d": Case(
            "torch.nn.functional.lp_pool1d",
            lambda value: functional.lp_pool1d(value, 2.0, 3, stride=2),
            (TensorSpec((2, 3, 16)),),
        ),
        "lp_pool2d": Case(
            "torch.nn.functional.lp_pool2d",
            lambda value: functional.lp_pool2d(value, 2.0, 3, stride=2),
            (TensorSpec((2, 3, 8, 8)),),
        ),
        "lp_pool3d": Case(
            "torch.nn.functional.lp_pool3d",
            lambda value: functional.lp_pool3d(value, 2.0, 3, stride=2),
            (TensorSpec((1, 2, 6, 6, 6)),),
        ),
        "lt": Case("torch.lt", torch.lt, binary),
        "matmul": Case(
            "torch.matmul", torch.matmul, (TensorSpec((8, 16)), TensorSpec((16, 4)))
        ),
        "max": Case("torch.max", lambda value: torch.max(value, dim=-1), (unary,)),
        "max_pool1d": Case(
            "torch.nn.functional.max_pool1d",
            lambda value: functional.max_pool1d(value, 3, stride=2, padding=1),
            (TensorSpec((2, 3, 16)),),
        ),
        "max_pool2d": Case(
            "torch.nn.functional.max_pool2d",
            lambda value: functional.max_pool2d(value, 3, stride=2, padding=1),
            (TensorSpec((2, 3, 8, 8)),),
        ),
        "max_pool3d": Case(
            "torch.nn.functional.max_pool3d",
            lambda value: functional.max_pool3d(value, 3, stride=2, padding=1),
            (TensorSpec((1, 2, 6, 6, 6)),),
        ),
        "maximum": Case("torch.maximum", torch.maximum, binary),
        "mean": Case("torch.mean", lambda value: torch.mean(value, dim=-1), (unary,)),
        "median": Case(
            "torch.median", lambda value: torch.median(value, dim=-1), (unary,)
        ),
        "mm": Case("torch.mm", torch.mm, (TensorSpec((8, 16)), TensorSpec((16, 4)))),
        "msort": Case("torch.msort", torch.msort, (unary,)),
        "mul": Case("torch.mul", torch.mul, binary),
        "ne": Case("torch.ne", torch.ne, binary),
        "neg": Case("torch.neg", torch.neg, (unary,)),
        "pow": Case("torch.pow", lambda value: torch.pow(value, 2.5), (unary,)),
        "quantile": Case(
            "torch.quantile", lambda value: torch.quantile(value, 0.5, dim=-1), (unary,)
        ),
        "relu": Case(
            "torch.nn.functional.relu",
            lambda value: functional.relu(value, inplace=False),
            (unary,),
        ),
        "rms_norm": Case(
            "torch.nn.functional.rms_norm",
            lambda value, weight: functional.rms_norm(value, (16,), weight),
            (unary, TensorSpec((16,))),
        ),
        "rot90": Case(
            "torch.rot90", lambda value: torch.rot90(value, 1, (0, 1)), (unary,)
        ),
        "rotary_position_embedding": Case(
            "test helper arithmetic",
            rotary,
            (
                TensorSpec((2, 4, 8, 16)),
                TensorSpec((1, 1, 8, 8)),
                TensorSpec((1, 1, 8, 8)),
            ),
        ),
        "round": Case("torch.round", torch.round, (unary,)),
        "rsqrt": Case("torch.rsqrt", torch.rsqrt, (unary,)),
        "scaled_dot_product_attention": Case(
            "torch.nn.functional.scaled_dot_product_attention",
            lambda query, key, value: functional.scaled_dot_product_attention(
                query, key, value
            ),
            (
                TensorSpec((1, 2, 8, 16)),
                TensorSpec((1, 2, 8, 16)),
                TensorSpec((1, 2, 8, 16)),
            ),
        ),
        "select_copy": Case(
            "torch.select_copy",
            lambda value: torch.select_copy(value, dim=1, index=2),
            (unary,),
        ),
        "sgn": Case("torch.sgn", torch.sgn, (unary,)),
        "sigmoid": Case("torch.sigmoid", torch.sigmoid, (unary,)),
        "sign": Case("torch.sign", torch.sign, (unary,)),
        "signbit": Case("torch.signbit", torch.signbit, (unary,)),
        "silu": Case(
            "torch.nn.functional.silu",
            lambda value: functional.silu(value, inplace=False),
            (unary,),
        ),
        "sin": Case("torch.sin", torch.sin, (unary,)),
        "softmax": Case(
            "torch.nn.functional.softmax",
            lambda value: functional.softmax(value, dim=-1),
            (unary,),
        ),
        "sort": Case("torch.sort", lambda value: torch.sort(value, dim=-1), (unary,)),
        "stack": Case(
            "torch.stack", lambda left, right: torch.stack((left, right), dim=0), binary
        ),
        "sub": Case(
            "torch.sub", lambda left, right: torch.sub(left, right, alpha=0.5), binary
        ),
        "tanh": Case("torch.tanh", torch.tanh, (unary,)),
        "threshold": Case(
            "torch.nn.functional.threshold",
            lambda value: functional.threshold(value, threshold=0.1, value=-0.2),
            (unary,),
        ),
    }
    return cases


def _configure_inductor(torch: Any) -> Any:
    import torch.utils._triton as torch_triton
    import triton.compiler.compiler as triton_compiler
    from torch._dynamo.device_interface import CudaInterface
    from torch.utils._triton import has_triton, has_triton_package

    properties = _fake_a100_properties()
    torch.cuda.get_device_properties = lambda device=None: properties
    torch.cuda.get_device_capability = lambda device=None: (8, 0)
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.is_available = lambda: True
    triton_compiler.triton_key = lambda: "triton-3.7.1"
    CudaInterface.is_available = staticmethod(lambda: True)
    CudaInterface.Worker.current_device = staticmethod(lambda: 0)
    CudaInterface.Worker.get_device_properties = staticmethod(
        lambda device=None: properties
    )
    CudaInterface.get_device_properties = staticmethod(lambda device=None: properties)
    has_triton_package.cache_clear()
    has_triton.cache_clear()
    if not has_triton():
        raise RuntimeError("TorchInductor did not recognize Triton")
    torch_triton.triton_hash_with_backend = lambda: "triton-3.7.1-cuda-sm80-offline"
    torch.cuda.is_available = lambda: False
    return properties


def _kernel_names(wrapper: str) -> list[str]:
    return re.findall(r"async_compile\.triton\('([^']+)', '''\n", wrapper)


def _wrapper_calls(wrapper: str) -> list[str]:
    patterns = (
        r"extern_kernels\.([A-Za-z0-9_]+)\(",
        r"torch\.ops\.([A-Za-z0-9_.]+)\(",
    )
    return sorted(
        {match for pattern in patterns for match in re.findall(pattern, wrapper)}
    )


def _signature(generated: str, function_name: str) -> dict[str, str]:
    tree = ast.parse(generated)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    argument_names = [argument.arg for argument in function.args.args]
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        metadata = next(
            (kw.value for kw in decorator.keywords if kw.arg == "triton_meta"), None
        )
        if not isinstance(metadata, ast.Dict):
            continue
        for key, value in zip(metadata.keys, metadata.values):
            if ast.literal_eval(key) != "signature":
                continue
            raw = ast.literal_eval(value)
            return {
                argument_names[key] if isinstance(key, int) else str(key): str(kind)
                for key, kind in raw.items()
            }
    raise RuntimeError(f"{function_name} has no TorchInductor signature metadata")


def _adapter(generated: str, function_name: str) -> str:
    body = _jit_compile_adapter(generated, function_name)
    imports = (
        "from torch._inductor.runtime import triton_helpers\n"
        "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n"
    )
    return body.replace(
        "import triton.language as tl\n\n\n",
        "import triton.language as tl\n" + imports,
        1,
    )


def _constexprs(signature: dict[str, str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, kind in signature.items():
        if kind != "constexpr":
            continue
        values[name] = 1 if name.startswith(("Y", "Z")) else 128
    return values


def _run_case(torch: Any, case: Case, directory: Path) -> dict[str, Any]:
    from torch._inductor.debug import DebugContext
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.fx_passes.replace_random import replace_random_passes
    from torch._inductor.graph import GraphLowering
    from torch._inductor.virtualized import V
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    # FakeTensor construction must observe the real no-driver state.  Inductor
    # itself checks CUDA availability later when selecting its backend.
    torch.cuda.is_available = lambda: False
    try:
        mode = FakeTensorMode()
        with mode:
            inputs = [
                torch.empty(spec.shape, device="cuda", dtype=getattr(torch, spec.dtype))
                for spec in case.inputs
            ]
            graph = make_fx(
                case.function,
                tracing_mode="fake",
                decomposition_table=select_decomp_table(),
            )(*inputs)
    except Exception as error:
        raise ProbeFailure("fx_trace", error) from error
    try:
        torch.cuda.is_available = lambda: True
        with DebugContext(), V.set_fake_mode(mode):
            replace_random_passes(graph)
            lowering = GraphLowering(graph, example_inputs=inputs, is_inference=True)
            lowering.freeze_runtime_asserts()
            with V.set_graph_handler(lowering):
                lowering.run(*inputs)
                wrapper, _ = lowering.codegen()
    except Exception as error:
        raise ProbeFailure("inductor_lowering", error) from error

    names = _kernel_names(wrapper.value)
    if not names:
        return {
            "status": "no_triton_kernel",
            "kernel_count": 0,
            "wrapper_calls": _wrapper_calls(wrapper.value),
            "seconds": round(time.monotonic() - started, 3),
        }

    hashes = []
    try:
        for index, name in enumerate(names):
            generated = _extract_inductor_kernel(wrapper.value, name)
            signature = _signature(generated, name)
            adapter_path = directory / f"{index:02d}_{name}.py"
            ttir_path = directory / f"{index:02d}_{name}.ttir"
            adapter_path.write_text(_adapter(generated, name), encoding="utf-8")
            _compile_ttir(
                adapter_path,
                name,
                signature,
                _constexprs(signature),
                ttir_path,
                f"ntops-torch/{name}.py",
            )
            hashes.append(hashlib.sha256(ttir_path.read_bytes()).hexdigest())
    except Exception as error:
        raise ProbeFailure("ttir_compile", error) from error
    return {
        "status": "ttir_generated",
        "kernel_count": len(names),
        "ttir_sha256": hashes,
        "seconds": round(time.monotonic() - started, 3),
    }


def _test_modules(repo: Path) -> set[str]:
    return {
        path.stem.removeprefix("test_") for path in (repo / "tests").glob("test_*.py")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntops-repo", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("build/ntops_torch_ttir.json")
    )
    parser.add_argument(
        "--only", action="append", default=[], help="run only the named operator"
    )
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    import torch

    repo = args.ntops_repo.resolve()
    _require_checkout(repo, NTOPS_COMMIT, "ntops")
    cases = _cases(torch)
    modules = _test_modules(repo)
    if modules != set(cases):
        missing = sorted(modules - set(cases))
        stale = sorted(set(cases) - modules)
        raise RuntimeError(
            f"operator inventory mismatch: missing={missing}, stale={stale}"
        )
    selected = args.only or sorted(cases)
    unknown = sorted(set(selected) - set(cases))
    if unknown:
        raise RuntimeError(f"unknown operators: {unknown}")

    results: dict[str, Any] = {}
    if args.in_process:
        _configure_inductor(torch)
        with tempfile.TemporaryDirectory(prefix="etv-ntops-ttir-") as temporary:
            root = Path(temporary)
            for name in selected:
                try:
                    result = _run_case(torch, cases[name], root / name)
                except ProbeFailure as failure:
                    result = {
                        "status": f"{failure.stage}_error",
                        "error_type": type(failure.original).__name__,
                        "error": str(failure).splitlines()[0][:500],
                    }
                except Exception as error:  # Retain unexpected harness failures.
                    result = {
                        "status": "harness_error",
                        "error_type": type(error).__name__,
                        "error": str(error).splitlines()[0][:500],
                    }
                results[name] = {"reference": cases[name].reference, **result}
                print(f"{name}: {result['status']}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="etv-ntops-workers-") as temporary:
            root = Path(temporary)
            for name in selected:
                worker_report = root / f"{name}.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--ntops-repo",
                        str(repo),
                        "--only",
                        name,
                        "--in-process",
                        "--output",
                        str(worker_report),
                    ],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode == 0 and worker_report.is_file():
                    result = json.loads(worker_report.read_text(encoding="utf-8"))[
                        "results"
                    ][name]
                else:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    result = {
                        "reference": cases[name].reference,
                        "status": "worker_error",
                        "error_type": "WorkerProcessError",
                        "error": (
                            detail.splitlines()[-1][:500]
                            if detail
                            else "worker produced no report"
                        ),
                    }
                results[name] = result
                print(f"{name}: {result['status']}", flush=True)

    counts: dict[str, int] = {}
    for result in results.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "format": "etv-ntops-torch-ttir-matrix-v1",
        "scope": {
            "ntops_commit": NTOPS_COMMIT,
            "operator_test_modules": len(modules),
            "specializations_per_module": 1,
            "cuda_execution": False,
            "target": "cuda:sm80",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "triton": version("triton"),
        },
        "counts": counts,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
