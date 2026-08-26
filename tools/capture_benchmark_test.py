"""Capture exact LHS and original-reference RHS artifacts for one pytest node."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


_CASE_DIR = Path(os.environ["ETV_BENCHMARK_CASE_DIR"]).resolve()
_CASE_DIR.mkdir(parents=True, exist_ok=True)
_LHS_DIR = _CASE_DIR / "lhs"
_RHS_DIR = _CASE_DIR / "rhs"
_LHS_DIR.mkdir(exist_ok=True)
_RHS_DIR.mkdir(exist_ok=True)

_HANDLES: list[Any] = []
_LHS_LAUNCHES: list[dict[str, Any]] = []
_REFERENCE_RECORDS: list["ReferenceRecord"] = []
_ORIGINAL_MAKE: Any = None
_ORIGINAL_AUTOTUNER_RUN: Any = None
_ORIGINAL_JITFUNCTION_RUN: Any = None
_ORIGINAL_INDUCTOR_AUTOTUNER_RUN: Any = None
_CURRENT_REFERENCE_DIR: Path | None = None
_CURRENT_RHS_LAUNCHES: list[dict[str, Any]] | None = None
_LHS_AUTOTUNER_DEPTH = 0
_PHASE = "lhs"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_value(value: Any) -> Any:
    try:
        import torch
        from triton.runtime.autotuner import Config

        if isinstance(value, torch.Tensor):
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": value.requires_grad,
            }
        if isinstance(value, torch.dtype):
            return {"kind": "torch.dtype", "value": str(value)}
        if isinstance(value, torch.device):
            return {"kind": "torch.device", "value": str(value)}
        if isinstance(value, Config):
            return {
                "kind": "triton.Config",
                "kwargs": value.kwargs,
                "num_warps": value.num_warps,
                "num_ctas": value.num_ctas,
                "num_stages": value.num_stages,
                "maxnreg": value.maxnreg,
            }
    except (ImportError, AttributeError):
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "items": [_json_value(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {str(key): _json_value(item) for key, item in value.items()},
        }
    return {"kind": type(value).__qualname__, "repr": repr(value)}


def _safe_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@dataclass
class ValueSnapshot:
    names: list[str]
    structures: dict[str, tuple[list[Any], Any, list[int]]]
    tensor_bytes: bytes

    @classmethod
    def capture(cls, names: list[str], namespace: dict[str, Any]) -> "ValueSnapshot":
        import torch
        from torch.utils import _pytree

        structures: dict[str, tuple[list[Any], Any, list[int]]] = {}
        tensors: list[Any] = []
        for name in names:
            leaves, spec = _pytree.tree_flatten(namespace[name])
            saved_leaves: list[Any] = []
            tensor_indices: list[int] = []
            for leaf in leaves:
                if isinstance(leaf, torch.Tensor):
                    tensor_indices.append(len(tensors))
                    tensors.append(leaf.detach())
                    saved_leaves.append(None)
                else:
                    tensor_indices.append(-1)
                    saved_leaves.append(_safe_copy(leaf))
            structures[name] = (saved_leaves, spec, tensor_indices)
        buffer = io.BytesIO()
        torch.save(tuple(tensors), buffer)
        return cls(names=names, structures=structures, tensor_bytes=buffer.getvalue())

    def restore(self) -> dict[str, Any]:
        import torch
        from torch.utils import _pytree

        tensors = torch.load(io.BytesIO(self.tensor_bytes), weights_only=False)
        namespace: dict[str, Any] = {}
        for name in self.names:
            saved_leaves, spec, tensor_indices = self.structures[name]
            leaves = [
                tensors[tensor_index] if tensor_index >= 0 else _safe_copy(saved)
                for saved, tensor_index in zip(saved_leaves, tensor_indices, strict=True)
            ]
            namespace[name] = _pytree.tree_unflatten(leaves, spec)
        return namespace


@dataclass
class RngSnapshot:
    cpu: Any
    cuda: list[Any]

    @classmethod
    def capture(cls) -> "RngSnapshot":
        import torch

        return cls(torch.get_rng_state().clone(), torch.cuda.get_rng_state_all())

    @contextlib.contextmanager
    def restore_temporarily(self):
        import torch

        old_cpu = torch.get_rng_state()
        old_cuda = torch.cuda.get_rng_state_all()
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state_all(self.cuda)
        try:
            yield
        finally:
            torch.set_rng_state(old_cpu)
            torch.cuda.set_rng_state_all(old_cuda)


@dataclass
class ReferenceSite:
    index: int
    lineno: int
    end_lineno: int
    expression: str
    code: Any
    free_names: list[str]
    target_names: list[str]


@dataclass
class ReferenceRecord:
    site: ReferenceSite
    execution: int
    values: ValueSnapshot
    rng: RngSnapshot
    capture_origin: str = "line_trace"
    output_bytes: bytes | None = None
    output_description: Any = None


def _target_names(target: ast.expr) -> list[str]:
    return [node.id for node in ast.walk(target) if isinstance(node, ast.Name)]


def _is_reference_target(names: list[str]) -> bool:
    for name in names:
        lowered = name.lower()
        if (
            lowered.startswith("reference_output")
            or lowered.startswith("reference_result")
            or lowered.startswith("reference_value")
        ):
            return True
        if lowered in {"expected", "ref_out", "ref_vals", "ref_idxs", "ref_indices"}:
            return True
    return False


def _reference_sites(function: Any) -> list[ReferenceSite]:
    source_path = Path(function.__code__.co_filename)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function.__name__
            and node.lineno == function.__code__.co_firstlineno
        ),
        None,
    )
    if function_node is None:
        function_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function.__name__
        )
    sites: list[ReferenceSite] = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            targets = node.targets
            expression = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            expression = node.value
        else:
            continue
        names = [name for target in targets for name in _target_names(target)]
        if expression is None or not _is_reference_target(names):
            continue
        free_names = sorted(
            {
                child.id
                for child in ast.walk(expression)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }
        )
        expression_ast = ast.Expression(body=expression)
        ast.fix_missing_locations(expression_ast)
        sites.append(
            ReferenceSite(
                index=len(sites),
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                expression=ast.unparse(expression),
                code=compile(expression_ast, str(source_path), "eval"),
                free_names=free_names,
                target_names=names,
            )
        )
    return sorted(sites, key=lambda site: (site.lineno, site.index))


def _snapshot_output(names: list[str], namespace: dict[str, Any]) -> tuple[bytes, Any]:
    import torch

    output = (
        namespace[names[0]]
        if len(names) == 1
        else tuple(namespace[name] for name in names)
    )
    buffer = io.BytesIO()
    torch.save(output, buffer)
    return buffer.getvalue(), _json_value(output)


def _compiled_kernel_ttir(kernel: Any) -> str | None:
    try:
        asm = kernel.asm
        ttir = asm.get("ttir")
        return str(ttir) if ttir is not None else None
    except Exception:
        return None


def _normalize_grid(grid: Any) -> list[int] | None:
    if isinstance(grid, int):
        return [grid, 1, 1]
    if isinstance(grid, (list, tuple)) and 1 <= len(grid) <= 3:
        values = list(grid) + [1] * (3 - len(grid))
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return values
    return None


def _program_count(grid: list[int] | None) -> int | None:
    if grid is None:
        return None
    result = 1
    for value in grid:
        result *= value
    return result


def pytest_configure(config: pytest.Config) -> None:
    del config
    import ninetoothed
    import ninetoothed.generation
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    global _ORIGINAL_MAKE, _ORIGINAL_AUTOTUNER_RUN, _ORIGINAL_JITFUNCTION_RUN
    global _ORIGINAL_INDUCTOR_AUTOTUNER_RUN
    ninetoothed.generation.CACHE_DIR = _LHS_DIR / "ninetoothed-source"
    ninetoothed.generation.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _ORIGINAL_MAKE = ninetoothed.make

    def recording_make(*args: Any, **kwargs: Any) -> Any:
        handle = _ORIGINAL_MAKE(*args, **kwargs)
        _HANDLES.append(handle)
        return handle

    ninetoothed.make = recording_make
    _ORIGINAL_AUTOTUNER_RUN = Autotuner.run

    def recording_autotuner_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LHS_AUTOTUNER_DEPTH
        launch_index = len(_LHS_LAUNCHES)
        runtime_arguments = [
            {"position": index, "name": name, "value": _json_value(value)}
            for index, (name, value) in enumerate(
                zip(getattr(self, "arg_names", ()), args, strict=False)
            )
        ]
        try:
            _LHS_AUTOTUNER_DEPTH += 1
            try:
                kernel = _ORIGINAL_AUTOTUNER_RUN(self, *args, **kwargs)
            finally:
                _LHS_AUTOTUNER_DEPTH -= 1
        except Exception as error:
            if _PHASE == "lhs":
                _LHS_LAUNCHES.append(
                    {
                        "index": launch_index,
                        "status": "failed",
                        "error": repr(error),
                    }
                )
            raise
        if _PHASE == "lhs":
            ttir = _compiled_kernel_ttir(kernel)
            raw_grid = kwargs.get("grid")
            if callable(raw_grid):
                try:
                    grid_meta = {
                        **dict(zip(getattr(self, "arg_names", ()), args, strict=False)),
                        **kwargs,
                        **getattr(self, "best_config").all_kwargs(),
                    }
                    raw_grid = raw_grid(grid_meta)
                except Exception:
                    raw_grid = None
            grid = _normalize_grid(raw_grid)
            launch: dict[str, Any] = {
                "index": launch_index,
                "status": "captured" if ttir else "ttir_unavailable",
                "best_config": _json_value(getattr(self, "best_config", None)),
                "runtime_arguments": runtime_arguments,
                "grid": grid,
                "programs": _program_count(grid),
                "kernel_name": getattr(kernel, "name", None),
                "kernel_hash": getattr(kernel, "hash", None),
            }
            if ttir:
                data = ttir.encode("utf-8")
                ttir_path = _LHS_DIR / f"launch-{launch_index:03d}.ttir"
                ttir_path.write_bytes(data)
                launch.update(
                    {
                        "ttir": str(ttir_path),
                        "ttir_sha256": _sha256_bytes(data),
                    }
                )
            _LHS_LAUNCHES.append(launch)
        return kernel

    Autotuner.run = recording_autotuner_run

    _ORIGINAL_JITFUNCTION_RUN = JITFunction.run

    def recording_jitfunction_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _ORIGINAL_JITFUNCTION_RUN(self, *args, **kwargs)
        if _PHASE != "lhs" or _LHS_AUTOTUNER_DEPTH != 0 or kwargs.get("warmup"):
            return kernel
        launch_index = len(_LHS_LAUNCHES)
        raw_grid = kwargs.get("grid")
        if callable(raw_grid):
            try:
                raw_grid = raw_grid(
                    dict(zip(getattr(self, "arg_names", ()), args, strict=False))
                )
            except Exception:
                raw_grid = None
        grid = _normalize_grid(raw_grid)
        ttir = _compiled_kernel_ttir(kernel)
        launch: dict[str, Any] = {
            "index": launch_index,
            "status": "captured" if ttir else "ttir_unavailable",
            "launch_kind": "jit_function",
            "runtime_arguments": [
                {"position": index, "name": name, "value": _json_value(value)}
                for index, (name, value) in enumerate(
                    zip(getattr(self, "arg_names", ()), args, strict=False)
                )
            ],
            "grid": grid,
            "programs": _program_count(grid),
            "kernel_name": getattr(kernel, "name", None),
            "kernel_hash": getattr(kernel, "hash", None),
        }
        if ttir:
            data = ttir.encode("utf-8")
            ttir_path = _LHS_DIR / f"launch-{launch_index:03d}.ttir"
            ttir_path.write_bytes(data)
            launch.update(
                {"ttir": str(ttir_path), "ttir_sha256": _sha256_bytes(data)}
            )
        _LHS_LAUNCHES.append(launch)
        return kernel

    JITFunction.run = recording_jitfunction_run

    _ORIGINAL_INDUCTOR_AUTOTUNER_RUN = CachingAutotuner.run

    def recording_inductor_autotuner_run(
        self: Any, *args: Any, stream: Any, **kwargs: Any
    ) -> Any:
        result = _ORIGINAL_INDUCTOR_AUTOTUNER_RUN(
            self, *args, stream=stream, **kwargs
        )
        if _PHASE != "rhs" or _CURRENT_RHS_LAUNCHES is None:
            return result

        launch_index = len(_CURRENT_RHS_LAUNCHES)
        launcher = self.launchers[0] if len(self.launchers) == 1 else None
        config = getattr(launcher, "config", None)
        grid = None
        if config is not None:
            try:
                _, raw_grid = self._interpret_args_grid(args, config)
                grid = _normalize_grid(raw_grid)
            except Exception:
                pass
        cache_hash = getattr(launcher, "cache_hash", None)
        ttir = None
        kernel_name = getattr(getattr(self, "fn", None), "__name__", None)
        for compile_result in getattr(self, "compile_results", ()):
            if getattr(compile_result, "config", None) != config:
                continue
            kernel = getattr(compile_result, "kernel", None)
            ttir = _compiled_kernel_ttir(kernel)
            kernel_name = getattr(kernel, "name", kernel_name)
            break
        launch: dict[str, Any] = {
            "index": launch_index,
            "status": "captured" if ttir else "cache_resolved",
            "kernel_name": kernel_name,
            "cache_hash": cache_hash,
            "config": _json_value(config),
            "grid": grid,
            "programs": _program_count(grid),
            "runtime_arguments": [
                {"position": index, "value": _json_value(value)}
                for index, value in enumerate(args)
            ],
        }
        if ttir and _CURRENT_REFERENCE_DIR is not None:
            data = ttir.encode("utf-8")
            ttir_path = _CURRENT_REFERENCE_DIR / f"selected-{launch_index:03d}.ttir"
            ttir_path.write_bytes(data)
            launch.update(
                {"ttir": str(ttir_path), "ttir_sha256": _sha256_bytes(data)}
            )
        _CURRENT_RHS_LAUNCHES.append(launch)
        return result

    CachingAutotuner.run = recording_inductor_autotuner_run


def _make_reference_callable(
    site: ReferenceSite,
    snapshot: ValueSnapshot,
    test_globals: dict[str, Any],
) -> tuple[Any, list[Any], list[str]]:
    import torch
    from torch.utils import _pytree

    restored = snapshot.restore()
    tensor_names: list[str] = []
    example_tensors: list[Any] = []
    templates: dict[str, tuple[list[Any], Any, list[int]]] = {}
    for name in snapshot.names:
        leaves, spec = _pytree.tree_flatten(restored[name])
        template: list[Any] = []
        indices: list[int] = []
        for leaf_index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor):
                indices.append(len(example_tensors))
                example_tensors.append(leaf)
                tensor_names.append(f"{name}[{leaf_index}]")
                template.append(None)
            else:
                indices.append(-1)
                template.append(leaf)
        templates[name] = (template, spec, indices)

    def reference_callable(*tensor_args: Any) -> Any:
        namespace: dict[str, Any] = {}
        for name in snapshot.names:
            template, spec, indices = templates[name]
            leaves = [
                tensor_args[index] if index >= 0 else value
                for value, index in zip(template, indices, strict=True)
            ]
            namespace[name] = _pytree.tree_unflatten(leaves, spec)
        return eval(site.code, test_globals, namespace)

    return reference_callable, example_tensors, tensor_names


def _compare_outputs(actual: Any, expected: Any) -> dict[str, Any]:
    import torch
    from torch.utils import _pytree

    actual_leaves, actual_spec = _pytree.tree_flatten(actual)
    expected_leaves, expected_spec = _pytree.tree_flatten(expected)
    result: dict[str, Any] = {
        "structure_equal": actual_spec == expected_spec,
        "leaf_count_actual": len(actual_leaves),
        "leaf_count_expected": len(expected_leaves),
        "tensor_leaves": [],
    }
    if actual_spec != expected_spec or len(actual_leaves) != len(expected_leaves):
        return result
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(actual_leaf, torch.Tensor) and isinstance(expected_leaf, torch.Tensor):
            difference = None
            if actual_leaf.shape == expected_leaf.shape and actual_leaf.dtype == expected_leaf.dtype:
                if actual_leaf.dtype == torch.bool:
                    difference = 0.0 if torch.equal(actual_leaf, expected_leaf) else 1.0
                else:
                    difference = float(
                        (actual_leaf - expected_leaf).abs().max().item()
                    ) if actual_leaf.numel() else 0.0
            result["tensor_leaves"].append(
                {
                    "equal": bool(torch.equal(actual_leaf, expected_leaf)),
                    "allclose_1e-3": bool(
                        actual_leaf.shape == expected_leaf.shape
                        and torch.allclose(
                            actual_leaf,
                            expected_leaf,
                            rtol=1e-3,
                            atol=1e-3,
                            equal_nan=True,
                        )
                    ),
                    "max_abs_error": difference,
                }
            )
    return result


def _compile_reference(
    record: ReferenceRecord,
    test_globals: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    global _CURRENT_REFERENCE_DIR, _CURRENT_RHS_LAUNCHES, _PHASE
    reference_dir = _RHS_DIR / f"reference-{record.site.index:03d}-{record.execution:03d}"
    reference_dir.mkdir(parents=True, exist_ok=True)
    triton_dir = reference_dir / "triton-cache"
    inductor_dir = reference_dir / "torchinductor-cache"
    triton_dir.mkdir(exist_ok=True)
    inductor_dir.mkdir(exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    _PHASE = "rhs"
    _CURRENT_REFERENCE_DIR = reference_dir
    rhs_launches: list[dict[str, Any]] = []
    _CURRENT_RHS_LAUNCHES = rhs_launches

    callable_, example_inputs, input_names = _make_reference_callable(
        record.site, record.values, test_globals
    )
    report: dict[str, Any] = {
        "site_index": record.site.index,
        "execution": record.execution,
        "capture_origin": record.capture_origin,
        "line": record.site.lineno,
        "expression": record.site.expression,
        "target_names": record.site.target_names,
        "input_names": input_names,
        "input_descriptions": [_json_value(value) for value in example_inputs],
        "captured_output": record.output_description,
    }
    try:
        with record.rng.restore_temporarily():
            graph_module = make_fx(callable_, tracing_mode="real")(*example_inputs)
        (reference_dir / "fx-graph.txt").write_text(
            str(graph_module.graph) + "\n", encoding="utf-8"
        )
        (reference_dir / "fx-code.py").write_text(
            graph_module.code + "\n", encoding="utf-8"
        )
        try:
            torch.save(graph_module, reference_dir / "fx-graph.pt")
        except Exception:
            report["fx_pickle_error"] = traceback.format_exc()

        compiled = torch.compile(graph_module, fullgraph=True, dynamic=False)
        fresh_inputs = _make_reference_callable(
            record.site, record.values, test_globals
        )[1]
        with record.rng.restore_temporarily():
            eager_output = graph_module(*fresh_inputs)
        fresh_inputs = _make_reference_callable(
            record.site, record.values, test_globals
        )[1]
        with record.rng.restore_temporarily():
            compiled_output = compiled(*fresh_inputs)
            torch.cuda.synchronize()

        captured_output = (
            torch.load(io.BytesIO(record.output_bytes), weights_only=False)
            if record.output_bytes is not None
            else None
        )
        report["checks"] = {
            "compiled_vs_fx_eager": _compare_outputs(compiled_output, eager_output),
            "compiled_vs_captured_reference": (
                _compare_outputs(compiled_output, captured_output)
                if captured_output is not None
                else None
            ),
        }
        report["status"] = "compiled"
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )

    ttir_files = sorted(triton_dir.rglob("*.ttir"))
    best_config_files = sorted(inductor_dir.rglob("*.best_config"))
    report["ttir"] = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in ttir_files
    ]
    report["best_configs"] = [
        {
            "path": str(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        }
        for path in best_config_files
    ]
    selected_hashes = {
        item["content"].get("triton_cache_hash")
        for item in report["best_configs"]
        if item["content"].get("triton_cache_hash")
    }
    selected_hashes.update(
        launch["cache_hash"] for launch in rhs_launches if launch.get("cache_hash")
    )
    selected_paths = [path for path in ttir_files if path.parent.name in selected_hashes]
    if not selected_paths and len(ttir_files) == 1 and len(rhs_launches) <= 1:
        selected_paths = ttir_files
    report["selected_ttir"] = [
        {
            "path": launch["ttir"],
            "source": "compiled_binary",
            "sha256": launch["ttir_sha256"],
            "bytes": Path(launch["ttir"]).stat().st_size,
        }
        for launch in rhs_launches
        if launch.get("ttir")
    ]
    selected_digests = {item["sha256"] for item in report["selected_ttir"]}
    for path in selected_paths:
        digest = _sha256(path)
        if digest in selected_digests:
            continue
        index = len(report["selected_ttir"])
        stable_path = reference_dir / f"selected-{index:03d}.ttir"
        shutil.copy2(path, stable_path)
        report["selected_ttir"].append(
            {
                "path": str(stable_path),
                "source_path": str(path),
                "source": "triton_cache",
                "sha256": digest,
                "bytes": stable_path.stat().st_size,
            }
        )
        selected_digests.add(digest)
    source_dir = reference_dir / "inductor-source"
    source_records = []
    for index, path in enumerate(sorted(inductor_dir.rglob("*.py"))):
        source_dir.mkdir(exist_ok=True)
        stable_path = source_dir / f"{index:03d}-{path.name}"
        shutil.copy2(path, stable_path)
        source_records.append(
            {"path": str(stable_path), "sha256": _sha256(stable_path)}
        )
    report["inductor_sources"] = source_records
    report["runtime_launches"] = rhs_launches
    if report.get("status") == "compiled" and not report["selected_ttir"]:
        report["ttir_status"] = "no_triton_kernel_launched"
    elif report["selected_ttir"]:
        report["ttir_status"] = "captured"
    else:
        report["ttir_status"] = "unavailable"
    _CURRENT_REFERENCE_DIR = None
    _CURRENT_RHS_LAUNCHES = None
    (reference_dir / "capture.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    import torch

    target_code = pyfuncitem.obj.__code__
    sites = _reference_sites(pyfuncitem.obj)
    sites_by_line: dict[int, list[ReferenceSite]] = {}
    for site in sites:
        sites_by_line.setdefault(site.lineno, []).append(site)
    pending: ReferenceRecord | None = None
    executions: dict[int, int] = {}
    final_frame_locals: dict[str, Any] = {}
    final_rng: RngSnapshot | None = None

    def finish_pending(frame: Any) -> None:
        nonlocal pending
        if pending is None:
            return
        if all(name in frame.f_locals for name in pending.site.target_names):
            try:
                output_bytes, description = _snapshot_output(
                    pending.site.target_names, frame.f_locals
                )
                pending.output_bytes = output_bytes
                pending.output_description = description
            except Exception:
                pending.output_description = {"snapshot_error": traceback.format_exc()}
        _REFERENCE_RECORDS.append(pending)
        pending = None

    def trace(frame: Any, event: str, arg: Any):
        nonlocal pending, final_rng
        del arg
        if frame.f_code is not target_code:
            return trace
        if event == "line":
            if pending is not None and not (
                pending.site.lineno <= frame.f_lineno <= pending.site.end_lineno
            ):
                finish_pending(frame)
            for site in sites_by_line.get(frame.f_lineno, []):
                # Python's line table may revisit the first line of a multiline
                # expression while evaluating that same statement.
                if pending is not None and pending.site.index == site.index:
                    continue
                finish_pending(frame)
                if not all(
                    name in frame.f_locals or name in frame.f_globals
                    for name in site.free_names
                ):
                    continue
                namespace = {
                    name: frame.f_locals.get(name, frame.f_globals.get(name))
                    for name in site.free_names
                }
                execution = executions.get(site.index, 0)
                executions[site.index] = execution + 1
                pending = ReferenceRecord(
                    site=site,
                    execution=execution,
                    values=ValueSnapshot.capture(site.free_names, namespace),
                    rng=RngSnapshot.capture(),
                )
        elif event in {"return", "exception"}:
            finish_pending(frame)
            final_frame_locals.clear()
            final_frame_locals.update(frame.f_locals)
            final_rng = RngSnapshot.capture()
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    outcome = yield
    sys.settrace(previous_trace)

    # A failing implementation call can prevent the original test from reaching
    # its reference assignment. The frame still contains the exact parameterized
    # inputs, so reconstruct that original AST expression instead of dropping RHS.
    for site in sites:
        if executions.get(site.index, 0) != 0:
            continue
        if not all(
            name in final_frame_locals or name in pyfuncitem.obj.__globals__
            for name in site.free_names
        ):
            continue
        namespace = {
            name: final_frame_locals.get(name, pyfuncitem.obj.__globals__.get(name))
            for name in site.free_names
        }
        try:
            values = ValueSnapshot.capture(site.free_names, namespace)
        except Exception:
            continue
        _REFERENCE_RECORDS.append(
            ReferenceRecord(
                site=site,
                execution=0,
                values=values,
                rng=final_rng or RngSnapshot.capture(),
                capture_origin="exception_frame_recovery",
            )
        )
        executions[site.index] = 1

    rhs_reports = [
        _compile_reference(record, pyfuncitem.obj.__globals__)
        for record in _REFERENCE_RECORDS
    ]
    callspec = getattr(pyfuncitem, "callspec", None)
    report = {
        "format": "etv-benchmark-test-capture-v1",
        "nodeid": pyfuncitem.nodeid,
        "test_file": str(Path(pyfuncitem.obj.__code__.co_filename).resolve()),
        "test_function": pyfuncitem.obj.__qualname__,
        "callspec": (
            {name: _json_value(value) for name, value in callspec.params.items()}
            if callspec is not None
            else {}
        ),
        "reference_sites": [
            {
                "index": site.index,
                "line": site.lineno,
                "end_line": site.end_lineno,
                "expression": site.expression,
                "free_names": site.free_names,
                "target_names": site.target_names,
            }
            for site in sites
        ],
        "reference_executions": len(_REFERENCE_RECORDS),
        "reference_recovered_after_exception": sum(
            record.capture_origin == "exception_frame_recovery"
            for record in _REFERENCE_RECORDS
        ),
        "lhs_launches": _LHS_LAUNCHES,
        "rhs_references": rhs_reports,
        "ninetoothed_sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted((_LHS_DIR / "ninetoothed-source").glob("*.py"))
        ],
        "pytest_exception": (
            repr(outcome.excinfo[1]) if outcome.excinfo is not None else None
        ),
        "runtime_locals": {
            name: _json_value(value) for name, value in final_frame_locals.items()
        },
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    (_CASE_DIR / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if os.environ.get("ETV_BENCHMARK_KEEP_CACHES", "0") != "1":
        shutil.rmtree(_LHS_DIR / "triton-cache", ignore_errors=True)
        for reference_dir in _RHS_DIR.glob("reference-*"):
            shutil.rmtree(reference_dir / "triton-cache", ignore_errors=True)
            shutil.rmtree(reference_dir / "torchinductor-cache", ignore_errors=True)
