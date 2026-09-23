"""Selected runtime launch capture through Python profiling, without patching Triton."""

from contextlib import contextmanager
import importlib
import importlib.metadata
import json
import hashlib
from pathlib import Path
import platform
import sys
from types import FrameType
from typing import Any, Iterator


class Capture:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.launches: list[dict[str, Any]] = []
        self.current_side: str | None = None
        self.outputs_by_side: dict[str, dict[str, str]] = {"lhs": {}, "rhs": {}}
        self.input_roles: dict[str, str] = {}
        self.storage_ids: dict[tuple[str, int], str] = {}
        self.diagnostics: list[str] = []
        self.output_sizes: dict[str, int] = {}
        self.storage_refs: list[Any] = []

    def argument(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "untyped_storage"):
            storage = value.untyped_storage()
            self.storage_refs.append(storage)
            base = storage.data_ptr()
            key = (str(value.device), base)
            identity = self.storage_ids.setdefault(key, f"storage-{len(self.storage_ids):04}")
            return {
                "kind": "buffer",
                "storage_id": identity,
                "allocation_base": base,
                "offset": (value.data_ptr() - base) // value.element_size(),
                "element_size": value.element_size(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
            }
        if type(value) is int:
            return {"kind": "integer", "value": value}
        return {"kind": "unsupported", "python_type": type(value).__name__}

    def inputs(self, values: dict[str, Any]) -> None:
        for role, value in values.items():
            argument = self.argument(value)
            if argument["kind"] != "buffer":
                raise ValueError("automatic role capture currently requires tensor inputs")
            self.input_roles[argument["storage_id"]] = role

    def outputs(self, side: str, values: dict[str, Any]) -> None:
        for role, value in values.items():
            argument = self.argument(value)
            if argument["kind"] != "buffer":
                raise ValueError("outputs must be tensors")
            self.outputs_by_side[side][argument["storage_id"]] = role
            if role == "Output":
                self.output_sizes[side] = value.numel()

    def _profile(self, frame: FrameType, event: str, result: Any) -> None:
        if event != "return" or self.current_side is None:
            return
        jit = (
            frame.f_globals.get("__name__") == "triton.runtime.jit"
            and frame.f_code.co_name == "run"
        )
        inductor = frame.f_code.co_name == "launcher" and "bin" in frame.f_globals
        if not jit and not inductor:
            return
        if jit and (frame.f_locals.get("warmup") or result is None):
            return
        selected_inductor = False
        parent = frame.f_back
        while parent is not None:
            module = parent.f_globals.get("__name__")
            name = parent.f_code.co_name
            if module == "triton.runtime.autotuner" and name == "_bench":
                return
            if module == "torch._inductor.runtime.triton_heuristics":
                if name in (
                    "bench",
                    "benchmark_all_configs",
                    "coordinate_descent_tuning",
                    "autotune_to_one_config",
                ):
                    return
                if name == "run":
                    if parent.f_locals.get("benchmark_run"):
                        return
                    launcher = parent.f_locals.get("launcher")
                    selected_inductor = getattr(launcher, "__code__", None) is frame.f_code
            parent = parent.f_back
        if inductor and not selected_inductor:
            return
        try:
            kernel = result if jit else frame.f_globals["bin"]
            bound = frame.f_locals["bound_args"] if jit else frame.f_locals
            signature = kernel.src.signature
            slots = []
            for source_name, compiler_type in signature.items():
                if compiler_type == "constexpr":
                    continue
                if not isinstance(source_name, str) or source_name not in bound:
                    raise ValueError("unsupported nested compiler signature")
                slots.append(
                    {
                        "slot": len(slots),
                        "source_name": source_name,
                        "compiler_type": compiler_type,
                        "runtime": self.argument(bound[source_name]),
                    }
                )
            ordinal = sum(launch["side"] == self.current_side for launch in self.launches)
            path = self.directory / self.current_side / f"launch-{ordinal:03}.ttir"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(kernel.asm["ttir"])
            grid = [frame.f_locals[f"grid_{axis}"] for axis in range(3)]
            self.launches.append(
                {
                    "side": self.current_side,
                    "ordinal": ordinal,
                    "step": ordinal,
                    "file": str(path.relative_to(self.directory)),
                    "function": kernel.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "grid": grid,
                    "slots": slots,
                    "compiler_signature": signature,
                    "constexprs": {str(k): repr(v) for k, v in kernel.src.constants.items()},
                    "selected_runtime_launch": True,
                }
            )
        except Exception as exc:
            self.diagnostics.append("capture mapping incomplete: " + type(exc).__name__)

    @contextmanager
    def side(self, side: str) -> Iterator[None]:
        if side not in ("lhs", "rhs") or self.current_side is not None:
            raise ValueError("invalid or nested capture side")
        previous = sys.getprofile()
        if previous is not None:
            raise RuntimeError("an existing profiler is active")
        self.current_side = side
        sys.setprofile(self._profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            self.current_side = None

    def write(self, source: dict[str, str]) -> Path:
        versions = {"python": platform.python_version()}
        for name in ("torch", "triton", "ninetoothed"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        cuda, gpu = "unavailable", "unavailable"
        try:
            torch = importlib.import_module("torch")
            cuda = str(torch.version.cuda)
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name()
        except ImportError:
            pass
        data = {
            "format": "etv-runtime-capture-v1",
            "source": source,
            "environment": {**versions, "cuda": cuda, "gpu": gpu},
            "launches": self.launches,
            "input_roles": self.input_roles,
            "outputs": self.outputs_by_side,
            "diagnostics": self.diagnostics,
        }
        if self.output_sizes.keys() == {"lhs", "rhs"} and len(set(self.output_sizes.values())) == 1:
            data["output_numel"] = self.output_sizes["lhs"]
        path = self.directory / "capture.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return path


def collect(entry: str, directory: Path, source: dict[str, str]) -> Path:
    module_name, name = entry.split(":", 1)
    function = getattr(importlib.import_module(module_name), name)
    capture = Capture(directory)
    function(capture)
    return capture.write(source)
