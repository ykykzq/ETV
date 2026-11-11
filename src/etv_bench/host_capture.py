"""Record actual ATen data dependencies alongside selected Triton launches.

The optional ``etv-host-operations-v1`` capture extension records events with
``side, step, operator, kind, inputs, output, attributes``. The step shares the
per-side counter used by Triton launches. Inputs/output are complete buffer
descriptors, including views. These events are observations of host-dispatched
operations, never compiler TTIR. A consumer must validate each operator and its
attributes before giving it semantics.

Supported kinds and attributes:
* constant: ``value`` is a typed scalar derived from a Python literal.
* copy: no attributes; input/output descriptors retain dtype and broadcast views.
* div_scalar: ``divisor`` is typed, ``rounding_mode`` is None or an explicit mode.
* pad: ``padding`` follows ATen order, ``value`` is typed (including -inf).
* unsupported: ``reason`` and ``writes`` describe the unresolved dependency.

Typed scalars retain the Python ``source`` and the value rounded to output dtype.
Finite floating values are exact rational strings; nonfinite values use explicit
strings and an encoding tag, so JSON never contains NaN/Infinity numbers. No
tensor contents are read. In particular arbitrary lift_fresh tensors, RNG state,
and scalar extraction are not treated as known constants.
"""

from contextlib import contextmanager
from fractions import Fraction
from functools import wraps
import math
import struct
import sys
from typing import Any, Iterator


def _source_scalar(value: Any) -> dict[str, Any]:
    if type(value) is bool:
        return {"kind": "boolean", "value": value}
    if type(value) is int:
        return {"kind": "integer", "value": value}
    if type(value) is float:
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "-inf" if value < 0 else "inf"
        else:
            encoded = str(Fraction(value))
        return {"kind": "float", "value": encoded}
    raise ValueError("constant provenance is not a Python scalar")


def _rounded_float(value: float, dtype: str) -> float:
    if dtype == "torch.float64":
        return value
    if dtype in ("torch.float16", "torch.float32", "torch.bfloat16"):
        try:
            rounded = struct.unpack(">f", struct.pack(">f", value))[0]
        except OverflowError:
            rounded = math.copysign(math.inf, value)
        if dtype == "torch.float16":
            # c10::Half's CUDA/x86 scalar conversion takes float32 as input.
            try:
                rounded = struct.unpack(">e", struct.pack(">e", rounded))[0]
            except OverflowError:
                rounded = math.copysign(math.inf, rounded)
        if dtype == "torch.bfloat16" and math.isfinite(rounded):
            bits = struct.unpack(">I", struct.pack(">f", rounded))[0]
            # Round float32 to nearest bfloat16, ties to even, as the scalar cast.
            bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
            rounded = struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]
        return rounded
    raise ValueError("unsupported scalar destination dtype " + dtype)


def typed_scalar(value: Any, dtype: str) -> dict[str, Any]:
    """Encode a known literal; never accept a Tensor or consult tensor contents."""
    source = _source_scalar(value)
    if dtype in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
        if type(value) is int and abs(value) > 2**53 and dtype != "torch.float64":
            raise ValueError("large integer to narrow float conversion is not modeled")
        rounded = _rounded_float(float(value), dtype)
        result = _source_scalar(rounded)
        result["encoding"] = "rational" if math.isfinite(rounded) else "nonfinite"
        if rounded == 0:
            result["negative_zero"] = math.copysign(1, rounded) < 0
    elif dtype == "torch.bool":
        result = {"kind": "boolean", "value": bool(value)}
    elif dtype in (
        "torch.uint8",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.uint16",
        "torch.uint32",
        "torch.uint64",
    ):
        converted = int(value)
        name = dtype.removeprefix("torch.")
        bits = int(name.removeprefix("uint").removeprefix("int"))
        minimum = 0 if name.startswith("u") else -(2 ** (bits - 1))
        maximum = 2**bits - 1 if name.startswith("u") else 2 ** (bits - 1) - 1
        if not minimum <= converted <= maximum:
            raise ValueError("out-of-range integer scalar conversion is not modeled")
        result = {"kind": "integer", "value": converted}
    else:
        raise ValueError("unsupported scalar destination dtype " + dtype)
    return {**result, "dtype": dtype, "source": source}


def _tensor_values(value: Any) -> list[Any]:
    if hasattr(value, "untyped_storage"):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensor_values(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_values(item)]
    return []


_VIEWS = {
    "alias",
    "detach",
    "view",
    "_unsafe_view",
    "reshape",
    "expand",
    "expand_as",
    "permute",
    "transpose",
    "t",
    "unsqueeze",
    "squeeze",
    "slice",
    "select",
    "as_strided",
    "split",
    "split_with_sizes",
    "unbind",
    "narrow",
    "diagonal",
}
_ALLOCATIONS = {"empty", "empty_strided", "empty_like", "new_empty", "new_empty_strided"}
_CONSTANT_ZERO = {"zeros", "zeros_like", "new_zeros", "zero_"}
_CONSTANT_ONE = {"ones", "ones_like", "new_ones"}
_RANDOM = ("rand", "normal", "bernoulli", "dropout", "seed")


class HostOperationRecorder:
    """The pure-Python recorder is independently testable without importing torch."""

    def __init__(self, capture: Any) -> None:
        self.capture = capture
        self.suppression = 0
        self.scalar_div_calls: list[tuple[Any, Any]] = []

    @contextmanager
    def suspended(self) -> Iterator[None]:
        self.suppression += 1
        try:
            yield
        finally:
            self.suppression -= 1

    def _enabled(self) -> bool:
        if self.suppression or self.capture.current_side is None:
            return False
        frame = sys._getframe(1)
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
            if module == "triton.runtime.autotuner" and name == "_bench":
                return False
            if module == "torch._inductor.runtime.triton_heuristics" and (
                name
                in {
                    "bench",
                    "benchmark_all_configs",
                    "coordinate_descent_tuning",
                    "autotune_to_one_config",
                }
                or (name == "run" and frame.f_locals.get("benchmark_run"))
            ):
                return False
            frame = frame.f_back
        return True

    def _describe(self, value: Any) -> list[dict[str, Any]]:
        with self.suspended():
            return [self.capture.argument(tensor) for tensor in _tensor_values(value)]

    def _append(
        self,
        operator: str,
        kind: str,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        attributes: dict[str, Any],
    ) -> None:
        event = {
            "side": self.capture.current_side,
            "step": self.capture.next_step(),
            "operator": operator,
            "kind": kind,
            "inputs": inputs,
            "output": outputs[0] if len(outputs) == 1 else None,
            "attributes": attributes,
        }
        if len(outputs) != 1:
            event["outputs"] = outputs
        self.capture.host_operations.append(event)
        if kind == "unsupported":
            self.capture.diagnostics.append(
                f"unsupported host operation ({event['side']} step {event['step']}): "
                f"{operator}: {attributes['reason']}"
            )

    def constructor(self, function: Any, args: tuple, kwargs: dict) -> Any:
        """Observe torch.tensor literal provenance before its lift_fresh loses it."""
        if not self._enabled():
            return function(*args, **kwargs)
        data = args[0] if args else kwargs.get("data")
        if hasattr(data, "untyped_storage"):
            # Copy construction from a tensor must retain its data dependency.
            return function(*args, **kwargs)
        with self.suspended():
            result = function(*args, **kwargs)
            outputs = self._describe(result)
        try:
            if len(outputs) != 1:
                raise ValueError("scalar constructor did not return one tensor")
            value = typed_scalar(data, outputs[0]["dtype"])
        except (ValueError, TypeError, OverflowError) as exc:
            self._append(
                "torch.tensor", "unsupported", [], outputs, {"reason": str(exc), "writes": True}
            )
        else:
            self._append("torch.tensor", "constant", [], outputs, {"value": value})
        return result

    def scalar_div(self, function: Any, args: tuple, kwargs: dict) -> Any:
        """Retain a Python scalar if ATen wraps it into a temporary scalar tensor."""
        tensor = args[0] if args else kwargs.get("input")
        divisor = args[1] if len(args) > 1 else kwargs.get("other")
        if not hasattr(tensor, "untyped_storage") or type(divisor) not in (bool, int, float):
            return function(*args, **kwargs)
        self.scalar_div_calls.append((tensor, divisor))
        try:
            return function(*args, **kwargs)
        finally:
            self.scalar_div_calls.pop()

    def dispatch(self, function: Any, args: tuple, kwargs: dict) -> Any:
        if not self._enabled():
            return function(*args, **kwargs)
        inputs = self._describe((args, kwargs))
        result = function(*args, **kwargs)
        outputs = self._describe(result)
        operator = str(function)
        parts = operator.split(".")
        name = parts[1] if len(parts) >= 3 else operator
        if name in _ALLOCATIONS:
            return result  # Allocations do not define initialized values.
        if name in _VIEWS and all(
            output["storage_id"] in {item["storage_id"] for item in inputs} for output in outputs
        ):
            return result  # The consumer's descriptor retains the alias/strides.
        try:
            self._record(operator, name, args, kwargs, inputs, outputs)
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            schema = getattr(function, "_schema", None)
            writes = any(
                getattr(getattr(argument, "alias_info", None), "is_write", False)
                for argument in getattr(schema, "arguments", [])
            )
            self._append(
                operator, "unsupported", inputs, outputs, {"reason": str(exc), "writes": writes}
            )
        return result

    def _record(self, operator, name, args, kwargs, inputs, outputs) -> None:
        if any(part in name for part in _RANDOM):
            raise ValueError("stochastic state is unpaired; no shared seed or mask is assumed")
        if len(outputs) != 1:
            raise ValueError("operation does not have a supported single tensor output")
        dtype = outputs[0]["dtype"]

        def arg(index, key, default=None):
            return args[index] if len(args) > index else kwargs.get(key, default)

        attributes: dict[str, Any] = {}
        if operator in {"aten.clone.default", "aten._to_copy.default"}:
            kind = "copy"
            inputs = self._describe(arg(0, "self"))
        elif operator == "aten.copy_.default":
            kind = "copy"
            # Destination storage is represented by output, not an extra source.
            inputs = self._describe(arg(1, "src"))
            if len(inputs) != 1:
                raise ValueError("copy_ source is not a tensor")
        elif operator in {
            "aten.div.Scalar",
            "aten.div.Scalar_mode",
            "aten.div.Tensor",
            "aten.div.Tensor_mode",
        }:
            kind = "div_scalar"
            source = arg(0, "self")
            divisor = arg(1, "other")
            if type(divisor) not in (bool, int, float) and self.scalar_div_calls:
                original, literal = self.scalar_div_calls[-1]
                if original is source:
                    divisor = literal
            inputs = self._describe(source)
            scalar_dtype = (
                "torch.float32" if dtype in ("torch.float16", "torch.bfloat16") else dtype
            )
            attributes = {
                "divisor": typed_scalar(divisor, scalar_dtype),
                "rounding_mode": arg(2, "rounding_mode"),
            }
        elif operator == "aten.constant_pad_nd.default":
            kind = "pad"
            padding = arg(1, "pad")
            if not isinstance(padding, (tuple, list)) or any(type(v) is not int for v in padding):
                raise ValueError("padding is not a literal integer sequence")
            attributes = {
                "padding": list(padding),
                "value": typed_scalar(arg(2, "value", 0), dtype),
            }
        elif operator == "aten.scalar_tensor.default":
            kind, inputs = "constant", []
            attributes = {"value": typed_scalar(arg(0, "s"), dtype)}
        elif operator == "aten.fill_.Scalar":
            kind, inputs = "constant", []
            attributes = {"value": typed_scalar(arg(1, "value"), dtype)}
        elif name in _CONSTANT_ZERO | _CONSTANT_ONE and operator.endswith(".default"):
            kind, inputs = "constant", []
            attributes = {"value": typed_scalar(1 if name in _CONSTANT_ONE else 0, dtype)}
        elif operator in {"aten.full.default", "aten.full_like.default", "aten.new_full.default"}:
            kind, inputs = "constant", []
            index = 2 if name == "new_full" else 1
            attributes = {"value": typed_scalar(arg(index, "fill_value"), dtype)}
        else:
            raise ValueError("ATen data dependency has no supported host semantics")
        self._append(operator, kind, inputs, outputs, attributes)


@contextmanager
def capture_host_operations(capture: Any) -> Iterator[None]:
    """Install public torch modes only when host capture was explicitly requested."""
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    recorder = HostOperationRecorder(capture)
    scalar_div_functions = {
        torch.div,
        torch.divide,
        torch.true_divide,
        torch.Tensor.div,
        torch.Tensor.divide,
        torch.Tensor.true_divide,
        torch.Tensor.__truediv__,
    }

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            return recorder.dispatch(function, args, kwargs or {})

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            if function is torch.tensor:
                return recorder.constructor(function, args, kwargs or {})
            if function in scalar_div_functions:
                return recorder.scalar_div(function, args, kwargs or {})
            return function(*args, **(kwargs or {}))

    with FunctionMode(), DispatchMode():
        yield


def instrument_compiled_callable(capture: Any, compiled: Any) -> Any:
    """Instrument the compiled runtime, outside Dynamo's compilation/guard path.

    A user dispatch mode around torch.compile's outer wrapper may force eager
    fallback. Installing it inside the backend-produced callable preserves the
    selected compiled program. Warmup calls are passed through without recording.
    wraps also retains the backend's optional _boxed_call calling convention.
    """

    @wraps(compiled)
    def run(*args, **kwargs):
        if capture.current_side is None or not capture.capture_host_operations:
            return compiled(*args, **kwargs)
        with capture_host_operations(capture):
            return compiled(*args, **kwargs)

    # Inductor returns a Dynamo-disabled callable. functools.wraps copies its
    # unwrapping marker, which would make Dynamo bypass this instrumentation and
    # call the original AOT graph directly. Keep the boxed calling convention,
    # but make this wrapper the actual runtime endpoint.
    run.__dict__.pop("_torchdynamo_orig_callable", None)
    return run
