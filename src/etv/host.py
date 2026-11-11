"""Checked semantics for captured straight-line tensor operations.

These are semantic host steps, not compiler TTIR.  Their provenance is the
content-addressed capture event; input tensor values remain symbolic loads.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from math import prod
from pathlib import Path
import struct
from typing import TYPE_CHECKING

from .errors import InputError, UnsupportedSemantics
from .ir import BOOL, FLOAT, INDEX, Expr, Kernel, StorageRef, Store, Type, const, var
from .jsonio import JSON, array, boolean, digest, integer, mapping, obj, read_json, string
from .semantics import validate_expr

if TYPE_CHECKING:
    from .pairspec import LaunchSpec


DTYPES = {
    "torch.float16": (FLOAT, 2),
    "torch.bfloat16": (FLOAT, 2),
    "torch.float32": (FLOAT, 4),
    "torch.float64": (FLOAT, 8),
    "torch.bool": (BOOL, 1),
    **{f"torch.int{bits}": (Type("int", bits), bits // 8) for bits in (8, 16, 32, 64)},
}

OPERATORS = {
    "constant": {
        "torch.tensor",
        "aten.scalar_tensor.default",
        "aten.full.default",
        "aten.full_like.default",
        "aten.new_full.default",
        "aten.zeros.default",
        "aten.zeros_like.default",
        "aten.new_zeros.default",
        "aten.ones.default",
        "aten.ones_like.default",
        "aten.new_ones.default",
        "aten.fill_.Scalar",
        "aten.zero_.default",
    },
    "copy": {"aten.clone.default", "aten._to_copy.default", "aten.copy_.default"},
    "div_scalar": {
        "aten.div.Scalar",
        "aten.div.Scalar_mode",
        "aten.div.Tensor",
        "aten.div.Tensor_mode",
    },
    "pad": {"aten.constant_pad_nd.default"},
}


@dataclass(frozen=True)
class HostBuffer:
    storage_id: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    element: Type


@dataclass(frozen=True)
class HostOperation:
    kind: str
    operator: str
    inputs: tuple[HostBuffer, ...]
    output: HostBuffer
    value: Expr | None
    padding: tuple[int, ...]
    source_hash: str
    event: int


def _buffer(raw: JSON) -> HostBuffer:
    item = obj(
        raw,
        "kind storage_id offset element_size dtype shape stride",
        "allocation_base device storage_nbytes",
    )
    if item["kind"] != "buffer":
        raise InputError("INVALID_PAIRSPEC", "host endpoint must be a captured buffer")
    dtype = string(item["dtype"])
    if dtype not in DTYPES:
        raise UnsupportedSemantics("HOST_DTYPE_UNSUPPORTED", dtype)
    element, size = DTYPES[dtype]
    if integer(item["element_size"], 1) != size:
        raise InputError("INVALID_PAIRSPEC", "host buffer dtype/element size mismatch")
    shape = tuple(integer(x, 1) for x in array(item["shape"]))
    stride = tuple(integer(x, 0) for x in array(item["stride"]))
    offset = integer(item["offset"], 0)
    if len(shape) != len(stride):
        raise InputError("INVALID_PAIRSPEC", "host shape/stride rank mismatch")
    if "storage_nbytes" in item:
        last = offset + sum((n - 1) * s for n, s in zip(shape, stride, strict=True))
        if (last + 1) * size > integer(item["storage_nbytes"], 0):
            raise InputError("INVALID_PAIRSPEC", "host view exceeds captured allocation")
    return HostBuffer(string(item["storage_id"]), offset, shape, stride, dtype, element)


def _literal_value(raw: JSON, dtype: str) -> int | bool | Fraction:
    source = obj(raw, "kind value")
    value: int | float | bool
    if source["kind"] == "integer":
        value = integer(source["value"])
    elif source["kind"] == "boolean":
        value = boolean(source["value"])
    elif source["kind"] == "float":
        encoded = string(source["value"])
        if encoded in ("inf", "-inf", "nan"):
            raise UnsupportedSemantics("HOST_NONFINITE_UNSUPPORTED", encoded)
        exact = Fraction(encoded)
        value = float(exact)
        if Fraction(value) != exact:
            raise InputError("INVALID_PAIRSPEC", "source is not an exact Python float")
    else:
        raise InputError("INVALID_PAIRSPEC", "host constant lacks Python scalar provenance")
    typ = DTYPES[dtype][0]
    if typ == BOOL:
        return bool(value)
    if typ.kind == "int":
        return int(value)
    rounded = float(value)
    if dtype == "torch.float16":
        # c10::Half scalar construction uses the float opmath conversion first.
        rounded = struct.unpack(">f", struct.pack(">f", rounded))[0]
    if dtype != "torch.float64":
        fmt = ">e" if dtype == "torch.float16" else ">f"
        rounded = struct.unpack(fmt, struct.pack(fmt, rounded))[0]
        if dtype == "torch.bfloat16" and math.isfinite(rounded):
            bits = struct.unpack(">I", struct.pack(">f", rounded))[0]
            bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
            rounded = struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]
    if not math.isfinite(rounded):
        raise UnsupportedSemantics("HOST_NONFINITE_UNSUPPORTED", "rounded host literal")
    return Fraction(rounded)


def _scalar(raw: JSON, dtype: str) -> Expr:
    item = obj(raw, "kind dtype value source", "encoding negative_zero")
    if string(item["dtype"]) != dtype:
        raise InputError("INVALID_PAIRSPEC", "host scalar dtype disagrees with output")
    typ = DTYPES[dtype][0]
    if "negative_zero" in item:
        boolean(item["negative_zero"])
    try:
        literal = _literal_value(item["source"], dtype)
    except (ValueError, ZeroDivisionError, OverflowError, struct.error) as exc:
        raise InputError("INVALID_PAIRSPEC", "invalid host scalar literal source") from exc
    if item["kind"] == "float" and typ == FLOAT:
        if item.get("encoding") == "nonfinite":
            raise UnsupportedSemantics("HOST_NONFINITE_UNSUPPORTED", string(item["value"]))
        if item.get("encoding") != "rational":
            raise InputError("INVALID_PAIRSPEC", "host float must have rational encoding")
        try:
            finite_value = Fraction(string(item["value"]))
            if finite_value != literal:
                raise InputError("INVALID_PAIRSPEC", "host literal rounding disagrees with source")
            return const(finite_value, FLOAT)
        except (ValueError, ZeroDivisionError) as exc:
            raise InputError("INVALID_PAIRSPEC", "invalid finite host scalar") from exc
    if item["kind"] == "integer" and typ.kind == "int":
        value = integer(item["value"])
        assert typ.bits is not None
        if not -(2 ** (typ.bits - 1)) <= value < 2 ** (typ.bits - 1):
            raise InputError("INVALID_PAIRSPEC", "host integer out of storage range")
        if value != literal:
            raise InputError("INVALID_PAIRSPEC", "host integer disagrees with literal source")
        return const(value, typ)
    if item["kind"] == "boolean" and typ == BOOL and type(item["value"]) is bool:
        if item["value"] != literal:
            raise InputError("INVALID_PAIRSPEC", "host boolean disagrees with literal source")
        return const(item["value"], BOOL)
    raise InputError("INVALID_PAIRSPEC", "host scalar kind/type mismatch")


def parse_host_operation(
    path: Path,
    expected_hash: str,
    event_index: int,
    side: str,
    step: int,
) -> HostOperation:
    actual_hash = digest(path)
    if actual_hash != expected_hash:
        raise InputError("FILE_HASH_MISMATCH", str(path))
    capture = mapping(read_json(path))
    if capture.get("format") != "etv-runtime-capture-v1":
        raise InputError("INVALID_PAIRSPEC", "host step requires runtime capture v1")
    events = array(capture.get("host_operations"))
    if event_index >= len(events):
        raise InputError("INVALID_PAIRSPEC", "host capture event does not exist")
    event = obj(events[event_index], "side step operator kind inputs output attributes")
    if event["side"] != side or integer(event["step"], 0) != step:
        raise InputError("INVALID_PAIRSPEC", "host event side/step mismatch")
    kind, operator = string(event["kind"]), string(event["operator"])
    if kind not in OPERATORS or operator not in OPERATORS[kind]:
        raise UnsupportedSemantics("HOST_OPERATION_UNSUPPORTED", operator)
    inputs = tuple(_buffer(x) for x in array(event["inputs"]))
    output = _buffer(event["output"])
    if len(inputs) != (0 if kind == "constant" else 1):
        raise InputError("INVALID_PAIRSPEC", "host operation input arity mismatch")
    if any(x.dtype != output.dtype for x in inputs):
        raise UnsupportedSemantics("HOST_DTYPE_CONVERSION_UNSUPPORTED", operator)
    if any(x.storage_id == output.storage_id for x in inputs):
        raise UnsupportedSemantics("HOST_ALIAS_UNSUPPORTED", "host input aliases output")
    # This conservative sufficient condition supports dense, transposed and
    # padded non-overlapping output views without pretending zero strides copy.
    extent = 1
    for stride, size in sorted(zip(output.stride, output.shape, strict=True)):
        if size > 1:
            if stride < extent:
                raise UnsupportedSemantics("HOST_OVERLAPPING_OUTPUT_UNSUPPORTED", operator)
            extent += (size - 1) * stride
    value = None
    padding: tuple[int, ...] = ()
    if kind == "constant":
        attrs = obj(event["attributes"], "value")
        value = _scalar(attrs["value"], output.dtype)
    elif kind == "div_scalar":
        attrs = obj(event["attributes"], "divisor rounding_mode")
        if attrs["rounding_mode"] is not None or output.element != FLOAT:
            raise UnsupportedSemantics("HOST_DIVISION_UNSUPPORTED", operator)
        scalar_dtype = (
            "torch.float32" if output.dtype in ("torch.float16", "torch.bfloat16") else output.dtype
        )
        value = _scalar(attrs["divisor"], scalar_dtype)
        if inputs[0].shape != output.shape:
            raise InputError("INVALID_PAIRSPEC", "host division shape mismatch")
    elif kind == "copy":
        obj(event["attributes"], "")
        shape = inputs[0].shape
        if len(shape) > len(output.shape) or any(
            n != 1 and n != m for n, m in zip(reversed(shape), reversed(output.shape))
        ):
            raise InputError("INVALID_PAIRSPEC", "host copy broadcast shape mismatch")
    else:
        attrs = obj(event["attributes"], "padding value")
        padding = tuple(integer(x) for x in array(attrs["padding"]))
        if len(padding) % 2 or len(padding) > 2 * len(output.shape):
            raise InputError("INVALID_PAIRSPEC", "invalid host padding rank")
        expected = list(inputs[0].shape)
        for axis in range(len(padding) // 2):
            expected[-axis - 1] += padding[2 * axis] + padding[2 * axis + 1]
        if tuple(expected) != output.shape:
            raise InputError("INVALID_PAIRSPEC", "host padding output shape mismatch")
        value = _scalar(attrs["value"], output.dtype)
    return HostOperation(kind, operator, inputs, output, value, padding, actual_hash, event_index)


def _binary(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(op, BOOL if op in ("lt", "ge", "and") else INDEX, (left, right))


def _coordinates(index: Expr, shape: tuple[int, ...]) -> tuple[Expr, ...]:
    return tuple(
        _binary("rem", _binary("div", index, const(prod(shape[axis + 1 :]))), const(size))
        for axis, size in enumerate(shape)
    )


def _address(buffer: HostBuffer, coordinates: tuple[Expr, ...]) -> Expr:
    address = const(buffer.offset)
    for coordinate, stride in zip(coordinates, buffer.stride, strict=True):
        address = _binary("add", address, _binary("mul", coordinate, const(stride)))
    return address


def lift_host(launch: "LaunchSpec") -> Kernel:
    operation = launch.host_operation
    assert operation is not None
    if digest(launch.file) != operation.source_hash:
        raise InputError("FILE_HASH_MISMATCH", str(launch.file))
    endpoints = {ep.name: ep for ep in launch.abi}
    buffers = operation.inputs + (operation.output,)
    if set(endpoints) != {f"arg{i}" for i in range(len(buffers))}:
        raise InputError("ABI_ROLE_MISSING", "host physical endpoints are incomplete")
    for i, buffer in enumerate(buffers):
        ep = endpoints[f"arg{i}"]
        if ep.kind != "block" or ep.offset != buffer.offset:
            raise InputError("INVALID_PAIRSPEC", "host endpoint disagrees with captured view")
    output = endpoints[f"arg{len(operation.inputs)}"]
    if len(launch.stores) != 1 or (launch.stores[0].index, launch.stores[0].role) != (
        0,
        output.role,
    ):
        raise InputError("INVALID_PAIRSPEC", "host selected store disagrees with output")
    index = var("__pid")
    coordinates = _coordinates(index, operation.output.shape)
    value = operation.value
    if operation.inputs:
        source = operation.inputs[0]
        source_coordinates = coordinates[len(coordinates) - len(source.shape) :]
        mask = const(True, BOOL)
        if operation.kind == "pad":
            adjusted = list(source_coordinates)
            for axis in range(len(operation.padding) // 2):
                adjusted[-axis - 1] = _binary(
                    "sub", adjusted[-axis - 1], const(operation.padding[2 * axis])
                )
            source_coordinates = tuple(adjusted)
            for coordinate, size in zip(source_coordinates, source.shape, strict=True):
                mask = _binary("and", mask, _binary("ge", coordinate, const(0)))
                mask = _binary("and", mask, _binary("lt", coordinate, const(size)))
        elif operation.kind == "copy":
            source_coordinates = tuple(
                const(0) if size == 1 else coordinate
                for coordinate, size in zip(source_coordinates, source.shape, strict=True)
            )
        ep = endpoints["arg0"]
        storage = StorageRef(launch.side, launch.id, ep.name, ep.role, ep.offset)
        default = value if operation.kind == "pad" else Expr("undefined", source.element)
        assert default is not None
        loaded = Expr(
            "load",
            source.element,
            (_address(source, source_coordinates), mask, default),
            (("storage", storage),),
        )
        if operation.kind == "div_scalar":
            assert value is not None
            value = Expr("fdiv", FLOAT, (loaded, value))
        else:
            value = loaded
    assert value is not None
    validate_expr(value)
    storage = StorageRef(launch.side, launch.id, output.name, output.role, output.offset)
    store = Store(
        0, storage, index, _address(operation.output, coordinates), const(True, BOOL), value
    )
    return Kernel(
        launch.id,
        operation.operator,
        const(prod(operation.output.shape)),
        (),
        (store,),
        operation.source_hash,
    )
