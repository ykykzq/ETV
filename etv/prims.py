"""Strict Torch Prims JSON reader and Semantic ETV lifting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .model import (
    Expr,
    InputError,
    Program,
    Sort,
    StoreTemplate,
    bool_const,
    float_const,
    int_const,
)
from .schema import parse_expr


FORMAT_PRIMS = "etv-prims-program-v1"
PRIMS_FRONTEND_VERSION = "1"


def _only_keys(value: Mapping[str, object], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InputError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}", "READ_ERROR") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}", "PARSE_ERROR") from exc
    if not isinstance(value, dict):
        raise InputError(f"top-level value in {path} must be an object")
    return value


def _int_op(op: str, lhs: Expr, rhs: Expr) -> Expr:
    return Expr(op, args=(lhs, rhs), sort=Sort.INT)


def _product(values: Sequence[Expr]) -> Expr:
    if not values:
        return int_const(1)
    result = values[0]
    for value in values[1:]:
        result = _int_op("imul", result, value)
    return result


def _shape(value: object, where: str) -> tuple[Expr, ...]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be a fixed-rank list")
    result = []
    for index, raw in enumerate(value):
        expression = parse_expr(raw, f"{where}[{index}]")
        if expression.sort != Sort.INT:
            raise InputError(f"{where}[{index}] must be an integer expression")
        if expression.op == "const_int" and int(expression.data) <= 0:
            raise InputError(f"{where}[{index}] must be positive")
        result.append(expression)
    return tuple(result)


def _indices_for(shape: Sequence[Expr], linear: Expr) -> tuple[Expr, ...]:
    indices = []
    for axis in range(len(shape)):
        tail = _product(shape[axis + 1 :])
        quotient = linear if tail == int_const(1) else _int_op("idiv", linear, tail)
        indices.append(_int_op("irem", quotient, shape[axis]))
    return tuple(indices)


def _row_major_offset(shape: Sequence[Expr], indices: Sequence[Expr]) -> Expr:
    if len(shape) != len(indices):
        raise ValueError("rank mismatch while constructing a Prims tensor offset")
    if not shape:
        return int_const(0)
    offset = indices[0]
    for dimension, index in zip(shape[1:], indices[1:]):
        offset = _int_op("iadd", _int_op("imul", offset, dimension), index)
    return offset


@dataclass(frozen=True)
class _Tensor:
    shape: tuple[Expr, ...]
    element: Callable[[tuple[Expr, ...]], Expr]


def _name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{where} must be a non-empty string")
    return value


def _argument_names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be a list of value names")
    return tuple(_name(item, f"{where}[{index}]") for index, item in enumerate(value))


def load_prims_program(path: Path) -> Program:
    """Lift a strict, functional Torch Prims graph into the common ETV IR."""

    path = Path(path).resolve()
    raw = _read_json(path)
    _only_keys(raw, {"format", "name", "inputs", "nodes", "output"}, str(path))
    if raw.get("format") != FORMAT_PRIMS:
        raise InputError(f"{path} must declare format {FORMAT_PRIMS!r}")
    name = _name(raw.get("name"), f"{path}.name")

    inputs = raw.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise InputError(f"{path}.inputs must be a non-empty list")
    values: dict[str, _Tensor] = {}
    for index, item in enumerate(inputs):
        where = f"{path}.inputs[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(item, {"name", "dtype", "shape"}, where)
        input_name = _name(item.get("name"), f"{where}.name")
        if input_name in values:
            raise InputError(f"duplicate Prims value name {input_name!r}")
        if item.get("dtype") != "float32":
            raise InputError(f"{where}.dtype must be 'float32'")
        input_shape = _shape(item.get("shape"), f"{where}.shape")

        def input_element(
            indices: tuple[Expr, ...],
            input_name: str = input_name,
            input_shape: tuple[Expr, ...] = input_shape,
        ) -> Expr:
            return Expr(
                "load",
                args=(
                    _row_major_offset(input_shape, indices),
                    bool_const(True),
                    float_const(Fraction(0)),
                ),
                data=input_name,
                sort=Sort.FLOAT,
            )

        values[input_name] = _Tensor(input_shape, input_element)

    nodes = raw.get("nodes")
    if not isinstance(nodes, list):
        raise InputError(f"{path}.nodes must be a list")
    binary = {
        "prims.add": "fadd",
        "prims.sub": "fsub",
        "prims.mul": "fmul",
        "prims.div": "fdiv",
    }
    unary = {
        "prims.neg": "fneg",
        "prims.sqrt": "fsqrt",
        "prims.rsqrt": "frsqrt",
    }
    for index, item in enumerate(nodes):
        where = f"{path}.nodes[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(item, {"name", "op", "args", "shape", "broadcast_dimensions"}, where)
        node_name = _name(item.get("name"), f"{where}.name")
        if node_name in values:
            raise InputError(f"duplicate Prims value name {node_name!r}")
        op = _name(item.get("op"), f"{where}.op")
        arguments = _argument_names(item.get("args"), f"{where}.args")
        missing = [argument for argument in arguments if argument not in values]
        if missing:
            raise InputError(f"{where} references unknown value(s): {', '.join(missing)}")

        if op == "prims.broadcast_in_dim":
            if len(arguments) != 1:
                raise InputError(f"{where} expects one input")
            result_shape = _shape(item.get("shape"), f"{where}.shape")
            dimensions = item.get("broadcast_dimensions")
            if not isinstance(dimensions, list) or any(
                not isinstance(axis, int) or isinstance(axis, bool) for axis in dimensions
            ):
                raise InputError(f"{where}.broadcast_dimensions must be an integer list")
            source = values[arguments[0]]
            if len(dimensions) != len(source.shape):
                raise InputError(
                    f"{where}.broadcast_dimensions length must equal the input rank"
                )
            if dimensions != sorted(set(dimensions)) or any(
                axis < 0 or axis >= len(result_shape) for axis in dimensions
            ):
                raise InputError(f"{where}.broadcast_dimensions must be sorted and in range")
            for input_axis, output_axis in enumerate(dimensions):
                if source.shape[input_axis] != result_shape[output_axis]:
                    raise InputError(
                        f"{where} changes a mapped dimension; explicit reshape is unsupported"
                    )

            def broadcast_element(
                indices: tuple[Expr, ...],
                source: _Tensor = source,
                dimensions: tuple[int, ...] = tuple(dimensions),
            ) -> Expr:
                return source.element(tuple(indices[axis] for axis in dimensions))

            values[node_name] = _Tensor(result_shape, broadcast_element)
            continue

        if "shape" in item or "broadcast_dimensions" in item:
            raise InputError(
                f"{where}.shape and .broadcast_dimensions are only valid for "
                "prims.broadcast_in_dim"
            )
        if op in binary:
            if len(arguments) != 2:
                raise InputError(f"{where} expects two inputs")
            lhs, rhs = (values[argument] for argument in arguments)
            if lhs.shape != rhs.shape:
                raise InputError(
                    f"{where} operands must have identical fixed-rank shapes; "
                    "use prims.broadcast_in_dim explicitly"
                )

            def binary_element(
                indices: tuple[Expr, ...],
                lhs: _Tensor = lhs,
                rhs: _Tensor = rhs,
                internal_op: str = binary[op],
            ) -> Expr:
                return Expr(
                    internal_op,
                    args=(lhs.element(indices), rhs.element(indices)),
                    sort=Sort.FLOAT,
                )

            values[node_name] = _Tensor(lhs.shape, binary_element)
            continue
        if op in unary:
            if len(arguments) != 1:
                raise InputError(f"{where} expects one input")
            source = values[arguments[0]]

            def unary_element(
                indices: tuple[Expr, ...],
                source: _Tensor = source,
                internal_op: str = unary[op],
            ) -> Expr:
                return Expr(internal_op, args=(source.element(indices),), sort=Sort.FLOAT)

            values[node_name] = _Tensor(source.shape, unary_element)
            continue
        raise InputError(f"unsupported Prims operation {op!r} in {where}", "PRIMS_OP_UNSUPPORTED")

    output = raw.get("output")
    if not isinstance(output, dict):
        raise InputError(f"{path}.output must be an object")
    _only_keys(output, {"value", "block"}, f"{path}.output")
    output_value = _name(output.get("value"), f"{path}.output.value")
    output_block = _name(output.get("block"), f"{path}.output.block")
    if output_value not in values:
        raise InputError(f"{path}.output.value references unknown value {output_value!r}")
    result = values[output_value]
    if not result.shape:
        raise InputError("scalar Prims outputs are outside the current memory contract")

    numel = _product(result.shape)
    linear = Expr("var", data="pid", sort=Sort.INT)
    indices = _indices_for(result.shape, linear)
    store = StoreTemplate(
        block=output_block,
        logical_index=linear,
        offset=linear,
        mask=bool_const(True),
        value=result.element(indices),
    )
    return Program(
        name=name,
        source=path,
        programs=numel,
        lanes=1,
        stores=(store,),
        frontend="torch_prims_json",
        frontend_version=PRIMS_FRONTEND_VERSION,
    )
