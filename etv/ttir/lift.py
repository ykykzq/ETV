"""Sound, deliberately bounded lifting from parsed TTIR to ETV IR."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Tuple, Union

from ..casts import TTIR_INTEGER_CAST_OPS, integer_element_type
from ..ir import (
    Expr,
    Program,
    Sort,
    StoreTemplate,
    bool_const,
    float_const,
    int_const,
)
from ..model import InputError, UnsupportedSemantics
from .model import TTIRModule, TTIROperation

Index = Tuple[Expr, ...]


def _op(name: str, *args: Expr, sort: Sort) -> Expr:
    return Expr(name, args=tuple(args), sort=sort)


def _shape(type_text: str) -> Tuple[int, ...]:
    marker = "tensor<"
    start = type_text.find(marker)
    if start < 0:
        return ()
    content = type_text[start + len(marker) :]
    depth = 0
    end = None
    for index, char in enumerate(content):
        if char == "<":
            depth += 1
        elif char == ">":
            if depth == 0:
                end = index
                break
            depth -= 1
    if end is None:
        raise InputError(
            f"invalid tensor type returned by libtriton: {type_text}", "TTIR_TYPE_ERROR"
        )
    body = content[:end]
    depth = 0
    type_part = body
    for index, char in enumerate(body):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            type_part = body[:index]
            break
    match = re.match(r"^((?:\d+x)*)", type_part.strip())
    if match is None or not match.group(1):
        return ()
    return tuple(int(value) for value in match.group(1).rstrip("x").split("x"))


def _sort(type_text: str) -> Sort:
    without_ptrs = re.sub(r"!tt\.ptr<([^<>]+)>", r"\1", type_text)
    if re.search(r"(?:^|x)i1(?:[,>]|$)", without_ptrs):
        return Sort.BOOL
    if re.search(r"(?:^|x)(?:i\d+|index)(?:[,>]|$)", without_ptrs):
        return Sort.INT
    if re.search(r"(?:^|x)(?:f\d+|bf16)(?:[,>]|$)", without_ptrs):
        return Sort.FLOAT
    raise UnsupportedSemantics(
        f"TTIR type is outside the semantic subset: {type_text}",
        "TTIR_TYPE_UNSUPPORTED",
    )


def _is_pointer(type_text: str) -> bool:
    return "!tt.ptr<" in type_text


@dataclass(frozen=True)
class TensorValue:
    shape: Tuple[int, ...]
    sort: Sort
    element: Callable[[Index], Expr]


@dataclass(frozen=True)
class PointerValue:
    shape: Tuple[int, ...]
    block: str
    offset: Callable[[Index], Expr]


Value = Union[TensorValue, PointerValue]


def _indices_for(shape: Tuple[int, ...], flat: Expr) -> Index:
    result = []
    for index, dimension in enumerate(shape):
        stride = math.prod(shape[index + 1 :])
        value = (
            flat if stride == 1 else _op("idiv", flat, int_const(stride), sort=Sort.INT)
        )
        if index != 0 or math.prod(shape) != dimension:
            value = _op("irem", value, int_const(dimension), sort=Sort.INT)
        result.append(value)
    return tuple(result)


def _flat_index(shape: Tuple[int, ...], indices: Index) -> Expr:
    result = int_const(0)
    for index, value in enumerate(indices):
        stride = math.prod(shape[index + 1 :])
        term = (
            value
            if stride == 1
            else _op("imul", value, int_const(stride), sort=Sort.INT)
        )
        result = _op("iadd", result, term, sort=Sort.INT)
    return result


def _broadcast_indices(
    source: Tuple[int, ...], target: Tuple[int, ...], indices: Index
) -> Index:
    if not source:
        return ()
    if len(source) > len(target):
        raise UnsupportedSemantics(
            f"cannot broadcast rank-{len(source)} value to rank-{len(target)}",
            "TTIR_BROADCAST_UNSUPPORTED",
        )
    offset = len(target) - len(source)
    result = []
    for position, dimension in enumerate(source):
        target_dimension = target[offset + position]
        if dimension == 1:
            result.append(int_const(0))
        elif dimension == target_dimension:
            result.append(indices[offset + position])
        else:
            raise UnsupportedSemantics(
                f"incompatible TTIR broadcast from {source} to {target}",
                "TTIR_BROADCAST_UNSUPPORTED",
            )
    return tuple(result)


def _tensor_at(value: TensorValue, target: Tuple[int, ...], indices: Index) -> Expr:
    return value.element(_broadcast_indices(value.shape, target, indices))


def _pointer_at(value: PointerValue, target: Tuple[int, ...], indices: Index) -> Expr:
    return value.offset(_broadcast_indices(value.shape, target, indices))


def _require_tensor(value: Value, operation: str) -> TensorValue:
    if not isinstance(value, TensorValue):
        raise UnsupportedSemantics(
            f"{operation} requires a non-pointer value", "TTIR_TYPE_ERROR"
        )
    return value


def _require_pointer(value: Value, operation: str) -> PointerValue:
    if not isinstance(value, PointerValue):
        raise UnsupportedSemantics(
            f"{operation} requires a pointer value", "TTIR_TYPE_ERROR"
        )
    return value


def _result_shape(operation: TTIROperation) -> Tuple[int, ...]:
    if len(operation.results) != 1:
        raise UnsupportedSemantics(
            f"{operation.name} has {len(operation.results)} results; lifting requires one",
            "TTIR_MULTI_RESULT_UNSUPPORTED",
        )
    return _shape(operation.results[0].type)


def _attribute_int(operation: TTIROperation, name: str) -> int:
    value = operation.attributes.get(name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    assembly = operation.assembly or ""
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)", assembly)
    if match is None:
        raise InputError(
            f"libtriton output omitted {name!r} on {operation.name}",
            "TTIR_ATTRIBUTE_ERROR",
        )
    return int(match.group(1))


def _constant(operation: TTIROperation) -> TensorValue:
    result = operation.results[0]
    shape = _shape(result.type)
    sort = _sort(result.type)
    raw = operation.attributes.get("value")
    assembly = operation.assembly or ""
    if raw is None:
        match = re.search(r"\barith\.constant\s+(?:dense<([^>]+)>|([^\s:]+))", assembly)
        if match is None:
            raise UnsupportedSemantics(
                f"cannot read constant assembly: {assembly!r}",
                "TTIR_CONSTANT_UNSUPPORTED",
            )
        raw = match.group(1) if match.group(1) is not None else match.group(2)
    if isinstance(raw, str) and (raw.startswith("[") or raw.startswith("{")):
        raise UnsupportedSemantics(
            "non-splat dense constants are not lifted", "TTIR_CONSTANT_UNSUPPORTED"
        )
    if sort == Sort.BOOL:
        value = bool_const(raw is True or str(raw).lower() == "true" or str(raw) == "1")
    elif sort == Sort.INT:
        value = int_const(int(raw))
    else:
        try:
            value = float_const(Fraction(str(raw)))
        except (ValueError, ZeroDivisionError) as exc:
            raise UnsupportedSemantics(
                f"floating constant {raw!r} is not representable in ABSTRACT_FLOAT",
                "TTIR_CONSTANT_UNSUPPORTED",
            ) from exc
    return TensorValue(
        shape=shape, sort=sort, element=lambda _indices, value=value: value
    )


_INTEGER_BINARY = {
    "arith.addi": "iadd",
    "arith.subi": "isub",
    "arith.muli": "imul",
    "arith.divsi": "idiv",
    "arith.remsi": "irem",
}
_FLOAT_BINARY = {
    "arith.addf": "fadd",
    "arith.subf": "fsub",
    "arith.mulf": "fmul",
    "arith.divf": "fdiv",
}
_PREDICATES = {
    0: "eq",
    1: "ne",
    2: "lt",
    3: "le",
    4: "gt",
    5: "ge",
    "eq": "eq",
    "ne": "ne",
    "slt": "lt",
    "sle": "le",
    "sgt": "gt",
    "sge": "ge",
}


def _predicate(operation: TTIROperation) -> str:
    raw = operation.attributes.get("predicate")
    if raw in _PREDICATES:
        return _PREDICATES[raw]
    match = re.search(r"\barith\.cmpi\s+\"?([a-z]+)\"?\s*,", operation.assembly or "")
    if match is not None and match.group(1) in _PREDICATES:
        return _PREDICATES[match.group(1)]
    raise UnsupportedSemantics(
        "only signed integer comparisons are lifted", "TTIR_PREDICATE_UNSUPPORTED"
    )


def _function_operations(module: TTIRModule) -> Tuple[TTIROperation, ...]:
    functions = [op for op in module.operations if op.name == "tt.func"]
    selected = None
    for operation in functions:
        if operation.attributes.get("sym_name") == module.function:
            selected = operation
            break
    if selected is None:
        if len(functions) == 1:
            selected = functions[0]
        else:
            raise InputError(
                f"cannot locate function {module.function!r} in parsed operation stream"
            )
    # MLIR's walk callback used by libtriton is post-order: a function's body
    # appears immediately before its tt.func operation.
    start = 0
    for operation in functions:
        if operation.index >= selected.index:
            break
        start = operation.index + 1
    return tuple(op for op in module.operations if start <= op.index < selected.index)


def lift_ttir(
    module: TTIRModule,
    programs: Expr,
    *,
    store_index: int | None = None,
) -> Program:
    """Lift the acyclic pointwise subset and reject every unsupported semantic case."""

    function_operations = _function_operations(module)
    region_operation = next(
        (operation for operation in function_operations if operation.regions), None
    )
    if region_operation is not None:
        raise UnsupportedSemantics(
            f"region operation {region_operation.name} is parsed but not yet lifted",
            "TTIR_REGION_SEMANTICS_UNSUPPORTED",
        )
    if any(
        ", #" in value.type
        for operation in function_operations
        for value in operation.operands + operation.results
    ):
        raise UnsupportedSemantics(
            "TTGIR tensor encodings are parsed but not part of the TTIR lifting model",
            "TTGIR_LAYOUT_UNSUPPORTED",
        )

    values: Dict[int, Value] = {}
    for argument in module.arguments:
        value = argument.value
        name = f"arg{argument.index}"
        if _is_pointer(value.type):
            if "!tt.ptr<tensor<" in value.type:
                raise UnsupportedSemantics(
                    "tensor/block-pointer arguments are parsed but not lifted",
                    "TTIR_BLOCK_POINTER_UNSUPPORTED",
                )
            values[value.id] = PointerValue((), name, lambda _indices: int_const(0))
        else:
            if _shape(value.type):
                raise UnsupportedSemantics(
                    "tensor-valued function arguments are parsed but not lifted",
                    "TTIR_TENSOR_ARGUMENT_UNSUPPORTED",
                )
            sort = _sort(value.type)
            if sort == Sort.BOOL:
                raise UnsupportedSemantics(
                    "boolean function arguments are not represented by PairSpec v1",
                    "TTIR_BOOL_ARGUMENT_UNSUPPORTED",
                )
            expression = (
                Expr("scalar", data=name, sort=Sort.FLOAT)
                if sort == Sort.FLOAT
                else Expr("var", data=name, sort=sort)
            )
            values[value.id] = TensorValue(
                (), sort, lambda _indices, expression=expression: expression
            )

    stores: List[Tuple[TTIROperation, PointerValue, TensorValue, TensorValue]] = []
    masked_loads_without_other: List[int] = []

    def operand(operation: TTIROperation, index: int) -> Value:
        try:
            return values[operation.operands[index].id]
        except (IndexError, KeyError) as exc:
            raise InputError(
                f"missing SSA operand {index} while lifting {operation.name}",
                "TTIR_SSA_ERROR",
            ) from exc

    for operation in function_operations:
        name = operation.name
        if name == "tt.return":
            continue
        if name == "arith.constant":
            result: Value = _constant(operation)
        elif name == "tt.get_program_id":
            assembly = operation.assembly or ""
            axis = operation.attributes.get("axis")
            if axis not in {None, 0} or (axis is None and " x " not in f" {assembly} "):
                raise UnsupportedSemantics(
                    "only program-id axis x is lifted", "TTIR_GRID_UNSUPPORTED"
                )
            result = TensorValue(
                (), Sort.INT, lambda _indices: Expr("var", data="pid", sort=Sort.INT)
            )
        elif name == "tt.make_range":
            start = _attribute_int(operation, "start")
            end = _attribute_int(operation, "end")
            shape = _result_shape(operation)
            if len(shape) != 1 or shape[0] != end - start:
                raise InputError(
                    "tt.make_range type does not match start/end", "TTIR_VERIFY_ERROR"
                )
            result = TensorValue(
                shape,
                Sort.INT,
                lambda indices, start=start: _op(
                    "iadd", int_const(start), indices[0], sort=Sort.INT
                ),
            )
        elif name in {"tt.splat", "tt.broadcast"}:
            source = operand(operation, 0)
            shape = _result_shape(operation)
            if isinstance(source, PointerValue):
                result = PointerValue(
                    shape,
                    source.block,
                    lambda indices, source=source, shape=shape: _pointer_at(
                        source, shape, indices
                    ),
                )
            else:
                result = TensorValue(
                    shape,
                    source.sort,
                    lambda indices, source=source, shape=shape: _tensor_at(
                        source, shape, indices
                    ),
                )
        elif name == "tt.expand_dims":
            source = operand(operation, 0)
            shape = _result_shape(operation)
            axis = _attribute_int(operation, "axis")
            if axis < 0 or axis >= len(shape) or shape[axis] != 1:
                raise InputError(
                    "invalid tt.expand_dims result shape", "TTIR_VERIFY_ERROR"
                )

            def without_axis(indices: Index, axis: int = axis) -> Index:
                return indices[:axis] + indices[axis + 1 :]

            if isinstance(source, PointerValue):
                result = PointerValue(
                    shape,
                    source.block,
                    lambda indices, source=source: source.offset(without_axis(indices)),
                )
            else:
                result = TensorValue(
                    shape,
                    source.sort,
                    lambda indices, source=source: source.element(
                        without_axis(indices)
                    ),
                )
        elif name == "tt.reshape":
            source = operand(operation, 0)
            shape = _result_shape(operation)
            if math.prod(source.shape) != math.prod(shape):
                raise InputError(
                    "tt.reshape changes element count", "TTIR_VERIFY_ERROR"
                )

            def source_indices(
                indices: Index, source_shape=source.shape, shape=shape
            ) -> Index:
                return _indices_for(source_shape, _flat_index(shape, indices))

            if isinstance(source, PointerValue):
                result = PointerValue(
                    shape,
                    source.block,
                    lambda indices, source=source: source.offset(
                        source_indices(indices)
                    ),
                )
            else:
                result = TensorValue(
                    shape,
                    source.sort,
                    lambda indices, source=source: source.element(
                        source_indices(indices)
                    ),
                )
        elif (
            name in _INTEGER_BINARY
            or name in _FLOAT_BINARY
            or name in {"arith.andi", "arith.ori"}
        ):
            lhs = _require_tensor(operand(operation, 0), name)
            rhs = _require_tensor(operand(operation, 1), name)
            shape = _result_shape(operation)
            if name in _INTEGER_BINARY:
                semantic_name, sort = _INTEGER_BINARY[name], Sort.INT
            elif name in _FLOAT_BINARY:
                semantic_name, sort = _FLOAT_BINARY[name], Sort.FLOAT
            else:
                semantic_name, sort = (
                    "and" if name == "arith.andi" else "or"
                ), Sort.BOOL
                if lhs.sort != Sort.BOOL or rhs.sort != Sort.BOOL:
                    raise UnsupportedSemantics(
                        f"integer bitwise {name} is not lifted", "TTIR_OP_UNSUPPORTED"
                    )
            result = TensorValue(
                shape,
                sort,
                lambda indices, lhs=lhs, rhs=rhs, shape=shape, semantic_name=semantic_name, sort=sort: _op(
                    semantic_name,
                    _tensor_at(lhs, shape, indices),
                    _tensor_at(rhs, shape, indices),
                    sort=sort,
                ),
            )
        elif name == "arith.cmpi":
            lhs = _require_tensor(operand(operation, 0), name)
            rhs = _require_tensor(operand(operation, 1), name)
            shape = _result_shape(operation)
            predicate = _predicate(operation)
            result = TensorValue(
                shape,
                Sort.BOOL,
                lambda indices, lhs=lhs, rhs=rhs, shape=shape, predicate=predicate: _op(
                    predicate,
                    _tensor_at(lhs, shape, indices),
                    _tensor_at(rhs, shape, indices),
                    sort=Sort.BOOL,
                ),
            )
        elif name == "arith.select":
            condition = _require_tensor(operand(operation, 0), name)
            true_value = _require_tensor(operand(operation, 1), name)
            false_value = _require_tensor(operand(operation, 2), name)
            shape = _result_shape(operation)
            result = TensorValue(
                shape,
                true_value.sort,
                lambda indices, condition=condition, true_value=true_value, false_value=false_value, shape=shape: _op(
                    "select",
                    _tensor_at(condition, shape, indices),
                    _tensor_at(true_value, shape, indices),
                    _tensor_at(false_value, shape, indices),
                    sort=true_value.sort,
                ),
            )
        elif name in TTIR_INTEGER_CAST_OPS:
            source = _require_tensor(operand(operation, 0), name)
            shape = _result_shape(operation)
            try:
                cast_data = (
                    integer_element_type(operation.operands[0].type),
                    integer_element_type(operation.results[0].type),
                )
            except ValueError as exc:
                raise UnsupportedSemantics(
                    f"cannot preserve integer cast types for {name}: {exc}",
                    "TTIR_CAST_TYPE_UNSUPPORTED",
                ) from exc
            semantic_name = TTIR_INTEGER_CAST_OPS[name]
            result_sort = _sort(operation.results[0].type)
            result = TensorValue(
                shape,
                result_sort,
                lambda indices, source=source, shape=shape, semantic_name=semantic_name, result_sort=result_sort, cast_data=cast_data: Expr(
                    semantic_name,
                    args=(_tensor_at(source, shape, indices),),
                    data=cast_data,
                    sort=result_sort,
                ),
            )
        elif name in {"arith.extf", "arith.truncf"}:
            source = _require_tensor(operand(operation, 0), name)
            result = TensorValue(
                _result_shape(operation),
                _sort(operation.results[0].type),
                source.element,
            )
        elif name in {"arith.negf", "math.sqrt", "math.rsqrt"}:
            source = _require_tensor(operand(operation, 0), name)
            shape = _result_shape(operation)
            semantic_name = {
                "arith.negf": "fneg",
                "math.sqrt": "fsqrt",
                "math.rsqrt": "frsqrt",
            }[name]
            result = TensorValue(
                shape,
                Sort.FLOAT,
                lambda indices, source=source, shape=shape, semantic_name=semantic_name: _op(
                    semantic_name, _tensor_at(source, shape, indices), sort=Sort.FLOAT
                ),
            )
        elif name == "math.fma":
            operands = tuple(
                _require_tensor(operand(operation, index), name) for index in range(3)
            )
            shape = _result_shape(operation)
            result = TensorValue(
                shape,
                Sort.FLOAT,
                lambda indices, operands=operands, shape=shape: _op(
                    "fma",
                    *(_tensor_at(value, shape, indices) for value in operands),
                    sort=Sort.FLOAT,
                ),
            )
        elif name == "tt.addptr":
            pointer = _require_pointer(operand(operation, 0), name)
            offset = _require_tensor(operand(operation, 1), name)
            shape = _result_shape(operation)
            result = PointerValue(
                shape,
                pointer.block,
                lambda indices, pointer=pointer, offset=offset, shape=shape: _op(
                    "iadd",
                    _pointer_at(pointer, shape, indices),
                    _tensor_at(offset, shape, indices),
                    sort=Sort.INT,
                ),
            )
        elif name == "tt.load":
            pointer = _require_pointer(operand(operation, 0), name)
            shape = _result_shape(operation)
            if _sort(operation.results[0].type) != Sort.FLOAT:
                raise UnsupportedSemantics(
                    "only abstract-floating TTIR memory loads are lifted",
                    "TTIR_MEMORY_TYPE_UNSUPPORTED",
                )
            mask = (
                _require_tensor(operand(operation, 1), name)
                if len(operation.operands) >= 2
                else TensorValue((), Sort.BOOL, lambda _indices: bool_const(True))
            )
            if len(operation.operands) >= 3:
                other = _require_tensor(operand(operation, 2), name)
            else:
                other = TensorValue(
                    (), Sort.FLOAT, lambda _indices: float_const(Fraction(0))
                )
                if len(operation.operands) == 2:
                    masked_loads_without_other.append(operation.operands[1].id)
            result = TensorValue(
                shape,
                Sort.FLOAT,
                lambda indices, pointer=pointer, mask=mask, other=other, shape=shape: Expr(
                    "load",
                    args=(
                        _pointer_at(pointer, shape, indices),
                        _tensor_at(mask, shape, indices),
                        _tensor_at(other, shape, indices),
                    ),
                    data=pointer.block,
                    sort=Sort.FLOAT,
                ),
            )
        elif name == "tt.store":
            pointer = _require_pointer(operand(operation, 0), name)
            value = _require_tensor(operand(operation, 1), name)
            if value.sort != Sort.FLOAT:
                raise UnsupportedSemantics(
                    "only abstract-floating TTIR memory stores are lifted",
                    "TTIR_MEMORY_TYPE_UNSUPPORTED",
                )
            mask = (
                _require_tensor(operand(operation, 2), name)
                if len(operation.operands) >= 3
                else TensorValue((), Sort.BOOL, lambda _indices: bool_const(True))
            )
            stores.append((operation, pointer, value, mask))
            continue
        else:
            raise UnsupportedSemantics(
                f"operation {name} is valid TTIR but has no ETV IR lifting rule",
                "TTIR_OP_UNSUPPORTED",
            )

        if len(operation.results) != 1:
            raise UnsupportedSemantics(
                f"cannot bind results of {name}", "TTIR_MULTI_RESULT_UNSUPPORTED"
            )
        values[operation.results[0].id] = result

    if not stores:
        raise UnsupportedSemantics("TTIR function has no tt.store", "TTIR_NO_STORE")
    if store_index is None and len(stores) != 1:
        raise UnsupportedSemantics(
            "TTIR lifting currently requires exactly one tt.store",
            "MULTIPLE_STORES_UNSUPPORTED",
        )
    selected_store = 0 if store_index is None else store_index
    if selected_store >= len(stores):
        raise UnsupportedSemantics(
            f"requested tt.store index {selected_store}, but function has {len(stores)} stores",
            "TTIR_STORE_INDEX_OUT_OF_RANGE",
        )
    store_operation, pointer, value, mask = stores[selected_store]
    store_mask_id = (
        store_operation.operands[2].id if len(store_operation.operands) >= 3 else None
    )
    if any(mask_id != store_mask_id for mask_id in masked_loads_without_other):
        raise UnsupportedSemantics(
            "a masked tt.load without 'other' is not guarded by the identical store mask",
            "TTIR_UNDEFINED_LOAD_LANE",
        )
    lanes = math.prod(pointer.shape) if pointer.shape else 1
    lane = Expr("var", data="lane", sort=Sort.INT)
    indices = _indices_for(pointer.shape, lane)
    logical_index = _op(
        "iadd",
        _op(
            "imul",
            Expr("var", data="pid", sort=Sort.INT),
            int_const(lanes),
            sort=Sort.INT,
        ),
        lane,
        sort=Sort.INT,
    )
    store = StoreTemplate(
        block=pointer.block,
        logical_index=logical_index,
        offset=pointer.offset(indices),
        mask=_tensor_at(mask, pointer.shape, indices),
        value=_tensor_at(value, pointer.shape, indices),
    )
    return Program(
        name=module.function,
        source=module.source,
        programs=programs,
        lanes=lanes,
        stores=(store,),
        frontend="libtriton",
        frontend_version=module.parser_version,
    )
