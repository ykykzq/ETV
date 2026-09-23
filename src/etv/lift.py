"""Static tensor index maps to immutable scalar semantics."""

from dataclasses import dataclass
from fractions import Fraction
from math import prod

from .errors import InputError, ResourceLimit, UnsupportedSemantics
from .frontend import IRType, Operation, TTIRModule
from .ir import BOOL, FLOAT, INDEX, Expr, Kernel, StorageRef, Store, Type, const, substitute, var
from .pairspec import LaunchSpec
from .semantics import CASTS, FLOAT_UNARY, validate_expr


@dataclass(frozen=True)
class Pointer:
    storage: StorageRef
    offset: Expr
    element: Type


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    term: Expr | Pointer


ARITH = {
    "addi": "add",
    "subi": "sub",
    "muli": "mul",
    "divsi": "div",
    "remsi": "rem",
    "andi": "and",
    "ori": "or",
    "xori": "xor",
    "addf": "fadd",
    "subf": "fsub",
    "mulf": "fmul",
    "divf": "fdiv",
    "negf": "fneg",
    "maxnumf": "fmax",
    "minnumf": "fmin",
    "select": "select",
    **{name: name for name in CASTS},
}
EXTERN = {
    "acosh": "acosh",
    "acoshf": "acosh",
    "atan": "atan",
    "atanf": "atan",
    "ceilf": "ceil",
    "coshf": "cosh",
    "erff": "erf",
    "expm1f": "expm1",
    "floorf": "floor",
    "nearbyintf": "nearbyint",
    "powf": "pow",
    "rsqrtf": "rsqrt",
    "sqrtf": "sqrt",
    "tanhf": "tanh",
}


def scalar_type(typ: IRType) -> Type:
    if typ.kind == "tensor" and typ.element is not None:
        return scalar_type(typ.element)
    if typ.kind == "float":
        return FLOAT
    if typ.kind == "int":
        if typ.signedness:
            raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "expected signless integer")
        return BOOL if typ.bits == 1 else Type("int", typ.bits)
    if typ.kind == "index":
        return INDEX
    raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", typ.assembly)


def shape(typ: IRType) -> tuple[int, ...]:
    if typ.kind != "tensor":
        return ()
    if typ.encoding or any(n < 1 for n in typ.shape):
        raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", typ.assembly)
    return typ.shape


def scalar(tensor: Tensor) -> Expr:
    if isinstance(tensor.term, Pointer):
        raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "pointer used as scalar")
    return tensor.term


def index_expr(op: str, a: Expr, b: Expr) -> Expr:
    return Expr(op, INDEX, (a, b))


def reindex(tensor: Tensor, index: Expr, new_shape: tuple[int, ...]) -> Tensor:
    replacements = (("__lane", index),)
    if isinstance(tensor.term, Pointer):
        term: Expr | Pointer = Pointer(
            tensor.term.storage, substitute(tensor.term.offset, replacements), tensor.term.element
        )
    else:
        term = substitute(tensor.term, replacements)
    return Tensor(new_shape, term)


def broadcast(tensor: Tensor, new_shape: tuple[int, ...]) -> Tensor:
    if not tensor.shape:
        return Tensor(new_shape, tensor.term)
    if len(tensor.shape) != len(new_shape):
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "broadcast rank")
    source_index = const(0)
    for axis, (old, new) in enumerate(zip(tensor.shape, new_shape, strict=True)):
        if old not in (1, new):
            raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "broadcast extent")
        coord = index_expr(
            "rem", index_expr("div", var("__lane"), const(prod(new_shape[axis + 1 :]))), const(new)
        )
        if old != 1:
            source_index = index_expr(
                "add", source_index, index_expr("mul", coord, const(prod(tensor.shape[axis + 1 :])))
            )
    return reindex(tensor, source_index, new_shape)


def _constant(op: Operation, typ: Type, result_shape: tuple[int, ...]) -> Expr:
    attr = op.attribute("value")
    if attr.kind not in ("dense", "integer", "float") or not attr.values or None in attr.values:
        raise UnsupportedSemantics("TTIR_CONSTANT_UNSUPPORTED", attr.assembly)
    if any(not isinstance(v, (int, Fraction)) for v in attr.values):
        raise UnsupportedSemantics("TTIR_CONSTANT_UNSUPPORTED", attr.assembly)
    values = tuple(const(v, typ) for v in attr.values if isinstance(v, (int, Fraction)))
    if attr.kind != "dense" or attr.splat:
        return values[0]
    if len(values) != prod(result_shape):
        raise UnsupportedSemantics("TTIR_CONSTANT_UNSUPPORTED", "dense shape mismatch")
    result = values[-1]
    for i in reversed(range(len(values) - 1)):
        result = Expr(
            "select", typ, (Expr("eq", BOOL, (var("__lane"), const(i))), values[i], result)
        )
    return result


def _pointwise(op: Operation, args: tuple[Expr, ...], typ: Type) -> Expr:
    name = op.name
    attrs: tuple[tuple[str, str | int], ...] = ()
    if name == "arith.cmpi":
        predicate = op.integer("predicate")
        names = {0: "eq", 1: "ne", 2: "lt", 3: "le", 4: "gt", 5: "ge"}
        if predicate not in names:
            raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "unsigned cmpi")
        operator = names[predicate]
        if predicate >= 2 and args[0].type == BOOL:
            args = tuple(
                Expr("extsi", Type("int", 2), (arg,), (("source_bits", 1), ("target_bits", 2)))
                for arg in args
            )
    elif name == "arith.cmpf":
        predicate = op.integer("predicate")
        if predicate in (0, 8, 7, 15):
            return const(predicate in (7, 15), BOOL)
        names = {
            1: "eq",
            2: "gt",
            3: "ge",
            4: "lt",
            5: "le",
            6: "ne",
            9: "eq",
            10: "gt",
            11: "ge",
            12: "lt",
            13: "le",
            14: "ne",
        }
        operator = names[predicate]
    elif name == "tt.clampf":
        if op.integer("propagateNan") != 0:
            raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "clampf propagateNan")
        return Expr("fmin", typ, (Expr("fmax", typ, args[:2]), args[2]))
    elif name == "tt.extern_elementwise":
        symbol = op.string("symbol")
        symbol = symbol.removeprefix("__nv_")
        if op.integer("pure") == 0 or symbol not in EXTERN:
            raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", name + ":" + symbol)
        operator = EXTERN[symbol]
    elif name.startswith("arith.") and name[6:] in ARITH:
        operator = ARITH[name[6:]]
        if operator in CASTS:
            source = op.operands[0].type.element or op.operands[0].type
            target = op.results[0].type.element or op.results[0].type
            attrs = tuple(
                sorted((("source_bits", source.bits or 0), ("target_bits", target.bits or 0)))
            )
        for attr in op.attributes:
            if attr.name == "overflowFlags" and op.integer(attr.name) != 0:
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "integer overflow flags")
    elif name.startswith("math.") and name[5:] in FLOAT_UNARY | {"fma"}:
        operator = name[5:]
    else:
        raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", name)
    result = Expr(operator, typ, args, attrs)
    try:
        validate_expr(result)
    except ValueError as exc:
        raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", str(exc)) from exc
    return result


def lift_kernel(module: TTIRModule, launch: LaunchSpec) -> Kernel:
    function = module.function
    if len(function.regions) != 1 or len(function.regions[0]) != 1:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", function.name)
    block = function.regions[0][0]
    for operation in block.operations:
        if operation.regions:
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", operation.name)
    values: dict[str, Tensor] = {}
    endpoints = {ep.name: ep for ep in launch.abi}
    bindings = dict(launch.bindings)
    physical_names = {f"arg{i}" for i in range(len(block.arguments))}
    if (
        not endpoints.keys() <= physical_names
        or not {k for k in bindings if k.startswith("arg")} <= physical_names
    ):
        raise InputError("ABI_ROLE_MISSING", launch.id)
    parameters: list[Expr] = []
    for i, value in enumerate(block.arguments):
        name = f"arg{i}"
        ep = endpoints.get(name)
        if value.type.kind == "pointer":
            if (
                ep is None
                or ep.kind == "scalar"
                or value.type.address_space != 1
                or value.type.element is None
            ):
                raise InputError("ABI_ROLE_MISSING", f"{launch.id}:{name}")
            element = scalar_type(value.type.element)
            if element != FLOAT:
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "nonfloat memory")
            storage = StorageRef(launch.side, launch.id, name, ep.role, ep.offset, ep.index)
            values[value.id] = Tensor((), Pointer(storage, const(ep.offset), element))
        else:
            argument_type = scalar_type(value.type)
            if name in bindings:
                term = Expr("coordinate", argument_type, (bindings[name],))
            elif ep is not None and ep.kind == "scalar":
                term = var("scalar." + ep.role, argument_type)
            else:
                raise InputError("ABI_ROLE_MISSING", f"{launch.id}:{name}")
            parameters.append(term)
            values[value.id] = Tensor((), term)
    stores: list[Store] = []
    effects: list[Expr] = []
    store_count = 0
    selected = {s.index: s.role for s in launch.stores}
    for op in block.operations:
        if op.regions:
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", op.name)
        if op.name == "tt.return":
            if op.operands:
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "return value")
            continue
        args = tuple(values[v.id] for v in op.operands)
        out_shape = shape(op.results[0].type) if op.results else args[0].shape if args else ()
        if prod(out_shape) > launch.max_instances:
            raise ResourceLimit("static tensor shape")
        if op.name == "tt.store":
            pointer = args[0].term
            if not isinstance(pointer, Pointer) or pointer.element != FLOAT:
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "store pointer")
            if store_count not in selected:
                raise UnsupportedSemantics("UNSELECTED_STORE", "all effects must be declared")
            if (
                selected[store_count] != pointer.storage.logical_role
                or pointer.storage.scalar_index is not None
            ):
                raise InputError("ABI_ROLE_MISSING", "store role disagrees with pointer")
            mask = scalar(args[2]) if len(args) == 3 else const(True, BOOL)
            value_expr = scalar(args[1])
            if value_expr.type != FLOAT:
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "nonfloat store")
            stores.append(
                Store(
                    store_count,
                    pointer.storage,
                    pointer.offset,
                    pointer.offset,
                    mask,
                    value_expr,
                    args[0].shape,
                )
            )
            store_count += 1
            continue
        if len(op.results) != 1:
            raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", op.name)
        typ = (
            scalar_type(op.results[0].type)
            if op.name
            not in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape")
            else None
        )
        if op.name == "arith.constant":
            assert typ is not None
            result = Tensor(out_shape, _constant(op, typ, out_shape))
        elif op.name == "tt.get_program_id":
            if op.integer("axis") != 0:
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "multidimensional program grid")
            assert typ is not None
            result = Tensor((), Expr("coordinate", typ, (var("__pid"),)))
        elif op.name == "tt.make_range":
            assert typ is not None
            if out_shape != (op.integer("end") - op.integer("start"),):
                raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "range shape")
            result = Tensor(
                out_shape,
                Expr(
                    "coordinate",
                    typ,
                    (index_expr("add", var("__lane"), const(op.integer("start"))),),
                ),
            )
        elif op.name == "tt.splat":
            if args[0].shape:
                raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "splat source")
            result = Tensor(out_shape, args[0].term)
        elif op.name == "tt.broadcast":
            result = broadcast(args[0], out_shape)
        elif op.name in ("tt.expand_dims", "tt.reshape"):
            if prod(args[0].shape) != prod(out_shape):
                raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reshape size")
            result = Tensor(out_shape, args[0].term)
        elif op.name == "tt.addptr":
            pointer = args[0].term
            if not isinstance(pointer, Pointer):
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "addptr base")
            delta = Expr("to_index", INDEX, (scalar(args[1]),))
            result = Tensor(
                out_shape,
                Pointer(pointer.storage, index_expr("add", pointer.offset, delta), pointer.element),
            )
        elif op.name == "tt.load":
            pointer = args[0].term
            if not isinstance(pointer, Pointer) or pointer.element != FLOAT:
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "load pointer")
            mask = scalar(args[1]) if len(args) > 1 else const(True, BOOL)
            default = scalar(args[2]) if len(args) > 2 else Expr("undefined", FLOAT)
            term = Expr(
                "load", FLOAT, (pointer.offset, mask, default), (("storage", pointer.storage),)
            )
            effects.append(term)
            result = Tensor(out_shape, term)
        else:
            assert typ is not None
            result = Tensor(out_shape, _pointwise(op, tuple(scalar(a) for a in args), typ))
            if op.name in ("arith.index_cast", "arith.index_castui") and launch.index_bits is None:
                raise UnsupportedSemantics("INDEX_WIDTH_REQUIRED", op.name)
        values[op.results[0].id] = result
    if set(selected) != set(range(store_count)):
        raise InputError("ABI_ROLE_MISSING", "selected store does not exist")
    lane_shape = max((s.lane_shape for s in stores), key=prod, default=())
    return Kernel(
        launch.id,
        launch.function,
        launch.programs,
        lane_shape,
        tuple(stores),
        module.source_hash,
        tuple(parameters),
        tuple(effects),
    )
