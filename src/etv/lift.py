"""Static tensor index maps to immutable scalar semantics."""

from dataclasses import dataclass
from fractions import Fraction
from math import prod

from .errors import InputError, ResourceLimit, UnsupportedSemantics
from .frontend import IRType, Operation, TTIRModule, operations
from .ir import BOOL, FLOAT, INDEX, Expr, Kernel, StorageRef, Store, Type, const, substitute, var
from .pairspec import LaunchSpec
from .semantics import CASTS, FLOAT_UNARY, validate_expr


@dataclass(frozen=True)
class Pointer:
    storage: StorageRef
    offset: Expr
    element: Type
    boolean_byte: bool = False


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
    "shrui": "shrui",
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
            tensor.term.storage,
            substitute(tensor.term.offset, replacements),
            tensor.term.element,
            tensor.term.boolean_byte,
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
    if "nan" in attr.values:
        raise UnsupportedSemantics("TTIR_NAN_CONSTANT_UNSUPPORTED", attr.assembly)
    if any(
        not isinstance(v, (int, Fraction)) and not (typ == FLOAT and v in ("+inf", "-inf"))
        for v in attr.values
    ):
        raise UnsupportedSemantics("TTIR_CONSTANT_UNSUPPORTED", attr.assembly)
    values = tuple(const(v, typ) for v in attr.values if isinstance(v, (int, Fraction, str)))
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


def _pointwise(
    op: Operation, args: tuple[Expr, ...], typ: Type, normalize_nonzero: bool = False
) -> Expr:
    # Exceptional-value assumptions such as ninf require a separate model.
    if any(attr.name == "fastmath" and op.integer("fastmath") != 0 for attr in op.attributes):
        raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "nonzero fastmath flags")
    name = op.name
    attrs: tuple[tuple[str, str | int], ...] = ()
    if name == "arith.cmpi":
        predicate = op.integer("predicate")
        names = {0: "eq", 1: "ne", 2: "lt", 3: "le", 4: "gt", 5: "ge"}
        if predicate in (6, 7, 8, 9):
            # Zero extend to an extra sign bit, then use signed comparison.
            bits = 1 if args[0].type == BOOL else args[0].type.bits
            if bits is None:
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "unsigned index comparison")
            args = tuple(Expr("extui", Type("int", bits + 1), (arg,)) for arg in args)
            operator = {6: "lt", 7: "le", 8: "gt", 9: "ge"}[predicate]
        else:
            operator = names[predicate]
        if predicate >= 2 and args[0].type == BOOL:
            args = tuple(
                Expr("extsi", Type("int", 2), (arg,), (("source_bits", 1), ("target_bits", 2)))
                for arg in args
            )
    elif name == "arith.cmpf":
        predicate = op.integer("predicate")
        if predicate in (0, 14, 7, 15):
            # There is no NaN in the supported extended-real domain, but the
            # operands can contain an unproved/invalid operation such as Inf-Inf.
            # Retain them so the definedness obligation cannot disappear.
            return Expr("ordered" if predicate in (7, 15) else "unordered", BOOL, args)
        names = {
            1: "eq",
            2: "gt",
            3: "ge",
            4: "lt",
            5: "le",
            6: "ne",
            8: "eq",
            9: "gt",
            10: "ge",
            11: "lt",
            12: "le",
            13: "ne",
        }
        operator = names[predicate]
    elif name in ("arith.minsi", "arith.maxsi"):
        # A signless i1 is interpreted as -1/0 by signed min/max.
        compared = (
            tuple(Expr("extsi", Type("int", 2), (arg,)) for arg in args) if typ == BOOL else args
        )
        return Expr(
            "select", typ, (Expr("lt" if name == "arith.minsi" else "gt", BOOL, compared), *args)
        )
    elif name == "tt.mulhiui":
        operator = "mulhiui"
    elif name == "tt.clampf":
        if op.integer("propagateNan") != 0:
            raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "clampf propagateNan")
        return Expr("fmin", typ, (Expr("fmax", typ, args[:2]), args[2]))
    elif name == "tt.extern_elementwise":
        symbol = op.string("symbol")
        symbol = symbol.removeprefix("__nv_")
        if symbol in ("isinff", "isinfd"):
            source = op.operands[0].type.element or op.operands[0].type if op.operands else None
            if not (
                normalize_nonzero
                and op.integer("pure") != 0
                and len(args) == 1
                and args[0].type == FLOAT
                and typ == Type("int", 32)
                and source is not None
                and source.kind == "float"
                and source.bits == (32 if symbol == "isinff" else 64)
            ):
                raise UnsupportedSemantics(
                    "TTIR_OP_UNSUPPORTED", "libdevice isinf requires exclusively zero/nonzero uses"
                )
            # Libdevice promises a nonzero integer for infinity, not exactly 1.
            # The caller has checked every SSA use is eq/ne against zero; only
            # under this condition is a normalized internal 0/1 representative
            # observationally exact. A raw integer result must never escape.
            is_infinite = Expr("eq", BOOL, (Expr("absf", FLOAT, args), const("+inf", FLOAT)))
            return Expr("select", typ, (is_infinite, const(1, typ), const(0, typ)))
        # CUDA's isnan returns an integer. Keep its operand's definedness even
        # though finite reals and the supported infinities are never NaN.
        if (
            symbol in ("isnan", "isnanf")
            and op.integer("pure") != 0
            and len(args) == 1
            and args[0].type == FLOAT
            and typ.kind == "int"
        ):
            return Expr("select", typ, (Expr("isnan", BOOL, args), const(1, typ), const(0, typ)))
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
            if attr.name == "isExact":
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "exact shift flag")
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


def _reduce(op: Operation, source: Tensor) -> Tensor:
    """Expand an associative, single-result static reduction as a balanced tree."""
    if len(op.regions) != 1 or len(op.regions[0]) != 1 or len(op.results) != 1:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "multi-result reduction")
    block = op.regions[0][0]
    if len(block.arguments) != 2 or len(block.operations) != 2:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "complex reduction combiner")
    combine, terminator = block.operations
    allowed = {
        "arith.addf",
        "arith.mulf",
        "arith.maxnumf",
        "arith.minnumf",
        "arith.addi",
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
    }
    if (
        combine.name not in allowed
        or combine.regions
        or len(combine.results) != 1
        or tuple(v.id for v in combine.operands) != tuple(v.id for v in block.arguments)
        or terminator.name != "tt.reduce.return"
        or tuple(v.id for v in terminator.operands) != (combine.results[0].id,)
    ):
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "non-associative reduction combiner")
    axis = op.integer("axis")
    if not 0 <= axis < len(source.shape):
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reduction axis")
    extent = source.shape[axis]
    if extent > 4096:
        raise ResourceLimit("static reduction extent")
    result_shape = source.shape[:axis] + source.shape[axis + 1 :]
    if shape(op.results[0].type) != result_shape:
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reduction result")
    tail = prod(source.shape[axis + 1 :])
    # A scalar result has one logical lane. Leaving a free __lane here would
    # let a later splat/reduction reindex the scalar's original input elements.
    base = (
        index_expr(
            "add",
            index_expr("mul", index_expr("div", var("__lane"), const(tail)), const(extent * tail)),
            index_expr("rem", var("__lane"), const(tail)),
        )
        if result_shape
        else const(0)
    )
    terms = [
        scalar(reindex(source, index_expr("add", base, const(i * tail)), result_shape))
        for i in range(extent)
    ]
    typ = scalar_type(op.results[0].type)
    while len(terms) > 1:
        terms = [
            _pointwise(combine, (terms[i], terms[i + 1]), typ) if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return Tensor(result_shape, terms[0])


def _nonzero_only_isinf_uses(block_operations: tuple[Operation, ...]) -> frozenset[int]:
    """Identify libdevice predicates whose integer return is observed only as truth."""
    nested = tuple(child for op in block_operations for child in operations(op))
    definitions = {value.id: op for op in nested for value in op.results}
    allowed: set[int] = set()
    for op in nested:
        if op.name != "tt.extern_elementwise" or len(op.results) != 1:
            continue
        if op.string("symbol").removeprefix("__nv_") not in ("isinff", "isinfd"):
            continue
        value_id = op.results[0].id
        users = [user for user in nested if any(v.id == value_id for v in user.operands)]
        if not users:
            continue
        for user in users:
            if (
                user.name != "arith.cmpi"
                or user.integer("predicate") not in (0, 1)
                or len(user.operands) != 2
            ):
                break
            other = next((v for v in user.operands if v.id != value_id), None)
            producer = definitions.get(other.id) if other is not None else None
            if producer is None or producer.name != "arith.constant":
                break
            attr = producer.attribute("value")
            if attr.kind not in ("integer", "dense") or not attr.values:
                break
            if not all(type(value) is int and value == 0 for value in attr.values):
                break
        else:
            allowed.add(op.ordinal)
    return frozenset(allowed)


def lift_kernel(module: TTIRModule, launch: LaunchSpec) -> Kernel:
    function = module.function
    if len(function.regions) != 1 or len(function.regions[0]) != 1:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", function.name)
    block = function.regions[0][0]
    for operation in block.operations:
        if operation.regions and operation.name != "tt.reduce":
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", operation.name)
    normalized_isinf = _nonzero_only_isinf_uses(block.operations)
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
            storage = StorageRef(launch.side, launch.id, name, ep.role, ep.offset, ep.index)
            values[value.id] = Tensor((), Pointer(storage, const(ep.offset), element))
        else:
            argument_type = scalar_type(value.type)
            if name in bindings:
                bound = bindings[name]
                if bound.type == FLOAT and argument_type == FLOAT:
                    term = bound
                elif bound.type == INDEX and argument_type.kind in ("int", "index"):
                    term = Expr("coordinate", argument_type, (bound,))
                else:
                    raise InputError("INVALID_PAIRSPEC", f"binding type disagrees with {name}")
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
        if op.name == "tt.reduce":
            if len(op.operands) != 1:
                raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "multi-operand reduction")
            values[op.results[0].id] = _reduce(op, values[op.operands[0].id])
            continue
        if op.regions:
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", op.name)
        if op.name == "gpu.barrier":
            if op.operands or op.results or op.attributes:
                raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "nonstandard gpu.barrier")
            if store_count:
                raise UnsupportedSemantics(
                    "GPU_BARRIER_MEMORY_DEPENDENCE_UNSUPPORTED", "gpu.barrier after a store"
                )
            # The accepted straight-line prefix has only pure expressions and
            # global-memory loads: all other effects (including atomics, async
            # operations and shared-memory operations) are unsupported. With no
            # prior writes to publish, this uniform barrier changes no value in
            # the launch's snapshot-read / simultaneous-write memory model.
            continue
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
            if not isinstance(pointer, Pointer):
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
            if pointer.boolean_byte:
                if value_expr.op == "extui" and value_expr.args[0].type == BOOL:
                    value_expr = value_expr.args[0]
                elif value_expr.op == "const" and value_expr.attr("value") in (0, 1):
                    value_expr = const(bool(value_expr.attr("value")), BOOL)
                else:
                    raise UnsupportedSemantics(
                        "TTIR_OP_UNSUPPORTED", "noncanonical boolean byte store"
                    )
            if value_expr.type != pointer.element:
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "store element type mismatch")
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
            not in (
                "tt.addptr",
                "tt.splat",
                "tt.broadcast",
                "tt.expand_dims",
                "tt.reshape",
                "tt.bitcast",
            )
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
            result = reindex(args[0], const(0), out_shape)
        elif op.name == "tt.broadcast":
            result = broadcast(args[0], out_shape)
        elif op.name in ("tt.expand_dims", "tt.reshape"):
            if prod(args[0].shape) != prod(out_shape):
                raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reshape size")
            result = Tensor(out_shape, args[0].term)
        elif op.name == "tt.gather":
            source, indices = args
            axis = op.integer("axis")
            if (
                not 0 <= axis < len(source.shape)
                or len(source.shape) != len(out_shape)
                or out_shape != indices.shape
                or isinstance(source.term, Pointer)
            ):
                raise UnsupportedSemantics(
                    "TTIR_SHAPE_UNSUPPORTED", "gather shape or pointer source"
                )
            offset = const(0)
            index = Expr("to_index", INDEX, (scalar(indices),))
            for dim, extent in enumerate(out_shape):
                coordinate = (
                    index
                    if dim == axis
                    else index_expr(
                        "rem",
                        index_expr("div", var("__lane"), const(prod(out_shape[dim + 1 :]))),
                        const(extent),
                    )
                )
                offset = index_expr(
                    "add",
                    offset,
                    index_expr("mul", coordinate, const(prod(source.shape[dim + 1 :]))),
                )
            selected_value = scalar(reindex(source, offset, out_shape))
            valid = Expr(
                "and",
                BOOL,
                (
                    Expr("ge", BOOL, (index, const(0))),
                    Expr("lt", BOOL, (index, const(source.shape[axis]))),
                ),
            )
            result = Tensor(
                out_shape,
                Expr(
                    "select",
                    selected_value.type,
                    (valid, selected_value, Expr("undefined", selected_value.type)),
                ),
            )
        elif op.name == "tt.bitcast":
            pointer = args[0].term
            target_type = op.results[0].type
            target = target_type.element if target_type.kind == "tensor" else target_type
            if (
                not isinstance(pointer, Pointer)
                or pointer.element != BOOL
                or target is None
                or target.kind != "pointer"
                or target.address_space != 1
                or target.element is None
                or scalar_type(target.element) != Type("int", 8)
            ):
                raise UnsupportedSemantics(
                    "TTIR_OP_UNSUPPORTED", "unsupported pointer/bit reinterpretation"
                )
            result = Tensor(out_shape, Pointer(pointer.storage, pointer.offset, BOOL, True))
        elif op.name == "tt.addptr":
            pointer = args[0].term
            if not isinstance(pointer, Pointer):
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "addptr base")
            delta = Expr("to_index", INDEX, (scalar(args[1]),))
            result = Tensor(
                out_shape,
                Pointer(
                    pointer.storage,
                    index_expr("add", pointer.offset, delta),
                    pointer.element,
                    pointer.boolean_byte,
                ),
            )
        elif op.name == "tt.load":
            pointer = args[0].term
            if not isinstance(pointer, Pointer):
                raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "load pointer")
            if pointer.boolean_byte and not launch.canonical_boolean_storage:
                raise UnsupportedSemantics(
                    "TTIR_OP_UNSUPPORTED", "boolean byte load requires canonical storage contract"
                )
            mask = scalar(args[1]) if len(args) > 1 else const(True, BOOL)
            default = scalar(args[2]) if len(args) > 2 else Expr("undefined", pointer.element)
            if pointer.boolean_byte:
                # The explicit representation predicate permits only 0/1 bytes.
                if default.op == "undefined":
                    default = Expr("undefined", BOOL)
                elif default.op == "const" and default.attr("value") in (0, 1):
                    default = const(bool(default.attr("value")), BOOL)
                else:
                    raise UnsupportedSemantics(
                        "TTIR_OP_UNSUPPORTED", "noncanonical boolean byte load default"
                    )
            term = Expr(
                "load",
                pointer.element,
                (pointer.offset, mask, default),
                (("storage", pointer.storage),),
            )
            effects.append(term)
            result = Tensor(
                out_shape, Expr("extui", Type("int", 8), (term,)) if pointer.boolean_byte else term
            )
        else:
            assert typ is not None
            result = Tensor(
                out_shape,
                _pointwise(op, tuple(scalar(a) for a in args), typ, op.ordinal in normalized_isinf),
            )
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
