"""Static tensor index maps to immutable scalar semantics."""

from dataclasses import dataclass
from fractions import Fraction
from math import prod

from .errors import InputError, ResourceLimit, UnsupportedSemantics
from .frontend import Block, IRType, Operation, TTIRModule, operations
from .ir import BOOL, FLOAT, INDEX, Expr, Kernel, MemoryRead, StorageRef, Store, Type
from .ir import const, substitute, var
from .pairspec import LaunchSpec
from .semantics import CASTS, FLOAT_UNARY, evaluate, simplify, validate_expr, width


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
    "exp": "exp",
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
    if any(
        not isinstance(v, (int, Fraction)) and not (typ == FLOAT and v in ("+inf", "-inf", "nan"))
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
            name = {0: "float_false", 7: "ordered", 14: "unordered", 15: "float_true"}[predicate]
            return Expr(name, BOOL, args)
        comparison = Expr(("eq", "gt", "ge", "lt", "le", "ne")[(predicate - 1) % 7], BOOL, args)
        if predicate >= 8:
            return Expr("or", BOOL, (Expr("unordered", BOOL, args), comparison))
        return comparison
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
            is_infinite = Expr("isinf", BOOL, args)
            return Expr("select", typ, (is_infinite, const(1, typ), const(0, typ)))
        # Keep the operand's definedness when lowering the integer predicate.
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
    if operator == "acosh":
        x = args[0]
        one = const(1, FLOAT)
        below = Expr("lt", BOOL, (x, one))
        equal = Expr("eq", BOOL, (x, one))
        radicand = Expr(
            "fmul", FLOAT, (Expr("fsub", FLOAT, (x, one)), Expr("fadd", FLOAT, (x, one)))
        )
        total = Expr("fadd", FLOAT, (x, Expr("sqrt", FLOAT, (radicand,))))
        real = Expr("select", FLOAT, (equal, const(0, FLOAT), Expr("log", FLOAT, (total,))))
        exceptional = Expr("or", BOOL, (Expr("isnan", BOOL, (x,)), below))
        return Expr("select", FLOAT, (exceptional, const("nan", FLOAT), real))
    return result


def _non_nan_reduction_operand(value: Expr) -> Expr:
    """Check that a comparator's total-order premise holds at every input lane."""
    ordered = Expr("ordered", BOOL, (value, value))
    return Expr("select", FLOAT, (ordered, value, Expr("undefined", FLOAT)))


def _comparison_reduction(block: Block) -> str | None:
    if len(block.operations) not in (3, 5):
        return None
    compare, *guards, choose, terminator = block.operations
    if (
        compare.name != "arith.cmpf"
        or compare.operands != block.arguments
        or choose.name != "arith.select"
        or choose.operands[1:] != block.arguments
        or terminator.name != "tt.reduce.return"
        or terminator.operands != choose.results
    ):
        return None
    condition = compare.results
    if guards:
        isnan, combine = guards
        lhs = block.arguments[0]
        if (
            isnan.name != "arith.cmpf"
            or isnan.integer("predicate") != 13  # UNE(x, x) detects NaN.
            or isnan.operands != (lhs, lhs)
            or combine.name != "arith.ori"
            or combine.operands not in (condition + isnan.results, isnan.results + condition)
        ):
            return None
        condition = combine.results
    if choose.operands[:1] != condition:
        return None
    predicate = compare.integer("predicate")
    if predicate in (2, 3, 9, 10):
        return "fmax"
    if predicate in (4, 5, 11, 12):
        return "fmin"
    return None


def _arg_reduction(op: Operation, sources: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    failure = "expected lexicographic value/index reduction"
    if len(sources) != 2 or len(op.results) != 2 or len(op.regions) != 1 or len(op.regions[0]) != 1:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", failure)
    block = op.regions[0][0]
    if len(block.arguments) != 4 or tuple(item.name for item in block.operations) != (
        "arith.cmpf",
        "arith.cmpi",
        "arith.andi",
        "arith.cmpf",
        "arith.ori",
        "arith.select",
        "arith.select",
        "tt.reduce.return",
    ):
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", failure)
    a, i, b, j = block.arguments
    equal, index, tie, better, choose, value, position, end = block.operations
    if (
        scalar_type(a.type) != FLOAT
        or scalar_type(i.type).kind != "int"
        or equal.operands != (a, b)
        or equal.integer("predicate") != 1
        or better.operands != (a, b)
        or better.integer("predicate") not in (2, 4)
        or index.operands != (i, j)
        or index.integer("predicate") not in (2, 4, 6, 8)
        or tie.operands != equal.results + index.results
        or choose.operands not in (better.results + tie.results, tie.results + better.results)
        or value.operands != choose.results + (a, b)
        or position.operands != choose.results + (i, j)
        or end.operands != value.results + position.results
    ):
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", failure)
    shape_out, offsets = _reduction_offsets(op, sources[0])
    if sources[0].shape != sources[1].shape or any(shape(r.type) != shape_out for r in op.results):
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "value/index reduction shapes")
    pairs = [
        (
            _non_nan_reduction_operand(scalar(reindex(sources[0], offset, shape_out))),
            scalar(reindex(sources[1], offset, shape_out)),
        )
        for offset in offsets
    ]

    def merge(left: tuple[Expr, Expr], right: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
        lhs, lhs_index = left
        rhs, rhs_index = right
        same = _pointwise(equal, (lhs, rhs), BOOL)
        preferred = _pointwise(index, (lhs_index, rhs_index), BOOL)
        outranks = _pointwise(better, (lhs, rhs), BOOL)
        take = Expr("or", BOOL, (outranks, Expr("and", BOOL, (same, preferred))))
        return Expr("select", lhs.type, (take, lhs, rhs)), Expr(
            "select", lhs_index.type, (take, lhs_index, rhs_index)
        )

    # Lexicographic min/max under a total order is independent of the reduction tree.
    while len(pairs) > 1:
        pairs = [
            merge(pairs[i], pairs[i + 1]) if i + 1 < len(pairs) else pairs[i]
            for i in range(0, len(pairs), 2)
        ]
    return tuple(Tensor(shape_out, term) for term in pairs[0])


def _reduction_offsets(op: Operation, source: Tensor) -> tuple[tuple[int, ...], list[Expr]]:
    axis = op.integer("axis")
    if not 0 <= axis < len(source.shape):
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reduction axis")
    extent = source.shape[axis]
    if extent > 4096:
        raise ResourceLimit("static reduction extent")
    result_shape = source.shape[:axis] + source.shape[axis + 1 :]
    tail = prod(source.shape[axis + 1 :])
    # Scalar results must not retain a lane variable that a subsequent splat can reindex.
    base = (
        index_expr(
            "add",
            index_expr("mul", index_expr("div", var("__lane"), const(tail)), const(extent * tail)),
            index_expr("rem", var("__lane"), const(tail)),
        )
        if result_shape
        else const(0)
    )
    return result_shape, [index_expr("add", base, const(i * tail)) for i in range(extent)]


def _reduce(op: Operation, source: Tensor) -> Tensor:
    """Expand an associative, single-result static reduction as a balanced tree."""
    if len(op.regions) != 1 or len(op.regions[0]) != 1 or len(op.results) != 1:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "multi-result reduction")
    block = op.regions[0][0]
    if len(block.arguments) != 2:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "complex reduction combiner")
    normalized = _comparison_reduction(block)
    if normalized is None and len(block.operations) != 2:
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "complex reduction combiner")
    combine, terminator = block.operations[0], block.operations[-1]
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
    if normalized is None and (
        combine.name not in allowed
        or combine.regions
        or len(combine.results) != 1
        or tuple(v.id for v in combine.operands) != tuple(v.id for v in block.arguments)
        or terminator.name != "tt.reduce.return"
        or tuple(v.id for v in terminator.operands) != (combine.results[0].id,)
    ):
        raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "non-associative reduction combiner")
    result_shape, offsets = _reduction_offsets(op, source)
    if shape(op.results[0].type) != result_shape:
        raise UnsupportedSemantics("TTIR_SHAPE_UNSUPPORTED", "reduction result")
    terms = [scalar(reindex(source, offset, result_shape)) for offset in offsets]
    if normalized is not None:
        terms = [_non_nan_reduction_operand(term) for term in terms]
    typ = scalar_type(op.results[0].type)

    def merge(left: Expr, right: Expr) -> Expr:
        return (
            Expr(normalized, typ, (left, right))
            if normalized
            else _pointwise(combine, (left, right), typ)
        )

    while len(terms) > 1:
        terms = [
            merge(terms[i], terms[i + 1]) if i + 1 < len(terms) else terms[i]
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
    nested = operations(function)
    structured = any(op.name in ("scf.for", "scf.if") for op in nested)
    if structured and any(
        attr.name == "overflowFlags" and attr.values != (0,)
        for op in nested
        for attr in op.attributes
    ):
        raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "integer overflow flags in SCF kernel")
    normalized_isinf = _nonzero_only_isinf_uses(block.operations)
    uses: dict[str, list[tuple[Operation, int]]] = {}
    definitions = {result.id: user for user in nested for result in user.results}
    for user in nested:
        for index, operand in enumerate(user.operands):
            uses.setdefault(operand.id, []).append((user, index))

    def bitcast_routes(result: str) -> bool:
        pending = [result]
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            for user, index in uses.get(name, []):
                if user.name == "tt.bitcast" and scalar_type(user.results[0].type) == FLOAT:
                    continue
                if user.name not in (
                    "tt.reshape",
                    "tt.expand_dims",
                    "tt.broadcast",
                    "tt.splat",
                    "tt.reduce",
                    "arith.xori",
                    "arith.select",
                ) or (user.name == "arith.select" and index == 0):
                    return False
                pending.extend(value.id for value in user.results)
        return True

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
    reads: list[MemoryRead] = []
    selected = {s.index: s.role for s in launch.stores}
    store_indices = {
        id(op): index for index, op in enumerate(op for op in nested if op.name == "tt.store")
    }
    store_count = 0
    conditions: list[Expr] = []
    expanded_ops = 0

    def guarded(condition: Expr, predicate: Expr) -> Expr:
        if condition == const(True, BOOL):
            return predicate
        return simplify(Expr("select", BOOL, (condition, predicate, const(False, BOOL))))

    def assign(op: Operation, env: dict[str, Tensor], results: tuple[Tensor, ...]) -> None:
        for value, result in zip(op.results, results, strict=True):
            env[value.id] = result

    def region(op: Operation, index: int) -> Block:
        if len(op.regions[index]) != 1:
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", op.name + ": expected one block")
        block = op.regions[index][0]
        if not block.operations or block.operations[-1].name != "scf.yield":
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", op.name + ": expected scf.yield")
        return block

    def execute(
        block: Block, env: dict[str, Tensor], args: tuple[Tensor, ...], active: Expr, depth: int
    ) -> tuple[Tensor, ...]:
        if depth > 8:
            raise ResourceLimit("structured control flow nesting")
        local = env | {arg.id: value for arg, value in zip(block.arguments, args, strict=True)}
        lift_block(block.operations[:-1], local, active, depth)
        return tuple(local[val.id] for val in block.operations[-1].operands)

    def lower_if(
        op: Operation, env: dict[str, Tensor], active: Expr, depth: int
    ) -> tuple[Tensor, ...]:
        condition_tensor = env[op.operands[0].id]
        if condition_tensor.shape or scalar(condition_tensor).type != BOOL:
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "scf.if: expected scalar i1")
        condition = simplify(scalar(reindex(condition_tensor, const(0), ())), launch.index_bits)
        then_block = region(op, 0)
        else_block = region(op, 1) if len(op.regions) > 1 and op.regions[1] else None
        if condition.op == "const":
            taken = then_block if condition.attr("value") else else_block
            return execute(taken, env, (), active, depth + 1) if taken else ()
        # Check that the branch decision is defined even when its results are unused.
        conditions.append(simplify(Expr("select", BOOL, (active, condition, const(True, BOOL)))))
        left = execute(then_block, env, (), guarded(active, condition), depth + 1)
        right = (
            execute(
                else_block, env, (), guarded(active, Expr("not", BOOL, (condition,))), depth + 1
            )
            if else_block
            else ()
        )
        merged = []
        for a, b in zip(left, right, strict=True):
            if isinstance(a.term, Pointer) or isinstance(b.term, Pointer):
                raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "scf.if: pointer results")
            merged.append(Tensor(a.shape, Expr("select", a.term.type, (condition, a.term, b.term))))
        return tuple(merged)

    def lower_for(
        op: Operation, env: dict[str, Tensor], active: Expr, depth: int
    ) -> tuple[Tensor, ...]:
        if len(op.regions) != 1 or any(a.name == "unsignedCmp" for a in op.attributes):
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "scf.for: expected signed loop")
        block = region(op, 0)
        args = tuple(env[v.id] for v in op.operands)
        bounds = tuple(scalar(a) for a in args[:3])
        index_type = bounds[0].type
        try:
            lower, upper, step = (evaluate(bound, index_bits=launch.index_bits) for bound in bounds)
            bits = width(index_type, launch.index_bits)
        except (KeyError, UnsupportedSemantics, ZeroDivisionError):
            raise UnsupportedSemantics(
                "TTIR_REGION_UNSUPPORTED", "scf.for: requires defined static bounds and step"
            ) from None
        if (
            not isinstance(lower, int)
            or not isinstance(upper, int)
            or not isinstance(step, int)
            or step <= 0
        ):
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "scf.for: requires positive step")
        trips = max(0, -((lower - upper) // step))
        limit = 1 << (bits - 1)
        if any(not -limit <= n < limit for n in (lower, upper, step, lower + trips * step)):
            raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", "scf.for: induction may overflow")
        carried = args[3:]
        if not trips:
            return carried
        summary = summarize_recurrence(block, env, carried, trips)
        if summary is not None:
            return summary
        if trips > min(64, launch.max_instances):
            raise ResourceLimit("scf.for: exact expansion exceeds 64 iterations or instance budget")
        for induction in range(lower, upper, step):
            iv = Tensor((), const(induction, index_type))
            carried = execute(block, env, (iv, *carried), active, depth + 1)
        return carried

    def summarize_recurrence(
        block: Block, env: dict[str, Tensor], carried: tuple[Tensor, ...], trips: int
    ) -> tuple[Tensor, ...] | None:
        body, terminator = block.operations[:-1], block.operations[-1]
        if not body and not carried:
            return ()
        if len(carried) != 1 or len(body) != 1:
            return None
        update = body[0]
        if update.name not in ("arith.addi", "arith.subi") or terminator.operands != update.results:
            return None
        arg = block.arguments[1]
        before, increment = update.operands
        if update.name == "arith.addi" and increment == arg:
            before, increment = increment, before
        initial, delta = carried[0], env.get(increment.id)
        if before != arg or delta is None or initial.shape != delta.shape:
            return None
        if not isinstance(initial.term, Expr) or not isinstance(delta.term, Expr):
            return None
        typ = initial.term.type
        delta_term = simplify(delta.term, launch.index_bits)
        if typ.kind != "int" or delta_term.op != "const":
            return None
        # Unflagged fixed-width integer operations have modular semantics.
        delta_value = delta_term.attr("value")
        assert isinstance(delta_value, int)
        amount = trips * delta_value
        combined = Expr(
            "add",
            typ,
            (initial.term, const(amount if update.name == "arith.addi" else -amount, typ)),
        )
        return (Tensor(initial.shape, simplify(combined, launch.index_bits)),)

    def lift_block(
        ops: tuple[Operation, ...], values: dict[str, Tensor], active: Expr, depth: int
    ) -> None:
        nonlocal expanded_ops, store_count
        for op in ops:
            if depth:
                expanded_ops += 1
                if expanded_ops > launch.max_instances:
                    raise ResourceLimit("structured control flow operation expansion")
            if op.name == "scf.for":
                assign(op, values, lower_for(op, values, active, depth))
                continue
            if op.name == "scf.if":
                assign(op, values, lower_if(op, values, active, depth))
                continue
            if op.name == "tt.reduce":
                if len(op.operands) != 1:
                    assign(op, values, _arg_reduction(op, tuple(values[v.id] for v in op.operands)))
                else:
                    values[op.results[0].id] = _reduce(op, values[op.operands[0].id])
                continue
            if op.regions:
                raise UnsupportedSemantics("TTIR_REGION_UNSUPPORTED", op.name)
            if op.name == "gpu.barrier":
                if op.operands or op.results or op.attributes:
                    raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "nonstandard gpu.barrier")
                if depth or stores:
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
                store_index = store_indices[id(op)] if structured else store_count
                store_count += 1
                pointer = args[0].term
                if not isinstance(pointer, Pointer):
                    raise UnsupportedSemantics("TTIR_TYPE_UNSUPPORTED", "store pointer")
                if store_index not in selected:
                    raise UnsupportedSemantics("UNSELECTED_STORE", "all effects must be declared")
                if (
                    selected[store_index] != pointer.storage.logical_role
                    or pointer.storage.scalar_index is not None
                ):
                    raise InputError("ABI_ROLE_MISSING", "store role disagrees with pointer")
                mask = scalar(args[2]) if len(args) == 3 else const(True, BOOL)
                mask = guarded(active, mask)
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
                    raise UnsupportedSemantics(
                        "TTIR_TYPE_UNSUPPORTED", "store element type mismatch"
                    )
                stores.append(
                    Store(
                        store_index,
                        pointer.storage,
                        pointer.offset,
                        pointer.offset,
                        mask,
                        value_expr,
                        args[0].shape,
                        structured=bool(depth),
                        order=len(reads) + len(stores),
                    )
                )
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
                axis = op.integer("axis")
                assert typ is not None
                coordinate = var("__pid") if axis == 0 else const(0)
                if launch.grid_shape is not None:
                    coordinate = index_expr(
                        "rem",
                        index_expr("div", var("__pid"), const(prod(launch.grid_shape[:axis]))),
                        const(launch.grid_shape[axis]),
                    )
                result = Tensor((), Expr("coordinate", typ, (coordinate,)))
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
                source_type = op.operands[0].type
                source_element = (
                    source_type.element if source_type.kind == "tensor" else source_type
                )
                if (
                    isinstance(pointer, Expr)
                    and source_element is not None
                    and target is not None
                    and source_element.bits == target.bits
                    and source_element.bits in (8, 16, 32, 64)
                    and {source_element.kind, target.kind} == {"float", "int"}
                ):
                    if source_element.kind == "float":
                        producer = definitions.get(op.operands[0].id)
                        if (
                            producer is not None and producer.name == "arith.select"
                        ) or not bitcast_routes(op.results[0].id):
                            raise UnsupportedSemantics(
                                "TTIR_OP_UNSUPPORTED", "unsupported pointer/bit reinterpretation"
                            )
                    term = Expr(
                        "float_to_bits" if source_element.kind == "float" else "bits_to_float",
                        scalar_type(target),
                        (pointer, var("__lane") if out_shape else const(0))
                        if source_element.kind == "float"
                        else (pointer,),
                        (("format", source_element.assembly), ("origin", op.ordinal))
                        if source_element.kind == "float"
                        else (("format", target.assembly),),
                    )
                    assign(op, values, (Tensor(out_shape, term),))
                    continue
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
                        "TTIR_OP_UNSUPPORTED",
                        "boolean byte load requires canonical storage contract",
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
                order = len(reads) + len(stores)
                term = Expr(
                    "load",
                    pointer.element,
                    (pointer.offset, mask, default),
                    (("order", order), ("storage", pointer.storage)),
                )
                effects.append(term)
                reads.append(MemoryRead(term, out_shape, order, active))
                result = Tensor(
                    out_shape,
                    Expr("extui", Type("int", 8), (term,)) if pointer.boolean_byte else term,
                )
            else:
                assert typ is not None
                result = Tensor(
                    out_shape,
                    _pointwise(
                        op, tuple(scalar(a) for a in args), typ, op.ordinal in normalized_isinf
                    ),
                )
                if (
                    op.name in ("arith.index_cast", "arith.index_castui")
                    and launch.index_bits is None
                ):
                    raise UnsupportedSemantics("INDEX_WIDTH_REQUIRED", op.name)
            values[op.results[0].id] = result

    lift_block(block.operations, values, const(True, BOOL), 0)
    expected = set(store_indices.values()) if structured else set(range(store_count))
    if expected - selected.keys():
        raise UnsupportedSemantics("UNSELECTED_STORE", "all effects must be declared")
    if selected.keys() - expected:
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
        tuple(conditions),
        tuple(reads),
    )
