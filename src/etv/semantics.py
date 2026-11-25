"""Exact scalar evaluation. Partial and non-algebraic functions stay explicit."""

from fractions import Fraction
from functools import lru_cache
from typing import Mapping

from .errors import UnsupportedSemantics
from .extended_float import SpecialFloat, evaluate_special
from .ir import BOOL, FLOAT, INDEX, Expr, Type, const, signed, trunc_div, walk

Value = int | bool | Fraction | SpecialFloat
CASTS = frozenset(
    "extsi extui trunci index_cast index_castui sitofp uitofp extf truncf fptosi fptoui".split()
)
FLOAT_UNARY = frozenset(
    "absf ceil cos erf exp exp2 floor log rsqrt sin sqrt acosh atan cosh expm1 nearbyint tanh".split()
)
FLOAT_BINARY = frozenset("fadd fsub fmul fdiv fmin fmax pow".split())
INTEGER_BINARY = frozenset("add sub mul div rem ceildiv shrui mulhiui".split())
COMPARISONS = frozenset("eq ne lt le gt ge".split())


def arity(op: str) -> int:
    if op in CASTS | FLOAT_UNARY | {
        "fneg",
        "not",
        "coordinate",
        "to_index",
        "read",
        "isnan",
        "isinf",
        "signbit",
    }:
        return 1
    if op in FLOAT_BINARY | INTEGER_BINARY | COMPARISONS | {
        "and",
        "or",
        "xor",
        "ordered",
        "unordered",
        "float_true",
        "float_false",
    }:
        return 2
    if op in {"select", "fma", "load"}:
        return 3
    if op in {"const", "var", "undefined", "input"}:
        return 0
    raise ValueError(f"unknown semantic operator: {op}")


def validate_expr(expr: Expr) -> None:
    if len(expr.args) != arity(expr.op):
        raise ValueError(f"arity mismatch: {expr.op}")
    types = [a.type for a in expr.args]
    if expr.op in FLOAT_BINARY | FLOAT_UNARY | {"fneg", "fma"}:
        valid = expr.type.kind == "abstract_float" and all(
            t.kind == "abstract_float" for t in types
        )
    elif expr.op in INTEGER_BINARY:
        valid = expr.type.kind in ("int", "index") and all(t == expr.type for t in types)
    elif expr.op in {
        "ordered",
        "unordered",
        "isnan",
        "isinf",
        "signbit",
        "float_true",
        "float_false",
    }:
        valid = expr.type == BOOL and all(t.kind == "abstract_float" for t in types)
    elif expr.op in COMPARISONS:
        valid = expr.type == BOOL and types[0] == types[1]
    elif expr.op in {"and", "or", "xor"}:
        valid = expr.type.kind in ("bool", "int") and all(t == expr.type for t in types)
    elif expr.op == "not":
        valid = expr.type == BOOL and all(t == BOOL for t in types)
    elif expr.op == "select":
        valid = types[0] == BOOL and types[1] == types[2] == expr.type
    elif expr.op in {"extf", "truncf"}:
        valid = expr.type.kind == types[0].kind == "abstract_float"
    elif expr.op in {"sitofp", "uitofp"}:
        valid = expr.type.kind == "abstract_float" and types[0].kind in ("int", "bool")
    elif expr.op in {"fptosi", "fptoui"}:
        valid = expr.type.kind == "int" and types[0].kind == "abstract_float"
    elif expr.op in CASTS:
        valid = expr.type.kind in ("int", "bool", "index") and types[0].kind in (
            "int",
            "bool",
            "index",
        )
    else:
        valid = True
    if not valid:
        raise ValueError(f"type mismatch: {expr.op}")


def width(typ: Type, index_bits: int | None) -> int:
    bits = 1 if typ == BOOL else index_bits if typ == INDEX else typ.bits
    if bits is None:
        raise UnsupportedSemantics("INDEX_WIDTH_REQUIRED", "cast requires target.index_bits")
    return bits


def evaluate(
    expr: Expr, environment: Mapping[str, Value] | None = None, index_bits: int | None = None
) -> Value:
    env = environment or {}
    op = expr.op
    if op == "const":
        value = expr.attr("value")
        assert isinstance(value, (int, bool, Fraction)) or value in ("+inf", "-inf", "nan")
        return value  # type: ignore[return-value]
    if op == "var":
        return env[str(expr.attr("name"))]
    if op == "input":
        return env[str(expr.attr("key"))]
    if op == "select":
        return evaluate(
            expr.args[1 if evaluate(expr.args[0], env, index_bits) else 2], env, index_bits
        )
    if op == "undefined":
        raise UnsupportedSemantics("UNDEFINED_LOAD_REACHABLE")
    if op in ("float_to_bits", "bits_to_float"):
        raise UnsupportedSemantics("TTIR_OP_UNSUPPORTED", "observable float bit pattern")
    values = tuple(evaluate(a, env, index_bits) for a in expr.args)
    a = values[0]
    b = values[1] if len(values) > 1 else 0
    if op in ("ordered", "unordered", "isnan"):
        return ("nan" not in values) if op == "ordered" else ("nan" in values)
    if op == "isinf":
        return a in ("+inf", "-inf")
    if op in ("float_true", "float_false"):
        return op == "float_true"
    if any(isinstance(v, str) for v in values):
        return evaluate_special(op, values)
    # All special values were handled above.
    assert not isinstance(a, str) and not isinstance(b, str)
    result: Value
    if op in ("coordinate", "to_index"):
        if expr.type.kind == "int":
            assert expr.type.bits is not None
            if not -(1 << (expr.type.bits - 1)) <= int(a) < (1 << (expr.type.bits - 1)):
                raise UnsupportedSemantics("INTEGER_DEFINEDNESS_NOT_PROVED")
        return int(a)
    if op in CASTS:
        if op in ("extf", "truncf"):
            return Fraction(a)
        if op in ("fptosi", "fptoui"):
            bits = width(expr.type, index_bits)
            value = int(Fraction(a))  # truncation toward zero, with poison out of range
            low, high = (-(1 << (bits - 1)), 1 << (bits - 1)) if op == "fptosi" else (0, 1 << bits)
            if not low <= value < high:
                raise UnsupportedSemantics("INTEGER_DEFINEDNESS_NOT_PROVED")
            return signed(value, bits)
        source_width = width(expr.args[0].type, index_bits)
        value = (
            int(a) % (1 << source_width)
            if op in ("extui", "uitofp", "index_castui")
            else signed(int(a), source_width)
        )
        if op in ("sitofp", "uitofp"):
            return Fraction(value)
        if expr.type == BOOL:
            return bool(value % 2)
        return signed(value, width(expr.type, index_bits))
    if op in ("add", "fadd"):
        result = a + b
    elif op in ("sub", "fsub"):
        result = a - b
    elif op in ("mul", "fmul"):
        result = a * b
    elif op in ("div", "rem"):
        if expr.type.bits and int(a) == -(1 << (expr.type.bits - 1)) and b == -1:
            raise UnsupportedSemantics("INTEGER_DEFINEDNESS_NOT_PROVED")
        quotient = trunc_div(int(a), int(b))
        result = quotient if op == "div" else int(a) - quotient * int(b)
    elif op == "ceildiv":
        if b <= 0:
            raise UnsupportedSemantics("INTEGER_DEFINEDNESS_NOT_PROVED")
        result = -(-int(a) // int(b))
    elif op in ("shrui", "mulhiui"):
        bits = width(expr.type, index_bits)
        left, right = int(a) % (1 << bits), int(b) % (1 << bits)
        if op == "shrui":
            if right >= bits:
                raise UnsupportedSemantics("INTEGER_DEFINEDNESS_NOT_PROVED")
            result = left >> right
        else:
            result = (left * right) >> bits
    elif op == "fdiv":
        result = Fraction(a) / Fraction(b)
    elif op in ("eq", "ne", "lt", "le", "gt", "ge"):
        return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[
            op
        ]
    elif op in ("and", "or", "xor") and expr.type.kind == "int":
        result = (
            int(a) & int(b) if op == "and" else int(a) | int(b) if op == "or" else int(a) ^ int(b)
        )
    elif op == "and":
        return bool(a) and bool(b)
    elif op == "or":
        return bool(a) or bool(b)
    elif op == "xor":
        return bool(a) != bool(b)
    elif op == "not":
        return not a
    elif op == "fneg":
        result = -a
    elif op == "absf":
        result = abs(Fraction(a))
    elif op == "pow" and b in (1, 2, 3, 4):
        result = Fraction(a) ** int(b)
    elif op == "pow" and a == 0 and b > 0:
        result = 0
    elif op in ("sqrt", "rsqrt") and a < 0:
        return "nan"
    elif op == "sqrt" and a == 0:
        result = 0
    elif op == "fmin":
        result = min(a, b)
    elif op == "fmax":
        result = max(a, b)
    elif op == "floor":
        result = Fraction(a).__floor__()
    elif op == "ceil":
        result = Fraction(a).__ceil__()
    elif op == "nearbyint":
        result = round(Fraction(a))
    elif op == "fma":
        c = values[2]
        assert not isinstance(c, str)
        result = a * b + c
    else:
        raise UnsupportedSemantics("EXACT_EVALUATION_UNAVAILABLE", op)
    if expr.type.kind == "int":
        assert expr.type.bits is not None
        return signed(int(result), expr.type.bits)
    return Fraction(result) if expr.type.kind == "abstract_float" else result


def _routed_float(bits: Expr, float_format: str) -> Expr | None:
    if bits.op == "float_to_bits":
        return bits.args[0] if bits.attr("format") == float_format else None
    if bits.op == "select":
        condition, yes, no = bits.args
        left, right = _routed_float(yes, float_format), _routed_float(no, float_format)
        if left is not None and right is not None:
            return Expr("select", left.type, (condition, left, right))
        return None
    if bits.op != "xor":
        return None

    # Expand each shared XOR node once and keep parity separately from
    # definedness. Only syntactically identical bits from the same cast cancel.
    pending = [bits]
    nodes: set[Expr] = set()
    while pending:
        node = pending.pop()
        if node in nodes:
            continue
        nodes.add(node)
        if node.op == "xor" and node.type == bits.type:
            pending.extend(node.args)
        elif node.op != "float_to_bits" or node.attr("format") != float_format:
            return None
    parity = {bits}
    sources = set()
    top_down = [node for node in reversed(walk(bits)) if node in nodes]
    for node in top_down:
        if node.op == "xor" and node.type == bits.type:
            if node in parity:
                parity.remove(node)
                for child in node.args:
                    parity.symmetric_difference_update((child,))
        elif node.op == "float_to_bits" and node.type == bits.type:
            sources.add(node)
    if len(parity) != 1 or not parity <= sources:
        return None
    chosen = next(iter(parity))
    value = chosen.args[0]
    # The canceled operands must still be defined. Both select arms carry
    # the same value, while the comparisons retain that proof obligation.
    for source in top_down:
        if source != chosen and source in sources:
            dep = source.args[0]
            value = Expr("select", FLOAT, (Expr("ordered", BOOL, (dep, dep)), value, value))
    return value


@lru_cache(maxsize=100000)
def simplify(expr: Expr, index_bits: int | None = None) -> Expr:
    if not expr.args:
        return expr
    args = tuple(simplify(a, index_bits) for a in expr.args)
    result = (
        expr
        if all(new is old for new, old in zip(args, expr.args, strict=True))
        else Expr(expr.op, expr.type, args, expr.attrs)
    )
    if expr.op == "select" and args[0].op == "const":
        return args[1 if args[0].attr("value") else 2]
    if (
        expr.op == "shrui"
        and args[0].op == "float_to_bits"
        and args[1].op == "const"
        and expr.type.bits is not None
        and args[1].attr("value") == expr.type.bits - 1
    ):
        return Expr(
            "select",
            expr.type,
            (Expr("signbit", BOOL, (args[0].args[0],)), const(1, expr.type), const(0, expr.type)),
        )
    source = args[0]
    if (
        source.op == "select"
        and source.type.kind == "int"
        and source.args[1] == const(1, source.type)
        and source.args[2] == const(0, source.type)
    ):
        condition = source.args[0]
        if expr.op == "and" and args[1] == const(1, source.type):
            return source
        if expr.op == "trunci":
            if expr.type == BOOL:
                return condition
            return Expr("select", expr.type, (condition, const(1, expr.type), const(0, expr.type)))
        if expr.op in ("eq", "ne") and args[1] == const(0, source.type):
            return condition if expr.op == "ne" else Expr("not", BOOL, (condition,))
    if expr.op == "bits_to_float":
        routed = _routed_float(args[0], str(expr.attr("format")))
        if routed is not None:
            return simplify(routed, index_bits)
    if all(a.op == "const" for a in args):
        try:
            return const(evaluate(result, index_bits=index_bits), result.type)
        except (UnsupportedSemantics, ZeroDivisionError):
            pass
    return result
