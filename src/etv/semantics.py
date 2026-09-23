"""Exact scalar evaluation. Partial and non-algebraic functions stay explicit."""

from fractions import Fraction
from functools import lru_cache
from typing import Mapping

from .errors import UnsupportedSemantics
from .ir import BOOL, INDEX, Expr, Type, const, signed, trunc_div

Value = int | bool | Fraction
CASTS = frozenset("extsi extui trunci index_cast index_castui sitofp uitofp extf truncf".split())
FLOAT_UNARY = frozenset(
    "absf ceil cos erf exp exp2 floor log rsqrt sin sqrt acosh atan cosh expm1 nearbyint tanh".split()
)
FLOAT_BINARY = frozenset("fadd fsub fmul fdiv fmin fmax pow".split())
INTEGER_BINARY = frozenset("add sub mul div rem ceildiv".split())
COMPARISONS = frozenset("eq ne lt le gt ge".split())


def arity(op: str) -> int:
    if op in CASTS | FLOAT_UNARY | {"fneg", "not", "coordinate", "to_index", "read"}:
        return 1
    if op in FLOAT_BINARY | INTEGER_BINARY | COMPARISONS | {"and", "or", "xor"}:
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
    elif expr.op in COMPARISONS:
        valid = expr.type == BOOL and types[0] == types[1]
    elif expr.op in {"and", "or", "xor", "not"}:
        valid = expr.type == BOOL and all(t == BOOL for t in types)
    elif expr.op == "select":
        valid = types[0] == BOOL and types[1] == types[2] == expr.type
    elif expr.op in {"extf", "truncf"}:
        valid = expr.type.kind == types[0].kind == "abstract_float"
    elif expr.op in {"sitofp", "uitofp"}:
        valid = expr.type.kind == "abstract_float" and types[0].kind in ("int", "bool")
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
        assert isinstance(value, (int, bool, Fraction))
        return value
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
    values = tuple(evaluate(a, env, index_bits) for a in expr.args)
    a = values[0]
    b = values[1] if len(values) > 1 else 0
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
    elif op == "fdiv":
        result = Fraction(a) / Fraction(b)
    elif op in ("eq", "ne", "lt", "le", "gt", "ge"):
        return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[
            op
        ]
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
        result = a * b + values[2]
    else:
        raise UnsupportedSemantics("EXACT_EVALUATION_UNAVAILABLE", op)
    if expr.type.kind == "int":
        assert expr.type.bits is not None
        return signed(int(result), expr.type.bits)
    return Fraction(result) if expr.type.kind == "abstract_float" else result


@lru_cache(maxsize=100000)
def simplify(expr: Expr, index_bits: int | None = None) -> Expr:
    if not expr.args:
        return expr
    args = tuple(simplify(a, index_bits) for a in expr.args)
    result = Expr(expr.op, expr.type, args, expr.attrs)
    if expr.op == "select" and args[0].op == "const":
        return args[1 if args[0].attr("value") else 2]
    if all(a.op == "const" for a in args):
        try:
            return const(evaluate(result, index_bits=index_bits), result.type)
        except (UnsupportedSemantics, ZeroDivisionError):
            pass
    return result
