"""Partial extended-real semantics: finite exact reals and signed infinities.

NaN and signed zero are deliberately absent. Invalid combinations are undefined,
not converted into finite sentinels. External floating inputs remain finite.
"""

from fractions import Fraction
from typing import Literal, TypeAlias

from .errors import UnsupportedSemantics

Infinity: TypeAlias = Literal["+inf", "-inf"]
ExtendedReal: TypeAlias = Fraction | int | bool | Infinity


def is_infinite(value: object) -> bool:
    return value in ("+inf", "-inf")


def undefined() -> None:
    raise UnsupportedSemantics(
        "FLOAT_DEFINEDNESS_NOT_PROVED", "NaN-producing or unsupported infinite arithmetic"
    )


def negative(value: ExtendedReal) -> bool:
    return value == "-inf" if isinstance(value, str) else value < 0


def negate(value: ExtendedReal) -> ExtendedReal:
    return ("-inf" if value == "+inf" else "+inf") if isinstance(value, str) else -value


def compare(op: str, a: ExtendedReal, b: ExtendedReal) -> bool:
    def key(value: ExtendedReal) -> tuple[int, Fraction]:
        if value == "-inf":
            return (-1, Fraction(0))
        if value == "+inf":
            return (1, Fraction(0))
        assert not isinstance(value, str)
        return (0, Fraction(value))

    left, right = key(a), key(b)
    return {
        "eq": left == right,
        "ne": left != right,
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[op]


def evaluate_special(op: str, values: tuple[ExtendedReal, ...]) -> ExtendedReal:
    """Evaluate operations with at least one infinity, without host floats."""
    a = values[0]
    b = values[1] if len(values) > 1 else 0
    ai, bi = is_infinite(a), is_infinite(b)
    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
        return compare(op, a, b)
    if op in ("extf", "truncf", "floor", "ceil", "nearbyint"):
        return a
    if op == "fneg":
        return negate(a)
    if op == "absf":
        return "+inf"
    if op == "fsub":
        return evaluate_special("fadd", (a, negate(b)))
    if op == "fadd":
        if ai and bi and a != b:
            undefined()
        return a if ai else b
    if op == "fmul":
        if (ai and b == 0) or (bi and a == 0):
            undefined()
        return "-inf" if negative(a) != negative(b) else "+inf"
    if op == "fdiv":
        if (ai and bi) or b == 0:
            undefined()
        if bi:
            return Fraction(0)
        return "-inf" if negative(a) != negative(b) else "+inf"
    if op in ("fmin", "fmax"):
        return a if compare("le" if op == "fmin" else "ge", a, b) else b
    if op == "fma":
        product = evaluate_special("fmul", (a, b)) if ai or bi else Fraction(a) * Fraction(b)
        c = values[2]
        return (
            evaluate_special("fadd", (product, c))
            if is_infinite(product) or is_infinite(c)
            else Fraction(product) + Fraction(c)
        )
    if op in ("exp", "exp2", "expm1"):
        return "+inf" if a == "+inf" else Fraction(-1 if op == "expm1" else 0)
    if op in ("log", "sqrt", "rsqrt", "acosh") and a == "+inf":
        return Fraction(0) if op == "rsqrt" else "+inf"
    if op == "cosh":
        return "+inf"
    if op in ("tanh", "erf"):
        return Fraction(-1 if a == "-inf" else 1)
    undefined()
    raise AssertionError("unreachable")
