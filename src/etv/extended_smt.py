"""Tagged SMT encoding of partial extended reals, without finite sentinels."""

from fractions import Fraction

import z3

FLOAT_SORT = z3.Datatype("ETVExtendedReal")
FLOAT_SORT.declare("Finite", ("finite_value", z3.RealSort()))
FLOAT_SORT.declare("PositiveInfinity")
FLOAT_SORT.declare("NegativeInfinity")
FLOAT_SORT = FLOAT_SORT.create()
POS_INF = FLOAT_SORT.PositiveInfinity
NEG_INF = FLOAT_SORT.NegativeInfinity


def finite(value: z3.ExprRef | int) -> z3.ExprRef:
    return FLOAT_SORT.Finite(value)


def number(value: int | Fraction | str) -> z3.ExprRef:
    if value == "+inf":
        return POS_INF
    if value == "-inf":
        return NEG_INF
    fraction = Fraction(value)
    return finite(z3.RealVal(f"{fraction.numerator}/{fraction.denominator}"))


def is_finite(value: z3.ExprRef) -> z3.BoolRef:
    return FLOAT_SORT.is_Finite(value)


def real(value: z3.ExprRef) -> z3.ArithRef:
    return FLOAT_SORT.finite_value(value)


def is_zero(value: z3.ExprRef) -> z3.BoolRef:
    return z3.And(is_finite(value), real(value) == 0)


def is_negative(value: z3.ExprRef) -> z3.BoolRef:
    return z3.Or(value == NEG_INF, z3.And(is_finite(value), real(value) < 0))


def negate(value: z3.ExprRef) -> z3.ExprRef:
    return z3.If(is_finite(value), finite(-real(value)), z3.If(value == POS_INF, NEG_INF, POS_INF))


def less(a: z3.ExprRef, b: z3.ExprRef) -> z3.BoolRef:
    return z3.Or(
        z3.And(a == NEG_INF, b != NEG_INF),
        z3.And(b == POS_INF, a != POS_INF),
        z3.And(is_finite(a), is_finite(b), real(a) < real(b)),
    )


def compare(op: str, a: z3.ExprRef, b: z3.ExprRef) -> z3.BoolRef:
    if op == "eq":
        return a == b
    if op == "ne":
        return a != b
    if op == "lt":
        return less(a, b)
    if op == "le":
        return z3.Or(a == b, less(a, b))
    if op == "gt":
        return less(b, a)
    if op == "ge":
        return z3.Or(a == b, less(b, a))
    raise ValueError(op)


def add(a: z3.ExprRef, b: z3.ExprRef) -> tuple[z3.ExprRef, z3.BoolRef]:
    value = z3.If(is_finite(a), z3.If(is_finite(b), finite(real(a) + real(b)), b), a)
    valid = z3.Not(z3.Or(z3.And(a == POS_INF, b == NEG_INF), z3.And(a == NEG_INF, b == POS_INF)))
    return value, valid


def multiply(a: z3.ExprRef, b: z3.ExprRef) -> tuple[z3.ExprRef, z3.BoolRef]:
    value = z3.If(
        z3.And(is_finite(a), is_finite(b)),
        finite(real(a) * real(b)),
        z3.If(is_negative(a) != is_negative(b), NEG_INF, POS_INF),
    )
    valid = z3.Not(
        z3.Or(z3.And(is_zero(a), z3.Not(is_finite(b))), z3.And(is_zero(b), z3.Not(is_finite(a))))
    )
    return value, valid


def operation(op: str, args: list[z3.ExprRef]) -> tuple[z3.ExprRef, z3.BoolRef, bool]:
    """Return value, local definedness, and exactness of supported operation."""
    a = args[0]
    b = args[1] if len(args) > 1 else a
    true = z3.BoolVal(True)
    if op == "fadd":
        value, valid = add(a, b)
        return value, valid, True
    if op == "fsub":
        value, valid = add(a, negate(b))
        return value, valid, True
    if op == "fmul":
        value, valid = multiply(a, b)
        return value, valid, True
    if op == "fdiv":
        value = z3.If(
            is_finite(a),
            z3.If(is_finite(b), finite(real(a) / real(b)), finite(0)),
            z3.If(is_negative(a) != is_negative(b), NEG_INF, POS_INF),
        )
        valid = z3.And(z3.Not(is_zero(b)), z3.Or(is_finite(a), is_finite(b)))
        return value, valid, True
    if op == "fma":
        product, product_valid = multiply(a, b)
        value, sum_valid = add(product, args[2])
        return value, z3.And(product_valid, sum_valid), True
    if op == "fneg":
        return negate(a), true, True
    if op == "absf":
        return z3.If(is_negative(a), negate(a), a), true, True
    if op in ("fmin", "fmax"):
        return z3.If(less(a, b) if op == "fmin" else less(b, a), a, b), true, True
    if op in ("floor", "ceil", "nearbyint"):
        x = real(a)
        if op == "floor":
            value = z3.ToReal(z3.ToInt(x))
        elif op == "ceil":
            value = -z3.ToReal(z3.ToInt(-x))
        else:
            low = z3.ToInt(x)
            value = z3.ToReal(
                z3.If(
                    x - low < z3.RealVal("1/2"),
                    low,
                    z3.If(x - low > z3.RealVal("1/2"), low + 1, z3.If(low % 2 == 0, low, low + 1)),
                )
            )
        return z3.If(is_finite(a), finite(value), a), true, True
    # Finite transcendental functions retain the existing uninterpreted semantics.
    # Infinity limits are exact only where listed. No arithmetic limit is inferred.
    finite_args = [real(arg) for arg in args]
    function = z3.Function("math." + op, *[z3.RealSort() for _ in args], z3.RealSort())
    value = finite(function(*finite_args))
    valid = z3.And(*(is_finite(arg) for arg in args))
    if op in ("sqrt", "rsqrt", "log", "acosh", "pow"):
        x = real(a)
        domain = x >= 0 if op == "sqrt" else x >= 1 if op == "acosh" else x > 0
        valid = z3.And(valid, domain)
    if op in ("exp", "exp2", "expm1"):
        limit = z3.If(a == POS_INF, POS_INF, finite(-1 if op == "expm1" else 0))
        return z3.If(is_finite(a), value, limit), true, z3.is_false(z3.simplify(is_finite(a)))
    if op in ("sqrt", "rsqrt", "log", "acosh"):
        limit = finite(0) if op == "rsqrt" else POS_INF
        return (
            z3.If(is_finite(a), value, limit),
            z3.Or(valid, a == POS_INF),
            z3.is_false(z3.simplify(is_finite(a))),
        )
    if op == "cosh":
        return z3.If(is_finite(a), value, POS_INF), true, z3.is_false(z3.simplify(is_finite(a)))
    if op in ("tanh", "erf"):
        limit = z3.If(a == NEG_INF, finite(-1), finite(1))
        return z3.If(is_finite(a), value, limit), true, z3.is_false(z3.simplify(is_finite(a)))
    return value, valid, False
