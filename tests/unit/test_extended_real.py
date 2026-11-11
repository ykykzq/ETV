"""Differential checks for the tagged solver and exact interpreter."""

from fractions import Fraction
from itertools import product

import pytest
import z3

from etv.counterexample import _number
from etv.errors import UnsupportedSemantics
from etv.ir import BOOL, FLOAT, Expr, RootPair, const, var
from etv.semantics import evaluate
from etv.smt import Encoder, query

VALUES = ("-inf", Fraction(-7, 3), Fraction(0), Fraction(11, 5), "+inf")


@pytest.mark.parametrize(
    "op", ["fadd", "fsub", "fmul", "fdiv", "fmin", "fmax", "eq", "ne", "lt", "le", "gt", "ge"]
)
def test_special_arithmetic_matches_exact_replay(op):
    typ = BOOL if op in ("eq", "ne", "lt", "le", "gt", "ge") else FLOAT
    for a, b in product(VALUES, repeat=2):
        expr = Expr(op, typ, (const(a, FLOAT), const(b, FLOAT)))
        encoder = Encoder()
        defined = z3.simplify(encoder.defined(expr))
        assert z3.is_true(defined) or z3.is_false(defined)
        if z3.is_true(defined):
            assert _number(z3.simplify(encoder.encode(expr))) == evaluate(expr)
        else:
            with pytest.raises((UnsupportedSemantics, ZeroDivisionError)):
                evaluate(expr)


def test_fma_special_arithmetic_matches_exact_replay():
    for values in product(VALUES, repeat=3):
        expr = Expr("fma", FLOAT, tuple(const(v, FLOAT) for v in values))
        encoder = Encoder()
        if z3.is_true(z3.simplify(encoder.defined(expr))):
            assert _number(z3.simplify(encoder.encode(expr))) == evaluate(expr)
        else:
            with pytest.raises(UnsupportedSemantics):
                evaluate(expr)


@pytest.mark.parametrize(
    "op,expected",
    [
        ("exp", (0, "+inf")),
        ("exp2", (0, "+inf")),
        ("expm1", (-1, "+inf")),
        ("cosh", ("+inf", "+inf")),
        ("tanh", (-1, 1)),
        ("erf", (-1, 1)),
        ("absf", ("+inf", "+inf")),
        ("fneg", ("+inf", "-inf")),
        ("floor", ("-inf", "+inf")),
        ("ceil", ("-inf", "+inf")),
        ("nearbyint", ("-inf", "+inf")),
    ],
)
def test_exact_infinite_limits(op, expected):
    for special, answer in zip(("-inf", "+inf"), expected, strict=True):
        expr = Expr(op, FLOAT, (const(special, FLOAT),))
        encoder = Encoder()
        value = encoder.encode(expr)
        assert encoder.exact
        assert z3.is_true(z3.simplify(encoder.defined(expr)))
        assert evaluate(expr) == answer
        assert z3.is_true(z3.simplify(value == encoder.encode(const(answer, FLOAT))))


def test_finite_memory_ground_rule_is_not_global_infinity_rule(tmp_path):
    from etv.eqsat import saturate
    from etv.pairspec import parse_pair_spec
    from etv.rewrites import admit_rules
    from test_contracts import minimal, write

    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    x = var("finite.input", FLOAT)
    expression = Expr("eq", BOOL, (x, const("+inf", FLOAT)))
    false = const(False, BOOL)
    registry = admit_rules(spec, RootPair(expression, false, const(0)))
    ground = [r for r in registry.rules if "observation_ground" in r.id]
    assert ground and all(not any(e.op == "pvar" for e in (r.lhs, r.rhs)) for r in ground)
    assert saturate(expression, false, registry, spec.limits).merged == (True,)
    infinite = Expr("eq", BOOL, (const("+inf", FLOAT), const("+inf", FLOAT)))
    assert saturate(infinite, false, registry, spec.limits).merged == (False,)
    # The underlying universal equation is invalid over extended variables.
    encoder = Encoder(extended_float_variables=True)
    assert query(encoder.encode(expression), (), 1000).status == "sat"
