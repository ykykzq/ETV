from fractions import Fraction

from hypothesis import given, strategies as st
import z3

from etv.ir import BOOL, FLOAT, Expr, Type, const
from etv.semantics import evaluate
from etv.smt import Encoder, query
from etv.extended_smt import real


@given(st.integers(2, 64), st.integers(-(2**65), 2**65), st.integers(-(2**65), 2**65))
def test_integer_operations_match_z3(bits, left, right):
    typ = Type("int", bits)
    for op in (
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "and",
        "or",
        "xor",
        "shrui",
        "mulhiui",
        "lt",
        "eq",
    ):
        expr = Expr(op, BOOL if op in ("lt", "eq") else typ, (const(left, typ), const(right, typ)))
        encoder = Encoder(64)
        if not z3.is_true(z3.simplify(encoder.defined(expr))):
            continue
        result = evaluate(expr, index_bits=64)
        encoded = z3.simplify(encoder.encode(expr))
        assert (z3.is_true(encoded) if expr.type == BOOL else encoded.as_long()) == result


@given(
    st.sampled_from((2, 8, 16, 32, 64)),
    st.sampled_from((2, 8, 16, 32, 64)),
    st.integers(-(2**65), 2**65),
)
def test_casts_match_z3(source, target, value):
    for op in ("extsi", "extui", "trunci", "sitofp", "uitofp"):
        typ = FLOAT if op.endswith("tofp") else Type("int", target)
        expr = Expr(op, typ, (const(value, Type("int", source)),))
        encoded = z3.simplify(Encoder(64).encode(expr))
        if typ == FLOAT:
            encoded = z3.simplify(real(encoded))
        actual = (
            Fraction(encoded.numerator_as_long(), encoded.denominator_as_long())
            if typ == FLOAT
            else encoded.as_long()
        )
        assert actual == evaluate(expr, index_bits=64)


def test_partial_float_domain():
    expr = Expr("fdiv", FLOAT, (const(1, FLOAT), const(0, FLOAT)))
    assert z3.is_false(z3.simplify(Encoder().defined(expr)))


def test_query_interpretation():
    x = z3.Int("x")
    assert query(x != x, (), 100).status == "unsat"
    answer = query(x == 3, (), 100)
    assert answer.status == "sat" and answer.model == (("x", "3"),)


def test_integer_memory_has_finite_signed_domain():
    expr = Expr("read", Type("int", 8), (const(0),), (("role", "A"),))
    encoder = Encoder()
    value = encoder.encode(expr)
    assert query(z3.Or(value < -128, value > 127), (), 1000).status == "unsat"
    boolean = Expr("read", BOOL, (const(0),), (("role", "B"),))
    assert z3.is_bool(encoder.encode(boolean))


@given(st.integers(-2048, 2048), st.integers(1, 17), st.sampled_from((2, 8, 16)))
def test_float_to_integer_poison_and_rounding(numerator, denominator, bits):
    value = Fraction(numerator, denominator)
    for op in ("fptosi", "fptoui"):
        expr = Expr(op, Type("int", bits), (const(value, FLOAT),))
        encoder = Encoder()
        low, high = (-(1 << (bits - 1)), 1 << (bits - 1)) if op == "fptosi" else (0, 1 << bits)
        expected_defined = low <= int(value) < high
        assert z3.is_true(z3.simplify(encoder.defined(expr))) == expected_defined
        if expected_defined:
            assert z3.simplify(encoder.encode(expr)).as_long() == evaluate(expr)
        else:
            import pytest
            from etv.errors import UnsupportedSemantics

            with pytest.raises(UnsupportedSemantics):
                evaluate(expr)


def test_logical_shift_and_high_product_edges():
    typ = Type("int", 8)
    assert evaluate(Expr("shrui", typ, (const(-1, typ), const(1, typ)))) == 127
    assert evaluate(Expr("mulhiui", typ, (const(-1, typ), const(-1, typ)))) == -2
    for amount in (-1, 8, 127):
        expr = Expr("shrui", typ, (const(-1, typ), const(amount, typ)))
        assert z3.is_false(z3.simplify(Encoder().defined(expr)))
