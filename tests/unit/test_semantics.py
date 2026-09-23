from fractions import Fraction

from hypothesis import given, strategies as st
import z3

from etv.ir import BOOL, FLOAT, Expr, Type, const
from etv.semantics import evaluate
from etv.smt import Encoder, query


@given(st.integers(2, 64), st.integers(-(2**65), 2**65), st.integers(-(2**65), 2**65))
def test_integer_operations_match_z3(bits, left, right):
    typ = Type("int", bits)
    for op in ("add", "sub", "mul", "div", "rem", "lt", "eq"):
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
