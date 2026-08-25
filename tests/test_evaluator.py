from hypothesis import given, strategies as st
import pytest

from etv.evaluator import SideRoles, eval_expr
from etv.model import FactContext, UnsupportedSemantics
from etv.schema import parse_expr

EMPTY_FACTS = FactContext(bindings={}, assumptions=(), disjoint_groups=())
EMPTY_ROLES = SideRoles(blocks={}, scalars={})


@given(
    k=st.integers(min_value=0, max_value=100_000),
    n=st.integers(min_value=1, max_value=1024),
)
def test_div_rem_flatten_identity_over_bounded_i32(k, n):
    expr = parse_expr(
        {
            "op": "iadd",
            "args": [
                {
                    "op": "imul",
                    "args": [
                        {"op": "idiv", "args": [{"var": "k"}, {"var": "n"}]},
                        {"var": "n"},
                    ],
                },
                {"op": "irem", "args": [{"var": "k"}, {"var": "n"}]},
            ],
        }
    )

    assert eval_expr(expr, {"k": k, "n": n}, EMPTY_FACTS, EMPTY_ROLES) == k


@pytest.mark.parametrize(
    ("lhs", "rhs", "quotient", "remainder"),
    [(-7, 3, -2, -1), (7, -3, -2, 1), (-7, -3, 2, -1)],
)
def test_divsi_and_remsi_truncate_toward_zero(lhs, rhs, quotient, remainder):
    div = parse_expr({"op": "idiv", "args": [{"var": "lhs"}, {"var": "rhs"}]})
    rem = parse_expr({"op": "irem", "args": [{"var": "lhs"}, {"var": "rhs"}]})
    env = {"lhs": lhs, "rhs": rhs}

    assert eval_expr(div, env, EMPTY_FACTS, EMPTY_ROLES) == quotient
    assert eval_expr(rem, env, EMPTY_FACTS, EMPTY_ROLES) == remainder


def test_i32_overflow_never_silently_uses_unbounded_integer_semantics():
    expr = parse_expr({"op": "iadd", "args": [2**31 - 1, 1]})

    with pytest.raises(UnsupportedSemantics) as error:
        eval_expr(expr, {}, EMPTY_FACTS, EMPTY_ROLES)
    assert error.value.code == "INTEGER_OVERFLOW"


@pytest.mark.parametrize(
    ("op", "data", "value", "expected"),
    [
        ("sext", ["i8", "i32"], 255, -1),
        ("zext", ["i8", "i32"], -1, 255),
        ("trunc", ["i32", "i8"], 255, -1),
        ("trunc", ["i8", "i1"], 2, False),
    ],
)
def test_integer_casts_use_explicit_bitwidth_semantics(op, data, value, expected):
    expr = parse_expr({"op": op, "args": [{"var": "value"}], "data": data})

    assert eval_expr(expr, {"value": value}, EMPTY_FACTS, EMPTY_ROLES) == expected


def test_target_dependent_index_cast_is_not_assumed_to_be_identity():
    expr = parse_expr(
        {"op": "index_cast", "args": [{"var": "value"}], "data": ["i32", "index"]}
    )

    with pytest.raises(UnsupportedSemantics) as error:
        eval_expr(expr, {"value": 7}, EMPTY_FACTS, EMPTY_ROLES)
    assert error.value.code == "INTEGER_CAST_UNSUPPORTED"
