from hypothesis import given, strategies as st
import pytest

from etv.evaluator import SideRoles, eval_expr
from etv.model import FactContext, UnsupportedSemantics
from etv.schema import parse_expr


EMPTY_FACTS = FactContext(bindings={}, assumptions=(), disjoint_groups=())
EMPTY_ROLES = SideRoles(blocks={}, scalars={})


@given(k=st.integers(min_value=0, max_value=100_000), n=st.integers(min_value=1, max_value=1024))
def test_div_rem_flatten_identity_over_bounded_i32(k, n):
    expr = parse_expr(
        {
            "op": "iadd",
            "args": [
                {"op": "imul", "args": [{"op": "idiv", "args": [{"var": "k"}, {"var": "n"}]}, {"var": "n"}]},
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
