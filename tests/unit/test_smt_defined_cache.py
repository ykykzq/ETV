"""Definedness visits shared DAG nodes once without mixing encoder contexts."""

import z3

from etv.ir import BOOL, FLOAT, Expr, Type, const, var
from etv.smt import Encoder


def test_definedness_cost_tracks_unique_nodes_not_paths(monkeypatch):
    expr = var("x", FLOAT)
    depth = 16
    for _ in range(depth):
        expr = Expr("fadd", FLOAT, (expr, expr))
    original = Encoder._defined
    visits = 0

    def counted(self, node):
        nonlocal visits
        visits += 1
        return original(self, node)

    monkeypatch.setattr(Encoder, "_defined", counted)
    encoder = Encoder()
    defined = encoder.defined(expr)
    assert z3.is_true(z3.simplify(defined))
    assert visits == depth + 1
    assert encoder.defined(expr) is defined
    assert visits == depth + 1


def test_definedness_cache_isolated_by_environment_and_integer_mode():
    x = var("x", FLOAT)
    expression = Expr("fdiv", FLOAT, (const(1, FLOAT), x))
    zero = Encoder(environment={"x": z3.RealVal(0)})
    nonzero = Encoder(environment={"x": z3.RealVal(2)})
    assert z3.is_false(z3.simplify(zero.defined(expression)))
    assert z3.is_true(z3.simplify(nonzero.defined(expression)))
    assert z3.is_false(z3.simplify(zero.defined(expression)))

    byte = Type("int", 8)
    overflow = Expr("add", byte, (const(127, byte), const(1, byte)))
    wrapping = Encoder(mathematical_int=False)
    mathematical = Encoder(mathematical_int=True)
    assert z3.is_true(z3.simplify(wrapping.defined(overflow)))
    assert z3.is_false(z3.simplify(mathematical.defined(overflow)))


def test_cached_partial_branch_preserves_path_sensitive_definedness():
    invalid = Expr("fsub", FLOAT, (const("+inf", FLOAT), const("+inf", FLOAT)))
    branch = Expr("select", FLOAT, (var("take_invalid", BOOL), invalid, const(3, FLOAT)))
    encoder = Encoder()
    assert z3.is_false(z3.simplify(encoder.defined(invalid)))
    defined = encoder.defined(branch)
    solver = z3.Solver()
    solver.add(defined != z3.Not(encoder.encode(var("take_invalid", BOOL))))
    assert solver.check() == z3.unsat


def test_first_definedness_visit_retains_non_algebraic_encoding_fact():
    expr = Expr("exp", FLOAT, (var("x", FLOAT),))
    encoder = Encoder()
    assert encoder.exact
    assert z3.is_true(z3.simplify(encoder.defined(expr)))
    assert not encoder.exact
    assert z3.is_true(z3.simplify(encoder.defined(expr)))
    assert not encoder.exact
