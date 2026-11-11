"""Independent counterexamples and proof-boundary checks for tagged infinities."""

import importlib

import pytest
import z3

from etv.ir import BOOL, FLOAT, Expr, const, var
from etv.pairspec import parse_pair_spec
from etv.rewrites import Rule, admit_candidates
from etv.semantics import evaluate
from etv.smt import Encoder
from test_contracts import minimal, write
from test_observation_encoding import program


def verify_values(tmp_path, monkeypatch, lhs, rhs):
    data = minimal(tmp_path)
    data["metadata"]["limits"].update(smt_timeout_ms=2000, egraph_timeout_ms=10000)
    path = write(tmp_path, data)
    spec = parse_pair_spec(path)
    pair = program(spec, lhs, rhs)
    module = importlib.import_module("etv.verify")
    monkeypatch.setattr(module, "_programs", lambda *_: pair)
    return module.verify(path)


def floating(op, *args):
    return Expr(op, FLOAT, args)


def invalid_expression(kind):
    positive, negative, zero = const("+inf", FLOAT), const("-inf", FLOAT), const(0, FLOAT)
    return {
        "inf_minus_inf": floating("fsub", positive, positive),
        "opposite_infinities": floating("fadd", positive, negative),
        "zero_times_inf": floating("fmul", zero, positive),
        "inf_over_inf": floating("fdiv", positive, positive),
    }[kind]


@pytest.mark.parametrize(
    "kind", ["inf_minus_inf", "opposite_infinities", "zero_times_inf", "inf_over_inf"]
)
def test_invalid_infinity_arithmetic_never_proves_even_identical_roots(tmp_path, monkeypatch, kind):
    expr = invalid_expression(kind)
    report = verify_values(tmp_path, monkeypatch, expr, expr)
    assert report.status == "UNKNOWN", report
    assert report.reason == "FLOAT_DEFINEDNESS_NOT_PROVED"
    assert not report.equalities


@pytest.mark.parametrize("op", ["ordered", "unordered", "isnan"])
def test_float_predicates_cannot_hide_invalid_operand(tmp_path, monkeypatch, op):
    bad = invalid_expression("inf_minus_inf")
    args = (bad,) if op == "isnan" else (const(3, FLOAT), bad)
    predicate = Expr(op, BOOL, args)
    value = floating("select", predicate, const(1, FLOAT), const(0, FLOAT))
    report = verify_values(tmp_path, monkeypatch, value, value)
    assert report.status == "UNKNOWN", report
    assert report.reason == "FLOAT_DEFINEDNESS_NOT_PROVED"


def test_inactive_select_branch_does_not_require_invalid_float_value(tmp_path, monkeypatch):
    chosen = floating(
        "select", const(False, BOOL), invalid_expression("zero_times_inf"), const(7, FLOAT)
    )
    report = verify_values(tmp_path, monkeypatch, chosen, const(7, FLOAT))
    assert report.status == "PROVED", report


@pytest.mark.parametrize("op,identity", [("fmax", "-inf"), ("fmin", "+inf")])
def test_infinite_reduction_padding_identity_is_proved(tmp_path, monkeypatch, op, identity):
    x = var("finite.input", FLOAT)
    padded = floating(op, x, const(identity, FLOAT))
    report = verify_values(tmp_path, monkeypatch, padded, x)
    assert report.status == "PROVED", report
    assert report.execution_scope == "ttir-only"
    assert all(rule.admission == "formal" for rule in report.rewrites if rule.used)


def test_negative_infinity_exponential_is_exact_zero(tmp_path, monkeypatch):
    value = floating("exp", floating("fsub", const("-inf", FLOAT), const(1000, FLOAT)))
    assert evaluate(value) == 0
    report = verify_values(tmp_path, monkeypatch, value, const(0, FLOAT))
    assert report.status == "PROVED", report


def test_infinity_is_not_an_arbitrary_large_finite_sentinel():
    x = var("unbounded.finite.input", FLOAT)
    positive, negative = const("+inf", FLOAT), const("-inf", FLOAT)
    encoder = Encoder()
    solver = z3.Solver()
    solver.add(
        z3.Or(
            encoder.encode(Expr("ge", BOOL, (x, positive))),
            encoder.encode(Expr("le", BOOL, (x, negative))),
        )
    )
    assert solver.check() == z3.unsat
    for huge in (10**1000, -(10**1000)):
        finite = const(huge, FLOAT)
        assert evaluate(Expr("lt", BOOL, (finite, positive))) is True
        assert evaluate(Expr("gt", BOOL, (finite, negative))) is True


def test_generic_real_cancellation_rule_is_rejected_for_infinite_substitutions(tmp_path):
    data = minimal(tmp_path)
    data["metadata"]["limits"]["smt_timeout_ms"] = 2000
    spec = parse_pair_spec(write(tmp_path, data))
    x = Expr("pvar", FLOAT, attrs=(("name", "x"),))
    rule = Rule(
        "unsafe.cancel", "algebraic", floating("fsub", x, x), const(0, FLOAT), (), "test", ""
    )
    registry = admit_candidates((rule,), spec)
    assert not registry.rules
    assert registry.facts[0].admission == "rejected"


def test_opposite_infinite_outputs_have_replayable_counterexample(tmp_path, monkeypatch):
    report = verify_values(tmp_path, monkeypatch, const("+inf", FLOAT), const("-inf", FLOAT))
    assert (report.status, report.reason) == ("DISPROVED", "COMPUTE_MISMATCH"), report
    assert {report.counterexample.lhs_value, report.counterexample.rhs_value} == {"+inf", "-inf"}


def test_finite_input_isinf_is_false_only_under_current_declared_input_domain():
    x = var("finite.input", FLOAT)
    pred = Expr("eq", BOOL, (floating("absf", x), const("+inf", FLOAT)))
    finite = Encoder()
    solver = z3.Solver()
    solver.add(finite.encode(pred))
    assert solver.check() == z3.unsat
    extended = Encoder(extended_float_variables=True)
    solver = z3.Solver()
    solver.add(extended.defined(pred), extended.encode(pred))
    assert solver.check() == z3.sat
