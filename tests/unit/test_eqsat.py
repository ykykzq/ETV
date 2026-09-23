from dataclasses import replace

from etv.eqsat import saturate
from etv.ir import FLOAT, Expr, RootPair, const, var
from etv.pairspec import Limits, parse_pair_spec
from etv.rewrites import admit_rules
from etv.rewrites import RuleRegistry
from test_contracts import minimal, write


def test_official_egglog_applies_formal_rule(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    x = var("x", FLOAT)
    lhs = Expr("fadd", FLOAT, (x, const(0, FLOAT)))
    rules = admit_rules(spec, RootPair(lhs, x, const(0)))
    assert all(f.admission == "formal" for f in rules.facts)
    result = saturate(lhs, x, rules, replace(Limits(), egraph_timeout_ms=10000))
    assert result.reason == "", result
    assert result.merged == (True,)
    assert dict(result.applications)["builtin.fadd_zero"] > 0


def test_unmerged_is_not_disproved(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    lhs, rhs = var("x", FLOAT), var("y", FLOAT)
    rules = admit_rules(spec, RootPair(lhs, rhs, const(0)))
    assert saturate(lhs, rhs, rules, Limits()).merged == (False,)


def test_resource_bounds():
    x, y = var("x", FLOAT), var("y", FLOAT)
    for limits in (
        replace(Limits(), max_enodes=1),
        replace(Limits(), max_iterations=0),
        replace(Limits(), egraph_timeout_ms=1),
    ):
        result = saturate(x, y, RuleRegistry((), ()), limits)
        assert result.reason == "RESOURCE_LIMIT"


def test_registry_is_local_to_run(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    x = var("x", FLOAT)
    lhs = Expr("fadd", FLOAT, (x, const(0, FLOAT)))
    rules = admit_rules(spec, RootPair(lhs, x, const(0)))
    assert saturate(lhs, x, rules, Limits()).merged == (True,)
    assert saturate(lhs, x, RuleRegistry((), ()), Limits()).merged == (False,)
