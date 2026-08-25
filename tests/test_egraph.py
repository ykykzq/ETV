from etv.egraph import EGraph
from etv.ir import Expr, Sort
from etv.model import Limits
from etv.z3_validator import validated_builtin_rules


def value(name):
    return Expr("input", data=name, sort=Sort.FLOAT)


def test_fma_and_mul_add_saturate_to_same_eclass():
    a, b, c = value("a"), value("b"), value("c")
    lhs = Expr("fadd", args=(c, Expr("fmul", args=(a, b), sort=Sort.FLOAT)), sort=Sort.FLOAT)
    rhs = Expr("fma", args=(a, b, c), sort=Sort.FLOAT)
    graph = EGraph()
    lhs_root = graph.add_expr(lhs)
    rhs_root = graph.add_expr(rhs)
    rules, _ = validated_builtin_rules()

    stats = graph.saturate(rules, Limits())

    assert graph.equivalent(lhs_root, rhs_root)
    assert stats["rule_applications"]["fma_def"] >= 1
    assert any(item.get("rule_id") == "fma_def" for item in graph.merge_log)


def test_rebuild_propagates_child_equality_by_congruence():
    a, b = value("a"), value("b")
    graph = EGraph()
    a_root = graph.add_expr(a)
    b_root = graph.add_expr(b)
    neg_a = graph.add_expr(Expr("fneg", args=(a,), sort=Sort.FLOAT))
    neg_b = graph.add_expr(Expr("fneg", args=(b,), sort=Sort.FLOAT))

    graph.union(a_root, b_root, {"kind": "test_axiom", "evidence": "TRUSTED_AXIOM"})
    graph.rebuild()

    assert graph.equivalent(neg_a, neg_b)
    assert graph.backend["name"] == "egglog"
    assert graph.backend["version"] == "13.2.0"


def test_add_and_sub_do_not_merge():
    a, b = value("a"), value("b")
    graph = EGraph()
    add_root = graph.add_expr(Expr("fadd", args=(a, b), sort=Sort.FLOAT))
    sub_root = graph.add_expr(Expr("fsub", args=(a, b), sort=Sort.FLOAT))
    rules, _ = validated_builtin_rules()

    graph.saturate(rules, Limits())

    assert not graph.equivalent(add_root, sub_root)


def test_egglog_enode_budget_is_reported_before_saturation():
    a, b = value("a"), value("b")
    graph = EGraph()
    graph.add_expr(Expr("fadd", args=(a, b), sort=Sort.FLOAT))
    rules, _ = validated_builtin_rules()

    stats = graph.saturate(rules, Limits(max_enodes=1))

    assert stats["stop_reason"] == "ENODE_LIMIT"
    assert stats["iterations"] == 0
