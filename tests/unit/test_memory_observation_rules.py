from dataclasses import replace

from etv.eqsat import saturate, saturate_roots
from etv.ir import BOOL, FLOAT, Expr, RootPair, const
from etv.pairspec import parse_pair_spec
from etv.rewrites import RuleRegistry, admit_rules
from test_contracts import minimal, write


def read(role, index):
    return Expr("read", FLOAT, (const(index),), (("role", role),))


def isinf(value):
    return Expr("eq", BOOL, (Expr("absf", FLOAT, (value,)), const("+inf", FLOAT)))


def test_finite_memory_template_proves_more_than_sixteen_observations(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    roots = tuple(
        RootPair(isinf(read("Input", i)), const(False, BOOL), const(i)) for i in range(40)
    )
    registry = admit_rules(spec, roots)
    memory_rules = tuple(r for r in registry.rules if "observation_memory" in r.id)
    assert len(memory_rules) == 1
    assert all(f.admission == "formal" for f in registry.facts if "observation_memory" in f.id)
    only_memory = RuleRegistry(memory_rules, ())
    result = saturate_roots(roots, only_memory, replace(spec.limits, egraph_timeout_ms=10000))
    assert result.merged == (True,) * len(roots)

    # The variable matches an address under a fixed read constructor; it must
    # never become a floating wildcard matching an infinite computed value.
    for value in (const("+inf", FLOAT), const("-inf", FLOAT)):
        assert saturate(isinf(value), const(False, BOOL), only_memory, spec.limits).merged == (
            False,
        )


def test_memory_template_preserves_distinct_roles_and_addresses(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    for lhs, rhs in [(read("A", 0), read("B", 0)), (read("A", 0), read("A", 1))]:
        registry = admit_rules(spec, RootPair(lhs, rhs, const(0)))
        assert all(
            fact.admission == "rejected"
            for fact in registry.facts
            if "observation_memory" in fact.id
        )
        assert saturate(lhs, rhs, registry, spec.limits).merged == (False,)
