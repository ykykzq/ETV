from etv.model import ProofLevel
from etv.rules import Rule, node, var
from etv.z3_validator import validate_rule, validated_builtin_rules


def test_every_builtin_rule_is_admitted_by_z3_unsat_check():
    accepted, results = validated_builtin_rules()

    assert len(accepted) == len(results)
    assert results
    assert all(result["status"] == "proved" for result in results)
    assert all(result["validation"]["result"] == "unsat" for result in results)
    assert all(result["validation"]["query_sha256"] for result in results)


def test_unsound_pair_algebraic_rule_is_rejected():
    a, b = var("a"), var("b")
    declaration = Rule(
        "unsound_sub_is_add",
        node("fsub", a, b),
        node("fadd", a, b),
        ProofLevel.ALGEBRAIC,
        "z3_real_unsat",
        "a - b == a + b",
        source="test",
    )

    result = validate_rule(declaration)

    assert result["status"] == "rejected"
    assert result["validation"]["result"] == "sat"
    assert result["validation"]["counterexample"]
