from etv.z3_validator import validated_builtin_rules


def test_every_builtin_rule_is_admitted_by_z3_unsat_check():
    accepted, results = validated_builtin_rules()

    assert len(accepted) == len(results)
    assert results
    assert all(result["status"] == "proved" for result in results)
    assert all(result["validation"]["result"] == "unsat" for result in results)
    assert all(result["validation"]["query_sha256"] for result in results)
