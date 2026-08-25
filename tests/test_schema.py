import json
from pathlib import Path

import pytest

from etv.model import InputError
from etv.schema import (
    load_internal_pair_spec,
    load_pair_spec,
    load_program,
    parse_expr,
    parse_rewrite_rule,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


def test_loads_example_program_and_spec():
    program = load_program(FIXTURES / "programs/add_ntops_2d.json")
    spec = load_internal_pair_spec(FIXTURES / "specs/add_proved.json")

    assert program.name == "ntops_add_2d"
    assert program.lanes == 128
    assert len(program.stores) == 1
    assert spec.pair_id == "add_ntops_2d_vs_inductor_linear"
    assert spec.facts.bindings["X"] == 128


def test_integer_cast_expression_requires_and_preserves_type_metadata():
    expression = parse_expr(
        {"op": "sext", "args": [{"var": "x"}], "data": ["i8", "i32"]}
    )

    assert expression.data == ("i8", "i32")
    assert expression.render() == "sext[i8->i32](var(x))"
    assert expression.to_json()["data"] == ("i8", "i32")


def test_integer_cast_expression_rejects_missing_type_metadata():
    with pytest.raises(InputError, match="cast data"):
        parse_expr({"op": "sext", "args": [{"var": "x"}]})


def test_rewrite_schema_preserves_exact_integer_cast_types():
    rule = parse_rewrite_rule(
        {
            "id": "typed_cast_rule",
            "kind": "algebraic",
            "lhs": {
                "op": "sext",
                "args": [{"match": "value"}],
                "data": ["i8", "i32"],
            },
            "rhs": {"match": "value"},
        }
    )

    assert rule.lhs.op == "sext"
    assert rule.lhs.data == ("i8", "i32")
    assert rule.lhs.match_data is True


def test_loads_side_specific_bindings_from_real_pair():
    spec = load_pair_spec(ROOT / "examples/add/pair.json")

    assert spec.facts.for_side("lhs").bindings["arg4"] == 8
    assert spec.facts.for_side("rhs").bindings["arg4"] == 128
    assert spec.lhs_frontend.kind == spec.rhs_frontend.kind == "ttir"
    assert "abi.Output" in spec.predicates.predicate_ids
    assert spec.facts.assumptions == ()
    assert spec.assumptions


def test_unknown_schema_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": "etv-semantic-program-v1",
                "name": "bad",
                "launch": {"programs": 1, "lanes": 1},
                "stores": [],
                "silently_ignored": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="unknown key"):
        load_program(path)


def test_unmapped_contract_role_is_rejected(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_proved.json").read_text(encoding="utf-8")
    )
    source["contract"]["output_role"] = "Missing"
    path = tmp_path / "bad-spec.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="output_role"):
        load_internal_pair_spec(path)


def test_loads_fact_gated_rewrite_with_llm_provenance(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_bad_compute.json").read_text(encoding="utf-8")
    )
    source["facts"]["assumptions"].append(
        "subtraction is equivalent to addition for this specialization"
    )
    source["rewrite_rules"] = [
        {
            "id": "specialized_sub_is_add",
            "kind": "trusted_fact",
            "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
            "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
            "requires": [
                {
                    "kind": "assumption",
                    "text": "subtraction is equivalent to addition for this specialization",
                }
            ],
            "provenance": {
                "generated_by": "llm",
                "generator": "test-model",
                "prompt_sha256": "a" * 64,
            },
        }
    ]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    spec = load_internal_pair_spec(path)

    declaration = spec.rewrite_rules[0]
    assert declaration.kind == "trusted_fact"
    assert declaration.generated_by == "llm"
    assert declaration.fact_requirements[0].evaluate(spec.facts)


def test_partitioning_requires_explicit_llm_enablement(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_proved.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_ntops_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_inductor_linear.json")
    source["partition"] = {"enabled": True}
    path = tmp_path / "partition-without-llm.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="requires llm.enabled"):
        load_internal_pair_spec(path)


def test_trusted_rewrite_without_fact_gate_is_rejected(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_bad_compute.json").read_text(encoding="utf-8")
    )
    source["rewrite_rules"] = [
        {
            "id": "unconditional_trusted_rule",
            "kind": "trusted_fact",
            "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
            "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
        }
    ]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="require at least one formal predicate gate"):
        load_internal_pair_spec(path)


def test_loads_symbolic_parameter_domain_and_constraint():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_parametric_shapes.json")

    assert spec.facts.parameters["a"].minimum == 1
    assert spec.facts.parameters["c"].maximum == 2**31 - 1
    assert spec.facts.constraints[0].sort.value == "bool"
    assert spec.contract.output_numel.data == "c"


def test_shared_binding_can_reference_symbolic_parameters(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_parametric_shapes.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_symbolic_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_symbolic_1d.json")
    source["facts"]["bindings"]["numel"] = {
        "op": "imul",
        "args": [{"var": "a"}, {"var": "b"}],
    }
    path = tmp_path / "symbolic-binding.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    spec = load_internal_pair_spec(path)

    assert spec.facts.bindings["numel"].render() == "imul(var(a), var(b))"


def test_production_pair_spec_rejects_internal_ir_frontends():
    with pytest.raises(InputError) as error:
        load_pair_spec(FIXTURES / "specs/add_proved.json")

    assert error.value.code == "TTIR_PAIR_REQUIRED"


@pytest.mark.parametrize("missing", ["kind", "function", "programs"])
def test_production_pair_spec_requires_explicit_ttir_metadata(tmp_path, missing):
    source = json.loads((ROOT / "examples/add/pair.json").read_text(encoding="utf-8"))
    source["metadata"]["lhs"] = str(ROOT / "examples/add/ttir/ntops_add.ttir")
    source["metadata"]["rhs"] = str(ROOT / "examples/add/ttir/torch_inductor_add.ttir")
    del source["metadata"]["frontends"]["lhs"][missing]
    path = tmp_path / "missing-frontend-field.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError):
        load_pair_spec(path)


def test_v2_loads_external_user_rewrite_with_predicate_gate(tmp_path):
    source = json.loads((ROOT / "examples/add/pair.json").read_text(encoding="utf-8"))
    source["metadata"]["lhs"] = str(ROOT / "examples/add/ttir/ntops_add.ttir")
    source["metadata"]["rhs"] = str(ROOT / "examples/add/ttir/torch_inductor_add.ttir")
    source["rewrites"] = [{"file": "rules.json"}]
    rules = {
        "format": "etv-rewrite-v1",
        "rules": [
            {
                "id": "user_sub_self",
                "kind": "trusted_predicate",
                "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "a"}]},
                "rhs": {"float": "0"},
                "requires": [{"kind": "predicate", "id": "abi.Output"}],
            }
        ],
    }
    spec_path = tmp_path / "pair.json"
    rule_path = tmp_path / "rules.json"
    spec_path.write_text(json.dumps(source), encoding="utf-8")
    rule_path.write_text(json.dumps(rules), encoding="utf-8")

    spec = load_pair_spec(spec_path)

    assert spec.rewrite_sources[0].kind == "user"
    assert len(spec.rewrite_sources[0].sha256) == 64
    assert spec.rewrite_rules[0].source == f"user:{rule_path.resolve()}"
    assert spec.rewrite_rules[0].predicate_requirements[0].evaluate(spec.facts)


def test_v2_llm_assumption_is_not_a_formal_rule_gate():
    spec = load_pair_spec(ROOT / "examples/add/pair.json")

    assert spec.assumptions
    assert spec.facts.assumptions == ()


def test_expression_operand_sorts_are_checked_at_schema_boundary():
    with pytest.raises(InputError, match="operands must be abstract_float"):
        parse_expr({"op": "fadd", "args": [1, 2]})
    with pytest.raises(InputError, match="offset must be an integer"):
        parse_expr(
            {
                "op": "load",
                "block": "arg0",
                "offset": {"float": "0"},
                "mask": True,
                "default": {"float": "0"},
            }
        )


def test_v2_rejects_non_integer_observation_size(tmp_path):
    source = json.loads((ROOT / "examples/add/pair.json").read_text(encoding="utf-8"))
    source["metadata"]["lhs"] = str(ROOT / "examples/add/ttir/ntops_add.ttir")
    source["metadata"]["rhs"] = str(ROOT / "examples/add/ttir/torch_inductor_add.ttir")
    source["observation"]["output_numel"] = {"float": "128"}
    path = tmp_path / "bad-output-numel.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="output_numel must be an integer"):
        load_pair_spec(path)


def test_v2_z3_custom_predicate_enters_formal_constraints(tmp_path):
    source = json.loads(
        (ROOT / "examples/add/pair_parametric.json").read_text(encoding="utf-8")
    )
    source["metadata"]["lhs"] = str(ROOT / "examples/add/ttir/parametric_2d_add.ttir")
    source["metadata"]["rhs"] = str(ROOT / "examples/add/ttir/parametric_1d_add.ttir")
    source["predicates"]["custom"] = [
        {
            "id": "shape.a_reflexive",
            "kind": "shape",
            "encoder": "z3_expr",
            "formula": {"op": "eq", "args": [{"var": "a"}, {"var": "a"}]},
        }
    ]
    path = tmp_path / "custom-predicate.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    spec = load_pair_spec(path)

    assert "shape.a_reflexive" in spec.predicates.predicate_ids
    assert spec.facts.constraints[-1].render() == "eq(var(a), var(a))"


def test_v2_custom_predicate_cannot_self_assert_proof(tmp_path):
    source = json.loads((ROOT / "examples/add/pair.json").read_text(encoding="utf-8"))
    source["metadata"]["lhs"] = str(ROOT / "examples/add/ttir/ntops_add.ttir")
    source["metadata"]["rhs"] = str(ROOT / "examples/add/ttir/torch_inductor_add.ttir")
    source["predicates"]["custom"] = [
        {
            "id": "layout.claimed",
            "kind": "relation",
            "encoder": "trusted",
            "status": "smt_proved",
        }
    ]
    path = tmp_path / "self-proved-predicate.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="cannot self-assert a proof"):
        load_pair_spec(path)
