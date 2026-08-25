import json
from pathlib import Path

import pytest

from etv.model import InputError
from etv.schema import load_pair_spec, load_program

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


def test_loads_example_program_and_spec():
    program = load_program(FIXTURES / "programs/add_ntops_2d.json")
    spec = load_pair_spec(FIXTURES / "specs/add_proved.json")

    assert program.name == "ntops_add_2d"
    assert program.lanes == 128
    assert len(program.stores) == 1
    assert spec.pair_id == "add_ntops_2d_vs_inductor_linear"
    assert spec.facts.bindings["X"] == 128


def test_loads_side_specific_bindings_from_real_pair():
    spec = load_pair_spec(ROOT / "examples/add/pair.json")

    assert spec.facts.for_side("lhs").bindings["arg4"].render() == "var(a)"
    assert spec.facts.for_side("rhs").bindings["torch_dim0"].render() == "var(a)"
    assert spec.facts.for_side("rhs").bindings["torch_dim1"].render() == "var(b)"


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
        load_pair_spec(path)


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

    spec = load_pair_spec(path)

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
        load_pair_spec(path)


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

    with pytest.raises(InputError, match="require at least one fact gate"):
        load_pair_spec(path)


def test_loads_symbolic_parameter_domain_and_constraint():
    spec = load_pair_spec(FIXTURES / "specs/add_parametric_shapes.json")

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

    spec = load_pair_spec(path)

    assert spec.facts.bindings["numel"].render() == "imul(var(a), var(b))"
