import json
from pathlib import Path

import pytest

from etv.model import InputError
from etv.schema import load_pair_spec, load_program


ROOT = Path(__file__).resolve().parents[1]


def test_loads_example_program_and_spec():
    program = load_program(ROOT / "examples/programs/add_ntops_2d.json")
    spec = load_pair_spec(ROOT / "examples/specs/add_proved.json")

    assert program.name == "ntops_add_2d"
    assert program.lanes == 128
    assert len(program.stores) == 1
    assert spec.pair_id == "add_ntops_2d_vs_inductor_linear"
    assert spec.facts.bindings["X"] == 128


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
    source = json.loads((ROOT / "examples/specs/add_proved.json").read_text(encoding="utf-8"))
    source["contract"]["output_role"] = "Missing"
    path = tmp_path / "bad-spec.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="output_role"):
        load_pair_spec(path)


def test_loads_fact_gated_rewrite_with_llm_provenance(tmp_path):
    source = json.loads((ROOT / "examples/specs/add_bad_compute.json").read_text(encoding="utf-8"))
    source["facts"]["assumptions"].append("subtraction is equivalent to addition for this specialization")
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


def test_trusted_rewrite_without_fact_gate_is_rejected(tmp_path):
    source = json.loads((ROOT / "examples/specs/add_bad_compute.json").read_text(encoding="utf-8"))
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
