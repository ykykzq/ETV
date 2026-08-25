import json
from dataclasses import replace
from pathlib import Path

import pytest

import etv.verify as verify_module
from etv.model import InputError, Status
from etv.prims import load_prims_program
from etv.schema import load_pair_spec
from etv.verify import verify_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


def test_prims_add_lifts_fixed_rank_symbolic_shapes():
    program = load_prims_program(FIXTURES / "programs/add_torch_prims.json")

    assert program.frontend == "torch_prims_json"
    assert program.frontend_version == "1"
    assert program.lanes == 1
    assert "var(rhs_dim0)" in program.programs.render()
    assert program.stores[0].value.op == "fadd"
    assert program.stores[0].value.args[1].op == "fmul"


def test_prims_frontend_rejects_implicit_broadcast(tmp_path):
    source = json.loads(
        (FIXTURES / "programs/add_torch_prims.json").read_text(encoding="utf-8")
    )
    source["nodes"][0] = {
        "name": "scaled_other",
        "op": "prims.mul",
        "args": ["rhs_alpha", "rhs_other"],
    }
    source["nodes"][1] = {
        "name": "result",
        "op": "prims.add",
        "args": ["rhs_input", "scaled_other"],
    }
    source["nodes"] = source["nodes"][:2]
    path = tmp_path / "implicit-broadcast.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(InputError, match="use prims.broadcast_in_dim explicitly"):
        load_prims_program(path)


def test_pair_spec_accepts_prims_without_launch_metadata():
    spec = load_pair_spec(FIXTURES / "specs/add_parametric_prims.json")

    assert spec.rhs_frontend.kind == "prims"
    assert spec.rhs_frontend.programs is None
    assert spec.rhs_frontend.function is None


def test_parametric_prims_equivalence_requires_fact_rewrites():
    report = verify_spec(FIXTURES / "specs/add_parametric_prims.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["inputs"]["frontends"]["rhs"] == {
        "name": "torch_prims_json",
        "version": "1",
    }
    egraph = report["proof"]["egraph"]
    assert egraph["initial_state"]["unmatched_root_pairs"] == 1
    assert egraph["after_fact_rewrites"]["unmatched_root_pairs"] == 0
    assert egraph["stats"]["phase"] == "FACT_DERIVED_RELATIONAL_REWRITES"
    assert len(egraph["stats"]["iteration_trace"]) >= 2
    assert egraph["stats"]["iteration_trace"][0]["rule_matches"]
    applied = {
        item["id"]
        for item in egraph["rule_application"]
        if item["used"]
    }
    assert any(rule_id.startswith("parametric_load_lhs") for rule_id in applied)
    assert any(rule_id.startswith("parametric_load_rhs") for rule_id in applied)
    assert {"parametric_store_lhs", "parametric_store_rhs"} <= applied


def test_parametric_prims_is_not_proved_when_relational_rules_are_removed(
    monkeypatch,
):
    original = verify_module.verify_parametric_pair

    def without_relational_rules(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, rewrite_rules=(), rewrite_admission=())

    monkeypatch.setattr(
        verify_module, "verify_parametric_pair", without_relational_rules
    )
    report = verify_spec(FIXTURES / "specs/add_parametric_prims.json")

    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "EGRAPH_NOT_EQUIVALENT"
