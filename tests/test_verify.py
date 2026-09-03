import json
from dataclasses import replace
from pathlib import Path

import pytest
import z3

from etv.llm import LLMAssistance
from etv.ir import Expr, Sort, int_const
from etv.model import ProofLevel, Status
from etv.parametric import ParametricFailure, SMTContext
from etv.reporting import write_report
from etv.rules import PredicateRequirement, Rule, node, var
from etv.schema import load_internal_pair_spec, load_program, parse_rewrite_rule
from etv.verify import (
    _definedness_issue,
    verify_internal_pair,
    verify_internal_spec as verify_spec,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("add_proved.json", Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT"),
        ("add_fma_proved.json", Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT"),
        ("add_bad_stride.json", Status.DISPROVED, "OUTPUT_ADDRESS_MISMATCH"),
        ("add_bad_mask.json", Status.DISPROVED, "MASK_MISMATCH"),
        ("add_bad_compute.json", Status.DISPROVED, "COMPUTE_MISMATCH"),
        ("add_missing_alias.json", Status.UNKNOWN, "MISSING_ALIAS_FACT"),
    ],
)
def test_end_to_end_verdicts(name, status, reason):
    report = verify_spec(FIXTURES / "specs" / name)

    assert report["status"] == status.value
    assert report["reason"] == reason


def test_compute_counterexample_is_replayable():
    report = verify_spec(FIXTURES / "specs/add_bad_compute.json")
    witness = report["counterexample"]

    assert witness["kind"] == "ABSTRACT_VALUE_MODEL"
    assert witness["lhs_value"] == "2"
    assert witness["rhs_value"] == "0"


def test_machine_report_is_deterministic():
    path = FIXTURES / "specs/add_fma_proved.json"

    assert verify_spec(path) == verify_spec(path)


def _pair_with_lhs_offset_cast():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_proved.json")
    lhs = load_program(spec.lhs_path)
    rhs = load_program(spec.rhs_path)
    store = lhs.stores[0]
    cast = Expr(
        "sext",
        args=(store.offset,),
        data=("i8", "i32"),
        sort=Sort.INT,
    )
    return spec, replace(lhs, stores=(replace(store, offset=cast),)), rhs


def test_bounded_verifier_does_not_silently_discharge_integer_cast():
    spec, lhs, rhs = _pair_with_lhs_offset_cast()

    report = verify_internal_pair(spec, lhs, rhs)

    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "CAST_EQUIVALENCE_NOT_REWRITTEN"
    cast_block = next(block for block in report["blocks"] if block["kind"] == "CAST")
    assert cast_block["status"] == Status.UNKNOWN.value
    assert "sext[i8->i32]" in cast_block["details"]["unresolved"][0]["expression"]


def test_bounded_verifier_does_not_silently_discharge_integer_to_float_cast():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_proved.json")
    lhs = load_program(spec.lhs_path)
    rhs = load_program(spec.rhs_path)
    store = lhs.stores[0]
    cast = Expr(
        "sitofp",
        args=(int_const(1),),
        data=("i32", "f32"),
        sort=Sort.FLOAT,
    )
    lhs = replace(lhs, stores=(replace(store, value=cast),))

    report = verify_internal_pair(spec, lhs, rhs)

    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "CAST_EQUIVALENCE_NOT_REWRITTEN"
    cast_block = next(block for block in report["blocks"] if block["kind"] == "CAST")
    assert "sitofp[i32->f32]" in cast_block["details"]["unresolved"][0]["expression"]


def test_bounded_verifier_reports_unmapped_load_as_unknown():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_proved.json")
    lhs = load_program(spec.lhs_path)
    rhs = load_program(spec.rhs_path)
    predicates = replace(
        spec.predicates,
        abi=tuple(role for role in spec.predicates.abi if role.logical == "Output"),
    )

    report = verify_internal_pair(replace(spec, predicates=predicates), lhs, rhs)

    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "MISSING_ROLE"


def test_bounded_verifier_uses_explicit_cast_rewrite_and_reports_trust():
    spec, lhs, rhs = _pair_with_lhs_offset_cast()
    value = var("value")
    declaration = Rule(
        "specialized_sext_identity",
        node(
            "sext",
            value,
            data=("i8", "i32"),
            match_data=True,
            sort=Sort.INT,
        ),
        value,
        ProofLevel.TRUSTED_AXIOM,
        "pair_predicate",
        "the bounded offset specialization makes sext identity",
        kind="trusted_fact",
        source="test",
        predicate_requirements=(
            PredicateRequirement(kind="assumption", value=spec.facts.assumptions[0]),
        ),
    )
    spec = replace(spec, rewrite_rules=(declaration,))

    report = verify_internal_pair(spec, lhs, rhs)

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["cast_rewrites"]["rule_matches"][declaration.rule_id] >= 1
    assert report["soundness"]["level"] == "conditional_on_unverified_rewrites"


def test_parametric_encoder_uses_integer_cast_bitwidth_semantics():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_parametric_shapes.json")
    value = z3.Int("cast_value")
    expression = Expr(
        "zext",
        args=(Expr("var", data="cast_value", sort=Sort.INT),),
        data=("i8", "i32"),
        sort=Sort.INT,
    )

    term = SMTContext(spec, {}).term(expression, {"cast_value": value})
    solver = z3.Solver()
    solver.add(value == -1, term.value != 255)

    assert solver.check() == z3.unsat


def test_parametric_encoder_rejects_target_dependent_index_cast():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_parametric_shapes.json")
    expression = Expr(
        "index_cast",
        args=(Expr("var", data="cast_value", sort=Sort.INT),),
        data=("i32", "index"),
        sort=Sort.INT,
    )

    with pytest.raises(ParametricFailure) as error:
        SMTContext(spec, {}).term(expression, {"cast_value": z3.Int("cast_value")})
    assert error.value.reason == "SYMBOLIC_INTEGER_CAST_UNSUPPORTED"


def test_all_admitted_rules_enter_the_unified_egglog_ruleset():
    report = verify_spec(FIXTURES / "specs/add_fma_proved.json")
    egraph = report["proof"]["egraph"]

    assert egraph["stats"]["backend"] == {"name": "egglog", "version": "13.2.0"}
    assert set(egraph["admitted_rule_ids"]) == {
        item["id"]
        for item in report["proof"]["rule_admission"]
        if item["status"] == "proved"
    }
    assert "fma_def" in egraph["stats"]["rule_matches"]


def test_writes_json_and_markdown_artifacts(tmp_path):
    report = verify_spec(FIXTURES / "specs/add_proved.json")

    write_report(report, tmp_path)

    assert (tmp_path / "report.json").is_file()
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Status: **PROVED**" in markdown
    assert "Soundness: `formal_under_declared_predicates`" in markdown
    assert "LLM-only context" in markdown
    assert "Rewrite registry" in markdown
    assert "It does not prove" in markdown


def test_partial_float_operations_require_proved_domains(tmp_path):
    value = Expr("input", data="X", sort=Sort.FLOAT)
    one = Expr("const_float", data=(1, 1), sort=Sort.FLOAT)

    assert _definedness_issue(Expr("fdiv", args=(value, one), sort=Sort.FLOAT)) is None
    assert (
        _definedness_issue(Expr("fdiv", args=(one, value), sort=Sort.FLOAT))[
            "operation"
        ]
        == "fdiv"
    )
    assert (
        _definedness_issue(Expr("fsqrt", args=(value,), sort=Sort.FLOAT))["operation"]
        == "fsqrt"
    )

    spec = json.loads((FIXTURES / "specs/add_proved.json").read_text(encoding="utf-8"))
    lhs = json.loads(
        (FIXTURES / "programs/add_ntops_2d.json").read_text(encoding="utf-8")
    )
    rhs = json.loads(
        (FIXTURES / "programs/add_inductor_linear.json").read_text(encoding="utf-8")
    )
    lhs["stores"][0]["value"] = {
        "op": "fdiv",
        "args": [lhs["stores"][0]["value"], {"scalar": "nt_alpha"}],
    }
    rhs["stores"][0]["value"] = {
        "op": "fdiv",
        "args": [
            rhs["stores"][0]["value"],
            {
                "op": "load",
                "block": "in_ptr1",
                "offset": 0,
                "mask": True,
                "default": {"float": "0"},
            },
        ],
    }
    spec["lhs"] = "lhs.json"
    spec["rhs"] = "rhs.json"
    (tmp_path / "lhs.json").write_text(json.dumps(lhs), encoding="utf-8")
    (tmp_path / "rhs.json").write_text(json.dumps(rhs), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    report = verify_spec(spec_path)
    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "FLOAT_DEFINEDNESS_NOT_PROVED"


def _write_trusted_rewrite_spec(tmp_path, include_fact):
    source = json.loads(
        (FIXTURES / "specs/add_bad_compute.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_ntops_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_inductor_bad_compute.json")
    fact = "subtraction is equivalent to addition for this specialization"
    if include_fact:
        source["facts"]["assumptions"].append(fact)
    source["rewrite_rules"] = [
        {
            "id": "specialized_sub_is_add",
            "kind": "trusted_fact",
            "statement": "specialized fsub(a, b) == fadd(a, b)",
            "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
            "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
            "requires": [{"kind": "assumption", "text": fact}],
        }
    ]
    path = tmp_path / "trusted.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_fact_gated_trusted_rewrite_is_used_and_audited(tmp_path):
    report = verify_spec(_write_trusted_rewrite_spec(tmp_path, include_fact=True))

    assert report["status"] == Status.PROVED.value
    uses = report["proof"]["egraph"]["trusted_rule_uses"]
    assert uses[0]["id"] == "specialized_sub_is_add"
    assert uses[0]["matches"] > 0
    assert "without SMT or formal validation" in uses[0]["validation"]["warning"]
    assert "TRUSTED_AXIOM" in next(
        block["proof_levels"]
        for block in report["blocks"]
        if block["kind"] == "COMPUTE"
    )
    assert report["soundness"]["level"] == "conditional_on_unverified_rewrites"
    assert (
        "unverified rewrite rule: specialized_sub_is_add"
        in report["soundness"]["conditional_on"]
    )


def test_fact_gated_rewrite_is_skipped_when_requirement_is_missing(tmp_path):
    report = verify_spec(_write_trusted_rewrite_spec(tmp_path, include_fact=False))

    assert report["status"] == Status.DISPROVED.value
    admission = next(
        item
        for item in report["proof"]["rule_admission"]
        if item["id"] == "specialized_sub_is_add"
    )
    assert admission["status"] == "skipped"
    assert admission["validation"]["predicate_checks"][0]["satisfied"] is False


def test_parametric_2d_to_1d_shape_relation_is_proved():
    report = verify_spec(FIXTURES / "specs/add_parametric_shapes.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["proof"]["parametric_domain"]["complete_for_parameter_domain"] is True
    assert report["proof"]["egraph"]["root_pairs"] == 1
    assert report["proof"]["egraph"]["initial_state"]["unmatched_root_pairs"] == 1
    assert (
        report["proof"]["egraph"]["after_predicate_rewrites"]["unmatched_root_pairs"]
        == 0
    )
    assert report["proof"]["egraph"]["stats"]["iterations"] > 0
    assert all(
        check["result"] in {"sat", "unsat"}
        for check in report["proof"]["parametric_domain"]["checks"]
    )
    assert "PARAMETRIC_SMT" in next(
        block["proof_levels"]
        for block in report["blocks"]
        if block["kind"] == "ADDRESS"
    )


def test_parametric_shape_relation_is_a_required_premise(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_parametric_shapes.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_symbolic_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_symbolic_1d.json")
    source["facts"]["constraints"] = []
    path = tmp_path / "missing-relation.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    report = verify_spec(path)

    assert report["status"] in {Status.DISPROVED.value, Status.UNKNOWN.value}
    assert report["reason"] != "OBSERVABLE_MEMORY_EQUIVALENT"


def test_unproved_algebraic_rule_can_be_applied_under_best_effort_policy(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_bad_compute.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_ntops_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_inductor_bad_compute.json")
    fact = "this specialization admits subtraction as addition"
    source["facts"]["assumptions"].append(fact)
    source["rule_policy"] = {"algebraic_validation": "best_effort"}
    source["rewrite_rules"] = [
        {
            "id": "conditional_unproved_sub_is_add",
            "kind": "algebraic",
            "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
            "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
            "requires": [{"kind": "assumption", "text": fact}],
        }
    ]
    path = tmp_path / "best-effort.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    report = verify_spec(path)

    assert report["status"] == Status.PROVED.value
    admission = next(
        item
        for item in report["proof"]["rule_admission"]
        if item["id"] == "conditional_unproved_sub_is_add"
    )
    assert admission["status"] == "admitted_unverified"
    assert admission["validation"]["result"] == "sat"
    usage = next(
        item
        for item in report["proof"]["egraph"]["rule_application"]
        if item["id"] == "conditional_unproved_sub_is_add"
    )
    assert usage["used"] is True
    assert report["proof"]["egraph"]["unverified_rule_uses"]


def test_llm_generated_conditional_rule_enters_second_saturation(tmp_path, monkeypatch):
    source = json.loads(
        (FIXTURES / "specs/add_bad_compute.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_ntops_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_inductor_bad_compute.json")
    fact = "this specialization admits subtraction as addition"
    source["facts"]["assumptions"].append(fact)
    source["llm"] = {"enabled": True}
    path = tmp_path / "llm-assisted.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    rule = parse_rewrite_rule(
        {
            "id": "llm_conditional_sub_is_add",
            "kind": "trusted_fact",
            "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
            "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
            "requires": [{"kind": "assumption", "text": fact}],
            "provenance": {
                "generated_by": "llm",
                "generator": "fake-deepseek",
                "prompt_sha256": "a" * 64,
            },
        },
        "test.rule",
    )

    def fake_propose_rules(spec, candidates):
        assert candidates
        return LLMAssistance(
            (rule,),
            {
                "enabled": True,
                "provider": "deepseek",
                "configured_model": "fake-deepseek",
                "generated_rule_ids": [rule.rule_id],
                "calls": [],
            },
        )

    monkeypatch.setattr("etv.verify.propose_rules", fake_propose_rules)

    report = verify_spec(path)

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["llm_assistance"]["generated_rule_ids"] == [rule.rule_id]
    assert len(report["proof"]["egraph"]["stats"]["phases"]) == 2
    usage = next(
        item
        for item in report["proof"]["egraph"]["rule_application"]
        if item["id"] == rule.rule_id
    )
    assert usage["used"] is True
    assert usage["admission_status"] == "admitted_unverified"
