"""Strict export gates validate actual scope/artifact relationships, not labels alone."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


MODULE = Path(__file__).resolve().parents[2] / "tools/export_ttir_only.py"
SPEC = importlib.util.spec_from_file_location("export_ttir_only", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "EXPECTED_POPULATION", 3)
    corpus, run, source = (tmp_path / name for name in ("corpus", "run", "source"))
    code = source / "src/etv/verify.py"
    code.parent.mkdir(parents=True)
    code.write_text("# fixture implementation\n")
    rows, retained, baseline_rows, recaptures = [], [], [], []
    for index, status in enumerate(("PROVED", "UNKNOWN", "NOT_VERIFIED")):
        nodeid = f"tests/test_scope.py::test_case[{index}]"
        identity = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        directory = corpus / identity
        arguments = {
            "q": 0.25,
            "input": {"shape": [2], "stride": [1], "dtype": "torch.float32", "device": "cuda:0"},
        }
        case_status = {"nodeid": nodeid, "arguments": arguments, "status": "captured"}
        write(directory / "case_status.json", case_status)
        launches = []
        for side in ("lhs", "rhs"):
            artifact = directory / side / "launch-000.ttir"
            artifact.parent.mkdir()
            artifact.write_text("module {} // recorded " + side)
            launches.append(
                {
                    "side": side,
                    "ordinal": 0,
                    "step": 2,
                    "file": f"{side}/launch-000.ttir",
                    "function": "copy",
                    "sha256": exporter.digest(artifact),
                    "selected_runtime_launch": True,
                }
            )
        capture = {
            "format": "etv-runtime-capture-v1",
            "output_numel": 2,
            "source": {"commit": "pinned", "test_id": nodeid},
            "launches": launches,
            "input_roles": {},
            "outputs": {"lhs": {}, "rhs": {}},
            "diagnostics": ["unsupported host operation: aten.sort.stable"],
            "host_operations": [{"kind": "unsupported", "operator": "aten.sort.stable"}],
        }
        write(directory / "capture.json", capture)
        row = {
            "case_id": identity,
            "nodeid": nodeid,
            "status": status,
            "execution_scope": "ttir-only",
            "capture": case_status,
            "diagnostics": [],
            "pair_returncode": 0,
            "pair_status": "ready",
        }
        if status == "PROVED":
            row["reason"] = "OBSERVABLE_MEMORY_EQUIVALENT"
        elif status == "UNKNOWN":
            row["reason"] = "TTIR_OP_UNSUPPORTED"
        else:
            row["pair_status"] = "needs_review"
            row["diagnostics"] = [exporter.SCOPE_DEPENDENCY + "input rhs arg0 storage=sort-values"]
        rows.append(row)
        retained.append(
            {
                "case_id": identity,
                "directory": identity,
                "nodeid": nodeid,
                "parameters": {"index": index},
            }
        )
        baseline_rows.append(
            {
                "case_id": identity,
                "nodeid": nodeid,
                "status": "NOT_VERIFIED",
                "category": "外部输入映射缺失",
                "capture": case_status,
            }
        )
        recaptures.append(
            {"case_id": identity, "enriched": True, "capture_origin": "new_runtime_host_capture"}
        )
    suite = {"source_commit": "pinned", "selected": [row["nodeid"] for row in rows]}
    write(corpus / "suite.lock.json", suite)
    write(corpus / "retained.json", retained)
    write(
        corpus / "recapture-audit.json",
        {"source_commit": "pinned", "full_selected": 3, "target_selected": 3, "cases": recaptures},
    )
    lock = {
        "parent_corpus_lock_sha256": "original-lock",
        "source_commit": "pinned",
        "files": {
            str(path.relative_to(corpus)): exporter.digest(path)
            for path in corpus.rglob("*")
            if path.is_file()
        },
    }
    write(corpus / "corpus.lock.json", lock)
    baseline = tmp_path / "baseline.json"
    write(
        baseline,
        {"summary": {"selected": 3, "corpus_lock_sha256": "original-lock"}, "cases": baseline_rows},
    )
    shutil.copytree(corpus, run / "work")
    for row in rows:
        identity = row["case_id"]
        capture = exporter.read(corpus / identity / "capture.json")
        programs = {
            side: [
                {
                    "id": side + ".0",
                    "file": side + "/launch-000.ttir",
                    "function": "copy",
                    "step": 0,
                }
            ]
            for side in ("lhs", "rhs")
        }
        pair_path = run / "work" / identity / "pair.json"
        write(
            pair_path,
            {
                "format": "etv-pair-v3",
                "metadata": {"pair_id": row["nodeid"], "execution_scope": "ttir-only"},
                "programs": programs,
                "predicates": {"constraints": [], "custom": [], "parameters": {}},
                "rewrites": [],
                "assumptions": {"for_llm": []},
                "observation": {"role": "Output", "numel": 2, "require_full_coverage": True},
            },
        )
        if row["status"] == "NOT_VERIFIED":
            continue
        report = {
            "pair_id": row["nodeid"],
            "execution_scope": "ttir-only",
            "status": row["status"],
            "reason": row["reason"],
            "physical_launch_count": 2,
            "execution_step_count": 2,
            "inputs": {
                "spec": {"kind": "spec", "path": "pair.json", "sha256": exporter.digest(pair_path)},
                "host_operations": [],
                "programs": [
                    {
                        "kind": "ttir",
                        "launch_id": launch["side"] + ".0",
                        "path": launch["file"],
                        "sha256": launch["sha256"],
                    }
                    for launch in capture["launches"]
                ],
            },
            "launches": [
                {
                    "kind": "ttir",
                    "id": side + ".0",
                    "side": side,
                    "file": "launch-000.ttir",
                    "function": "copy",
                    "step": 0,
                }
                for side in ("lhs", "rhs")
            ],
            "soundness": {"trusted_axioms": []},
        }
        write(run / "analysis" / identity / "report.json", report)
    summary = {
        "selected": 3,
        "execution_scope": "ttir-only",
        "verdicts": dict(Counter(row["status"] for row in rows)),
        "cases": rows,
        "corpus_lock_sha256": exporter.digest(corpus / "corpus.lock.json"),
    }
    write(run / "analysis/summary.json", summary)
    write(run / "completed.json", {key: value for key, value in summary.items() if key != "cases"})
    write(
        run / "run.json",
        {
            "selected": 3,
            "execution_scope": "ttir-only",
            "source_sha256": exporter.source_hashes(source),
            "corpus_lock_sha256": summary["corpus_lock_sha256"],
        },
    )
    return run, corpus, baseline, source


def export(experiment):
    run, corpus, baseline, source = experiment
    return exporter.export(run, corpus, baseline, source_root=source)


def test_strict_export_audits_scope_hashes_and_ignores_excluded_host_diagnostics(experiment):
    result = export(experiment)
    assert result["format"] == "etv-ttir-only-export-v1"
    assert result["execution_scope"] == "ttir-only"
    assert result["scope_audit"]["pairs_audited"] == 3
    assert result["scope_audit"]["reports_audited"] == 2
    assert result["scope_audit"]["report_host_inputs"] == 0
    assert result["proof_audit"]["counterexamples_replayed"] == 0
    assert result["proof_audit"]["whole_operator_numeric_counterexamples_confirmed"] is None
    assert result["cases"][1]["category"] == "TTIR 操作未支持"
    assert result["cases"][2]["category"] == "TTIR-only 范围外的宿主数据依赖 / 最终输出"
    assert result["cases"][0]["scope_audit"]["physical_launches"][0]["capture_step"] == 2
    assert result["cases"][0]["scope_audit"]["physical_launches"][0]["pair_step"] == 0


def test_complete_1365_population_is_not_optional(experiment, monkeypatch):
    monkeypatch.setattr(exporter, "EXPECTED_POPULATION", 1365)
    with pytest.raises(ValueError, match="complete fixed 1365"):
        export(experiment)


def test_nonready_partial_pair_remains_in_the_fixed_denominator(experiment):
    run, _, _, _ = experiment
    row = exporter.read(run / "analysis/summary.json")["cases"][2]
    path = run / "work" / row["case_id"] / "pair.json"
    pair = exporter.read(path)
    pair["programs"] = {"lhs": [], "rhs": []}
    write(path, pair)
    result = export(experiment)
    assert result["summary"]["selected"] == 3
    assert result["scope_audit"]["nonready_count"] == 1
    assert result["scope_audit"]["formal_reports_audited"] == 2
    assert result["scope_audit"]["pair_specs_audited"] == 3
    assert result["cases"][2]["scope_audit"]["omitted_capture_launches"] == 2


def test_excluded_capture_diagnostic_is_not_a_formal_host_semantics_reason():
    row = {
        "status": "NOT_VERIFIED",
        "control_flow": False,
        "diagnostics": ["unsupported host operation: aten.sort.stable"],
    }
    assert exporter.strict_category(row) == "TTIR-only 宿主事件已排除；采集证据待检查"


@pytest.mark.parametrize("diagnostics,expected", [
    (["duplicate physical endpoints for one role require review"], "重复物理端点 / 存储配对"),
    (["storage role has inconsistent element types"], "ABI 输出类型冲突"),
    (["INVALID_PAIRSPEC"], "PairSpec 契约尚未有效建立"),
    (["external input lacks a unique declared storage role",
      "duplicate physical endpoints for one role require review"], "外部输入映射缺失"),
])
def test_structural_control_flow_does_not_override_actual_pair_failure(diagnostics, expected):
    assert exporter.strict_category({
        "status": "NOT_VERIFIED", "control_flow": True, "diagnostics": diagnostics,
    }) == expected


@pytest.mark.parametrize("reason,diagnostic,expected", [
    ("TTIR_REGION_UNSUPPORTED", "scf.for", "分支 / 循环暂不支持"),
    ("TTIR_REGION_UNSUPPORTED", "tt.reduce", "复杂 reduction region 未支持"),
    ("TTIR_OP_UNSUPPORTED", "unsupported bitcast", "TTIR 操作未支持"),
    ("INPLACE_MEMORY_DEPENDENCY_UNPROVED", "cross-lane dependency", "原地 / 跨 lane 读写依赖未证明"),
])
def test_actual_formal_stop_overrides_structure_and_previous_pair_diagnostics(reason, diagnostic, expected):
    assert exporter.strict_category({
        "status": "UNKNOWN", "control_flow": True, "reason": reason,
        "diagnostic": diagnostic,
        "diagnostics": ["external input lacks a unique declared storage role"],
    }) == expected


def test_strict_baseline_can_describe_the_same_frozen_corpus(experiment):
    baseline_result = export(experiment)
    _, _, baseline_path, _ = experiment
    write(baseline_path, baseline_result)
    current = export(experiment)
    assert current["evidence"]["baseline_data_sha256"] == exporter.digest(baseline_path)
    assert current["baseline"] == baseline_result["summary"]
    assert current["baseline_execution_scope"] == "ttir-only"
    assert current["transitions"] == {"PROVED → PROVED": 1, "UNKNOWN → UNKNOWN": 1, "NOT_VERIFIED → NOT_VERIFIED": 1}


def test_same_corpus_exception_is_strict_and_source_bound(experiment):
    result = export(experiment)
    _, _, baseline_path, _ = experiment
    result["evidence"]["source_commit"] = "another-source"
    write(baseline_path, result)
    with pytest.raises(ValueError, match="parent frozen corpus"):
        export(experiment)


def test_target_membership_uses_baseline_failure_and_preserves_current_unknown():
    members = []
    for index, category in enumerate(exporter.TARGET_CATEGORIES.values()):
        members.append({
            "case_id": str(index), "baseline_status": "NOT_VERIFIED",
            "baseline_category": category, "status": "UNKNOWN",
            "category": "分支 / 循环暂不支持", "control_flow": True,
        })
    groups = exporter.target_groups(members)
    assert [group["selected"] for group in groups] == [1, 1, 1]
    assert groups[0]["transitions"] == {"NOT_VERIFIED → UNKNOWN": 1}
    assert groups[0]["unresolved_categories"] == {"分支 / 循环暂不支持": 1}


def encoded_pair(experiment):
    run, corpus, _, _ = experiment
    row = exporter.read(run / "analysis/summary.json")["cases"][0]
    identity = row["case_id"]
    pair_path = run / "work" / identity / "pair.json"
    pair = exporter.read(pair_path)
    capture = exporter.read(corpus / identity / "capture.json")
    capture["source"]["operator"] = "eq"
    capture["outputs"] = {side: {"output-storage": "Output"} for side in ("lhs", "rhs")}
    representation = {"kind": "boolean_numeric_01"}
    pair["metadata"]["target"] = {"boolean_storage": "canonical_i8"}
    pair["predicates"]["roles"] = {}
    for side in ("lhs", "rhs"):
        element = "abstract_float" if side == "lhs" else "bool"
        pair["predicates"]["roles"]["Output_" + side] = {"kind": "buffer", "element": element}
        representation[side] = {"role": "Output_" + side,
                                "encoding": "identity" if side == "lhs" else "bool_to_numeric_01"}
        next(launch for launch in capture["launches"] if launch["side"] == side)["slots"] = [
            {"runtime": {"storage_id": "output-storage",
                         "dtype": "torch.float32" if side == "lhs" else "torch.bool"}}
        ]
    pair["observation"]["value_representation"] = representation
    write(pair_path, pair)
    write(pair_path.parent / "capture.json", capture)
    write(pair_path.parent / "provenance.json", {
        "pair_sha256": exporter.digest(pair_path),
        "capture_sha256": exporter.digest(pair_path.parent / "capture.json"),
        "output_representation_evidence": {"operator": "eq", "value_representation": representation},
    })
    return row, pair_path, pair, capture, corpus / identity


def test_encoded_output_requires_bound_capture_and_explicit_report_contract(experiment):
    row, pair_path, _, capture, directory = encoded_pair(experiment)
    pair_audit = exporter.audit_pair_scope(pair_path, capture, directory, require_complete=True)
    contract = pair_audit["comparison_contract"]
    assert contract["kind"] == "logical-value-under-encoding"
    assert contract["physical_elements"] == {"lhs": "abstract_float", "rhs": "bool"}
    assert contract["bit_identical"] is False
    assert contract["output_dtype_equality_claimed"] is False
    report_path = experiment[0] / "analysis" / row["case_id"] / "report.json"
    report = exporter.read(report_path)
    row["reason"] = report["reason"] = "LOGICAL_BOOLEAN_OUTPUT_EQUIVALENT"
    report["inputs"]["spec"]["sha256"] = exporter.digest(pair_path)
    write(report_path, report)
    with pytest.raises(ValueError, match="comparison contract"):
        exporter.audit_report_scope(report_path, row, pair_path, pair_audit)
    report["comparison_contract"] = dict(contract)
    write(report_path, report)
    assert exporter.audit_report_scope(report_path, row, pair_path, pair_audit)["comparison_contract"] == contract
    report["comparison_contract"]["bit_identical"] = True
    write(report_path, report)
    with pytest.raises(ValueError, match="comparison contract"):
        exporter.audit_report_scope(report_path, row, pair_path, pair_audit)


@pytest.mark.parametrize("mutation,match", [
    (lambda pair, cap: pair["observation"].update(numel=1), "original full output"),
    (lambda pair, cap: cap["source"].update(operator="add"), "operator identity"),
    (lambda pair, cap: pair["predicates"]["roles"]["Output_rhs"].update(element="abstract_float"), "physical role type"),
    (lambda pair, cap: cap["launches"][1]["slots"][0]["runtime"].update(dtype="torch.float32"), "captured physical dtypes"),
    (lambda pair, cap: pair["metadata"]["target"].clear(), "storage premise"),
])
def test_encoded_output_cannot_hide_dtype_or_observation_changes(experiment, mutation, match):
    _, path, pair, capture, directory = encoded_pair(experiment)
    mutation(pair, capture)
    write(path, pair)
    with pytest.raises(ValueError, match=match):
        exporter.audit_pair_scope(path, capture, directory, require_complete=True)


@pytest.mark.parametrize("target", ["source", "runtime_arguments", "capture", "completion"])
def test_frozen_run_inputs_cannot_drift(experiment, target):
    run, corpus, baseline, source = experiment
    identity = exporter.read(run / "analysis/summary.json")["cases"][0]["case_id"]
    if target == "source":
        (source / "src/etv/verify.py").write_text("# changed\n")
    elif target == "runtime_arguments":
        path = run / "work" / identity / "case_status.json"
        value = exporter.read(path)
        value["arguments"]["q"] = 0.75
        write(path, value)
    elif target == "capture":
        (run / "work" / identity / "capture.json").write_text("{}")
    else:
        write(run / "completed.json", {"selected": 2})
    with pytest.raises(ValueError):
        export(experiment)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda p: p["metadata"].update(execution_scope="ttir-with-host-ops"),
            "explicitly TTIR-only",
        ),
        (lambda p: p["programs"]["lhs"][0].update(kind="host_op"), "non-TTIR"),
        (lambda p: p["programs"]["lhs"][0].update(file="synthetic.ttir"), "captured physical TTIR"),
        (lambda p: p["predicates"].update(custom=[{"host_sorted": True}]), "semantic premises"),
        (lambda p: p["observation"].update(numel=1), "original full output"),
    ],
)
def test_pair_scope_rejects_host_or_synthetic_programs(experiment, mutation, match):
    run, corpus, _, _ = experiment
    row = exporter.read(run / "analysis/summary.json")["cases"][0]
    identity = row["case_id"]
    pair_path = run / "work" / identity / "pair.json"
    pair = exporter.read(pair_path)
    mutation(pair)
    write(pair_path, pair)
    with pytest.raises(ValueError, match=match):
        exporter.audit_pair_scope(
            pair_path,
            exporter.read(corpus / identity / "capture.json"),
            corpus / identity,
            require_complete=True,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(execution_scope="ttir-with-host-ops"), "explicitly TTIR-only"),
        (lambda r: r["inputs"].update(host_operations=[{"kind": "host_capture"}]), "host input"),
        (lambda r: r["launches"][0].update(kind="host_op"), "host execution step"),
        (lambda r: r.update(execution_step_count=3), "launch counts disagree"),
        (lambda r: r["inputs"]["programs"][0].update(sha256="changed"), "captured physical TTIR"),
    ],
)
def test_report_scope_is_checked_independently_of_pair_labels(experiment, mutation, match):
    run, corpus, _, _ = experiment
    row = exporter.read(run / "analysis/summary.json")["cases"][0]
    identity = row["case_id"]
    pair_path = run / "work" / identity / "pair.json"
    pair_audit = exporter.audit_pair_scope(
        pair_path,
        exporter.read(corpus / identity / "capture.json"),
        corpus / identity,
        require_complete=True,
    )
    report_path = run / "analysis" / identity / "report.json"
    report = exporter.read(report_path)
    mutation(report)
    write(report_path, report)
    with pytest.raises(ValueError, match=match):
        exporter.audit_report_scope(report_path, row, pair_path, pair_audit)
