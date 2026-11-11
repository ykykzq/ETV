"""Recapture provenance and population integrity, independent of GPU execution."""

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "prepare_host_recapture.py"
SPEC = importlib.util.spec_from_file_location("prepare_host_recapture", MODULE_PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def make_frozen(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    rows = []
    cases = []
    for index, category in enumerate((*workflow.CATEGORIES, "可观察内存等价")):
        nodeid = f"tests/test_example.py::test_case[{index}]"
        identity = workflow.case_id(nodeid)
        directory = frozen / identity
        directory.mkdir()
        workflow.write(directory / "capture.json", {"original": index})
        workflow.write(
            directory / "case_status.json",
            {"nodeid": nodeid, "operator": "example", "arguments": {}, "status": "captured"},
        )
        cases.append(
            {"case_id": identity, "directory": identity, "nodeid": nodeid, "ttir_files": 0}
        )
        rows.append(
            {
                "case_id": identity,
                "nodeid": nodeid,
                "category": category,
                "status": "NOT_VERIFIED",
                "control_flow": False,
            }
        )
    suite = {
        "source_commit": "pinned",
        "selected": [r["nodeid"] for r in rows],
        "all_cases": rows,
        "files": {},
    }
    workflow.write(frozen / "suite.lock.json", suite)
    workflow.write(frozen / "retained.json", cases)
    workflow.write(
        frozen / "corpus.lock.json",
        {
            "source_commit": "pinned",
            "cases": cases,
            "files": {
                str(path.relative_to(frozen)): workflow.digest(path)
                for path in frozen.rglob("*.json")
            },
        },
    )
    baseline = tmp_path / "baseline.json"
    workflow.write(baseline, {"cases": rows})
    return frozen, baseline


def record(plan_dir, target, *, failure=False, commit="pinned", host=True):
    directory = plan_dir / "recaptured" / target["case_id"]
    directory.mkdir()
    workflow.write(
        directory / "process.json",
        {
            "nodeid": target["nodeid"],
            "case_id": target["case_id"],
            "returncode": 1 if failure else 0,
            "timeout": False,
        },
    )
    if failure:
        return
    workflow.write(directory / "upstream.json", {"nodeid": target["nodeid"], "status": "passed"})
    launches = []
    for side in ("lhs", "rhs"):
        path = directory / f"{side}.ttir"
        path.write_text("fixture TTIR " + side)
        launches.append({"file": path.name, "side": side, "sha256": workflow.digest(path)})
    capture = {
        "format": "etv-runtime-capture-v1",
        "source": {"test_id": target["nodeid"], "commit": commit},
        "launches": launches,
        "diagnostics": ["unsupported_random_state"],
    }
    if host:
        capture.update(
            host_capture_format="etv-host-operations-v1", host_operations=[{"kind": "constant"}]
        )
    workflow.write(directory / "capture.json", capture)
    workflow.write(
        directory / "case_status.json",
        {
            "nodeid": target["nodeid"],
            "operator": "example",
            "arguments": {},
            "status": "needs_review",
        },
    )


def test_merge_preserves_denominator_and_failed_original_with_explicit_provenance(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    original_lock = workflow.digest(frozen / "corpus.lock.json")
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    assert plan["target_selected"] == 2
    assert plan["full_selected"] == 3
    record(plan_dir, plan["cases"][0])
    record(plan_dir, plan["cases"][1], failure=True)
    out = tmp_path / "derived"
    audit = workflow.merge(frozen, plan_dir, out)
    assert (audit["enriched"], audit["failed_recaptures"], audit["unchanged_untargeted"]) == (
        1,
        1,
        1,
    )
    assert audit["cases"][0]["diagnostics"] == ["unsupported_random_state"]
    assert audit["cases"][1]["capture_origin"] == "retained_original_after_recapture_failure"
    assert (out / "suite.lock.json").read_bytes() == (frozen / "suite.lock.json").read_bytes()
    failed = plan["cases"][1]["case_id"]
    assert (out / failed / "capture.json").read_bytes() == (
        frozen / failed / "capture.json"
    ).read_bytes()
    assert workflow.digest(frozen / "corpus.lock.json") == original_lock
    workflow.verify_frozen(out)
    with pytest.raises(FileExistsError):
        workflow.merge(frozen, plan_dir, out)


def test_missing_run_result_does_not_silently_drop_case(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    record(plan_dir, plan["cases"][0])
    with pytest.raises(ValueError, match="has not completed"):
        workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_source_revision_mismatch_aborts_before_mutation(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    record(plan_dir, plan["cases"][0], commit="different")
    record(plan_dir, plan["cases"][1])
    with pytest.raises(ValueError, match="source revision"):
        workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_non_enriched_runtime_record_is_retained_as_explicit_failure(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    record(plan_dir, plan["cases"][0], host=False)
    record(plan_dir, plan["cases"][1])
    audit = workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert audit["cases"][0]["failure"] == "recapture_host_recording_not_enabled"
    assert audit["cases"][0]["enriched"] is False


def test_changed_frozen_capture_cannot_be_used_as_baseline(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    original = next(frozen.glob("*/capture.json"))
    original.write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="frozen corpus changed"):
        workflow.prepare(frozen, baseline, tmp_path / "plan")


EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_host_iteration", MODULE_PATH.parent / "export_host_iteration.py"
)
exporter = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(exporter)


def test_iteration_export_uses_actual_full_run_and_keeps_recapture_separate(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    baseline_rows = workflow.read(baseline)["cases"]
    workflow.write(
        baseline,
        {
            "summary": {
                "selected": 3,
                "verdicts": {"NOT_VERIFIED": 3},
                "corpus_lock_sha256": workflow.digest(frozen / "corpus.lock.json"),
            },
            "cases": baseline_rows,
        },
    )
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    for row in plan["cases"]:
        record(plan_dir, row)
    corpus = tmp_path / "derived"
    workflow.merge(frozen, plan_dir, corpus)
    run = tmp_path / "run"
    (run / "analysis").mkdir(parents=True)
    actual_rows = []
    for index, original in enumerate(baseline_rows):
        row = dict(original, status="PROVED" if index == 0 else "NOT_VERIFIED")
        if index == 0:
            row["reason"] = "OBSERVABLE_MEMORY_EQUIVALENT"
            directory = run / "analysis" / row["case_id"]
            directory.mkdir()
            pair_path = run / "work" / row["case_id"] / "pair.json"
            pair_path.parent.mkdir(parents=True)
            workflow.write(
                pair_path, {"format": "etv-pair-v3", "metadata": {"pair_id": row["nodeid"]}}
            )
            workflow.write(
                directory / "report.json",
                {
                    "pair_id": row["nodeid"],
                    "inputs": {
                        "spec": {
                            "kind": "spec",
                            "path": "pair.json",
                            "sha256": workflow.digest(pair_path),
                        }
                    },
                    "status": "PROVED",
                    "reason": row["reason"],
                    "soundness": {"trusted_axioms": []},
                },
            )
        actual_rows.append(row)
    summary = {
        "selected": 3,
        "verdicts": {"PROVED": 1, "NOT_VERIFIED": 2},
        "cases": actual_rows,
        "corpus_lock_sha256": workflow.digest(corpus / "corpus.lock.json"),
    }
    workflow.write(run / "analysis" / "summary.json", summary)
    workflow.write(
        run / "completed.json", {key: value for key, value in summary.items() if key != "cases"}
    )
    workflow.write(
        run / "run.json", {"selected": 3, "corpus_lock_sha256": summary["corpus_lock_sha256"]}
    )
    payload = exporter.export(run, corpus, baseline)
    assert payload["host_iteration"]["enriched"] == 2
    assert payload["host_iteration"]["target_verdicts"]["PROVED"] == 1
    assert payload["proof_audit"]["proved_audited"] == 1
    assert payload["proof_audit"]["counterexamples_replayed"] == 0
    assert payload["proof_audit"]["whole_operator_numeric_counterexamples_confirmed"] is None
    assert payload["transitions"] == {"NOT_VERIFIED → PROVED": 1, "NOT_VERIFIED → NOT_VERIFIED": 2}
    (run / "completed.json").unlink()
    with pytest.raises(FileNotFoundError):
        exporter.export(run, corpus, baseline)


@pytest.mark.parametrize(
    ("row", "capture", "expected"),
    [
        (
            {"status": "UNKNOWN", "control_flow": False, "reason": "HOST_NONFINITE_UNSUPPORTED"},
            {},
            "宿主非有限浮点值未支持",
        ),
        (
            {
                "status": "NOT_VERIFIED",
                "control_flow": False,
                "diagnostics": ["output view needs logical observation projection: lhs.Output"],
            },
            {},
            "非连续输出视图的观察投影尚未支持",
        ),
        (
            {"status": "NOT_VERIFIED", "control_flow": False},
            {"host_operations": [{"kind": "unsupported", "operator": "aten.randint.low"}]},
            "随机状态或随机算子对应尚未建立",
        ),
        (
            {"status": "UNKNOWN", "control_flow": True, "reason": "HOST_OPERATION_UNSUPPORTED"},
            {},
            "分支 / 循环暂不支持",
        ),
        (
            {"status": "NOT_VERIFIED", "control_flow": False, "diagnostics": ["INVALID_PAIRSPEC"]},
            {},
            "PairSpec 契约尚未有效建立",
        ),
    ],
)
def test_iteration_categories_use_actual_evidence(row, capture, expected):
    assert exporter.category(row, capture) == expected


def report_identity_fixture(tmp_path, status="UNKNOWN", reason="HOST_NONFINITE_UNSUPPORTED"):
    row = {
        "case_id": "case",
        "nodeid": "tests/test_pool.py::test_case[0]",
        "status": status,
        "reason": reason,
    }
    pair_path = tmp_path / "pair.json"
    workflow.write(pair_path, {"format": "etv-pair-v3", "metadata": {"pair_id": row["nodeid"]}})
    report = {
        "pair_id": "pair",
        "status": status,
        "reason": reason,
        "inputs": {
            "spec": {"kind": "spec", "path": "pair.json", "sha256": workflow.digest(pair_path)},
            "programs": [],
            "host_operations": [],
        },
        "launches": [],
        "obligations": [],
        "proof": {"roots": []},
    }
    path = tmp_path / "report.json"
    workflow.write(path, report)
    return row, pair_path, path, report


def test_early_host_schema_unknown_binds_identity_through_actual_spec_hash(tmp_path):
    row, pair, path, _ = report_identity_fixture(tmp_path)
    audit = exporter.audit_report(path, row, pair)
    assert audit["identity_check"] == "early_host_schema_unknown_with_bound_spec"
    assert audit["pair_sha256"] == workflow.digest(pair)


@pytest.mark.parametrize("status", ["PROVED", "DISPROVED"])
def test_proof_and_disproof_never_accept_filename_identity(tmp_path, status):
    row, pair, path, _ = report_identity_fixture(tmp_path, status=status)
    with pytest.raises(ValueError, match="identity mismatch"):
        exporter.audit_report(path, row, pair)


@pytest.mark.parametrize(
    "change",
    [
        "wrong_spec_hash",
        "wrong_spec_case",
        "missing_spec",
        "late_execution",
        "non_host_reason",
        "wrong_stem",
    ],
)
def test_early_unknown_fallback_rejects_unbound_or_late_reports(tmp_path, change):
    row, pair, path, report = report_identity_fixture(tmp_path)
    if change == "wrong_spec_hash":
        report["inputs"]["spec"]["sha256"] = "0" * 64
    elif change == "wrong_spec_case":
        workflow.write(pair, {"format": "etv-pair-v3", "metadata": {"pair_id": "another-case"}})
        report["inputs"]["spec"]["sha256"] = workflow.digest(pair)
    elif change == "missing_spec":
        report["inputs"]["spec"] = None
    elif change == "late_execution":
        report["launches"] = [{"id": "lhs.0"}]
    elif change == "non_host_reason":
        report["reason"] = row["reason"] = "TTIR_OP_UNSUPPORTED"
    elif change == "wrong_stem":
        report["pair_id"] = "other-filename"
    workflow.write(path, report)
    with pytest.raises(ValueError):
        exporter.audit_report(path, row, pair)


def test_exact_identity_report_also_requires_current_spec_hash(tmp_path):
    row, pair, path, report = report_identity_fixture(
        tmp_path, status="PROVED", reason="OBSERVABLE_MEMORY_EQUIVALENT"
    )
    report["pair_id"] = row["nodeid"]
    workflow.write(path, report)
    assert exporter.audit_report(path, row, pair)["identity_check"] == "exact_pair_id"
    pair.write_text(pair.read_text() + "\n")
    with pytest.raises(ValueError, match="expected PairSpec"):
        exporter.audit_report(path, row, pair)


def test_control_flow_can_be_recaptured_without_enabling_its_verification(tmp_path):
    frozen, baseline = make_frozen(tmp_path)
    data = workflow.read(baseline)
    data["cases"][0]["control_flow"] = True
    workflow.write(baseline, data)
    category = workflow.CATEGORIES[0]
    with pytest.raises(ValueError, match="control flow"):
        workflow.prepare(frozen, baseline, tmp_path / "rejected", categories=[category])
    plan = workflow.prepare(
        frozen, baseline, tmp_path / "allowed", categories=[category], capture_control_flow=True
    )
    assert plan["target_selected"] == 1
    assert plan["full_selected"] == 3
    assert plan["control_flow_capture_only"] is True
    assert plan["control_flow_semantics_supported"] is False
    assert plan["cases"][0]["nodeid"] == data["cases"][0]["nodeid"]
