"""Export a completed 1,365-case TTIR-only run with strict artifact/scope checks.

Host capture records are retained as excluded provenance. They never supply a
program step, constant, initialization, or assumption for the exported proofs.
An optional composite run is an explicitly separate historical appendix.
"""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re


def _helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _helper("export_host_iteration")
recapture = _helper("prepare_host_recapture")
SCOPE = "ttir-only"
EXPECTED_POPULATION = 1365
SCOPE_DEPENDENCY = "host dependency outside ttir-only execution scope: "
TARGET_CATEGORIES = {
    "external_mapping": "外部输入映射缺失",
    "duplicate_endpoints": "重复物理端点 / 存储配对",
    "output_dtype": "ABI 输出类型冲突",
}
STRICT_REASONS = {
    "INPLACE_MEMORY_DEPENDENCY_UNPROVED": "原地 / 跨 lane 读写依赖未证明",
    "TTIR_NAN_CONSTANT_UNSUPPORTED": "NaN 常量尚未支持",
    "SMT_UNKNOWN": "SMT 求解未决 / 达到预算",
}
read, digest, counts = legacy.read, legacy.digest, legacy.counts


def source_hashes(root):
    result = {
        str(path.relative_to(root)): digest(path)
        for directory in ("src/etv", "src/etv_bench", "tools")
        for path in sorted((root / directory).rglob("*.py"))
    }
    result.update({
        str(path.relative_to(root)): digest(path)
        for path in sorted((root / "native").rglob("*"))
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}
    })
    return result


def strict_category(row):
    """Classify current pair/proof diagnostics, never excluded host diagnostics."""
    if row["status"] == "PROVED":
        if row.get("comparison_contract", {}).get("encoding") == "boolean_numeric_01":
            return "TTIR 输出在显式布尔 0/1 编码下逻辑值等价"
        return "TTIR 执行片段的可观察内存等价"
    if row["status"] == "DISPROVED":
        return "TTIR 执行片段不等价"
    # A syntax scan is a separate structural statistic, never a root cause.
    # For an actual formal report, its stopping reason takes precedence over
    # capture/pair diagnostics (which may describe an earlier construction step).
    reason = row.get("reason", "")
    if row["status"] == "UNKNOWN" and reason:
        diagnostic = row.get("diagnostic") or row.get("audit", {}).get("diagnostic", "")
        if reason == "TTIR_REGION_UNSUPPORTED" and str(diagnostic).startswith(("scf.", "cf.")):
            return "分支 / 循环暂不支持"
        return STRICT_REASONS.get(reason, legacy.REASONS.get(reason, reason))
    diagnostics = row.get("diagnostics", [])
    if any(text.startswith(SCOPE_DEPENDENCY) for text in diagnostics):
        return "TTIR-only 范围外的宿主数据依赖 / 最终输出"
    if any(text.startswith("host evidence outside ttir-only scope") for text in diagnostics):
        return "TTIR-only 排除边界的采集证据不完整"
    if "external input lacks a unique declared storage role" in diagnostics:
        return "外部输入映射缺失"
    if any(
        text.startswith(
            (
                "unsupported host operation",
                "host event incomplete",
                "unsupported host capture format",
            )
        )
        for text in diagnostics
    ):
        return "TTIR-only 宿主事件已排除；采集证据待检查"
    # This fallback uses the established diagnostic vocabulary, with the
    # structural flag removed so it cannot hide duplicate endpoints, dtype
    # conflicts, or INVALID_PAIRSPEC.
    return legacy.category(row | {"control_flow": False}, {})


def target_groups(cases):
    """Freeze target membership by corrected baseline evidence, not new verdicts."""
    groups = []
    categories = TARGET_CATEGORIES | {"nonfinite_constants": "非有限浮点常量未支持"}
    for key, category in categories.items():
        members = [
            row for row in cases
            if row["baseline_status"] == (
                "UNKNOWN" if key == "nonfinite_constants" else "NOT_VERIFIED"
            )
            and row["baseline_category"] == category
        ]
        if key == "nonfinite_constants" and not members:
            continue
        groups.append({
            "key": key,
            "baseline_category": category,
            "selected": len(members),
            "case_ids": [row["case_id"] for row in members],
            "baseline_verdicts": counts([
                {"status": row["baseline_status"]} for row in members
            ]),
            "verdicts": counts(members),
            "pair_ready": sum(row.get("pair_status") == "ready" for row in members),
            "formal_reports": sum(row["status"] != "NOT_VERIFIED" for row in members),
            "newly_proved": sum(row.get("newly_proved", False) for row in members),
            "diagnostic_category_changed": sum(
                row["category"] != row["baseline_category"] for row in members
            ),
            "transitions": dict(Counter(
                row["baseline_status"] + " → " + row["status"] for row in members
            )),
            "unresolved_categories": dict(Counter(
                row["category"] for row in members
                if row["status"] in ("UNKNOWN", "NOT_VERIFIED")
            ).most_common()),
            "control_flow_structural": sum(row["control_flow"] for row in members),
        })
    return groups


def audit_output_representation(pair, capture, pair_path):
    representation = pair["observation"].get("value_representation")
    roles = pair["predicates"].get("roles", {})
    contract = {
        "kind": "typed-observable-values",
        "encoding": None,
        "logical_role": "Output",
        "physical_roles": {"lhs": "Output", "rhs": "Output"},
        "physical_elements": {
            side: roles.get("Output", {}).get("element") for side in ("lhs", "rhs")
        },
        "encodings": {"lhs": "identity", "rhs": "identity"},
        "bit_identical": False,
        "output_dtype_equality_claimed": False,
    }
    if representation is None:
        return contract
    if representation.get("kind") != "boolean_numeric_01":
        raise ValueError("unaudited output representation contract")
    source = capture["source"]
    operator = source.get("operator")
    if operator is None:
        matched = re.fullmatch(
            r"(?:[^:]+/)?test_([a-z_]+)\.py::test_\1(?:\[.*\])?", source.get("test_id", "")
        )
        operator = matched.group(1) if matched else None
    if operator not in {"eq", "ne", "lt", "le", "gt", "ge", "isinf", "isnan"}:
        raise ValueError("boolean output contract lacks supported captured operator identity")
    encodings = {}
    physical = {}
    elements = {}
    for side in ("lhs", "rhs"):
        side_repr = representation.get(side, {})
        role, encoding = side_repr.get("role"), side_repr.get("encoding")
        element = "bool" if encoding == "bool_to_numeric_01" else "abstract_float"
        if (
            role != "Output_" + side
            or encoding not in ("identity", "bool_to_numeric_01")
            or roles.get(role) != {"kind": "buffer", "element": element}
        ):
            raise ValueError("output representation disagrees with physical role type")
        outputs = capture["outputs"].get(side, {})
        if len(outputs) != 1 or set(outputs.values()) != {"Output"}:
            raise ValueError("encoded observation is not the original captured output")
        storage = next(iter(outputs))
        dtypes = {
            slot["runtime"].get("dtype")
            for launch in capture["launches"] if launch["side"] == side
            for slot in launch.get("slots", [])
            if slot.get("runtime", {}).get("storage_id") == storage
        }
        allowed = {"torch.bool"} if element == "bool" else {
            "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"
        }
        if not dtypes or not dtypes <= allowed:
            raise ValueError("output encoding is not supported by captured physical dtypes")
        encodings[side], physical[side], elements[side] = encoding, role, element
    if sorted(encodings.values()) != ["bool_to_numeric_01", "identity"]:
        raise ValueError("boolean output encoding requires one bool and one numeric side")
    if pair["metadata"].get("target", {}).get("boolean_storage") != "canonical_i8":
        raise ValueError("encoded boolean storage premise is not explicitly declared")
    provenance = read(pair_path.parent / "provenance.json")
    evidence = provenance.get("output_representation_evidence") or {}
    if (
        provenance.get("pair_sha256") != digest(pair_path)
        or provenance.get("capture_sha256") != digest(pair_path.parent / "capture.json")
        or evidence.get("operator") != operator
        or evidence.get("value_representation") != representation
    ):
        raise ValueError("encoded output provenance is not bound to the actual pair/capture")
    return contract | {
        "kind": "logical-value-under-encoding",
        "encoding": "boolean_numeric_01",
        "physical_roles": physical,
        "physical_elements": elements,
        "encodings": encodings,
    }


def audit_pair_scope(pair_path, capture, corpus_case, *, require_complete):
    pair = read(pair_path)
    if (
        pair.get("format") != "etv-pair-v3"
        or pair.get("metadata", {}).get("execution_scope") != SCOPE
    ):
        raise ValueError("PairSpec is not explicitly TTIR-only")
    for section in ("constraints", "custom", "parameters"):
        if pair.get("predicates", {}).get(section):
            raise ValueError("automatic TTIR-only export cannot add semantic premises")
    if pair.get("assumptions", {}).get("for_llm") or pair.get("rewrites"):
        raise ValueError("automatic TTIR-only export cannot add host assumptions or rewrites")
    observation = pair.get("observation", {})
    if (
        observation.get("role") != "Output"
        or observation.get("require_full_coverage") is not True
        or observation.get("numel") != capture.get("output_numel", 0)
    ):
        raise ValueError("TTIR-only observation does not cover the original full output")
    comparison_contract = audit_output_representation(pair, capture, pair_path)
    captured = {}
    for launch in capture["launches"]:
        key = (launch["side"], launch["file"], launch["function"])
        if key in captured:
            raise ValueError("ambiguous captured TTIR identity")
        artifact = recapture.contained(corpus_case, launch["file"])
        if not launch.get("selected_runtime_launch") or digest(artifact) != launch.get("sha256"):
            raise ValueError("captured physical TTIR hash or runtime selection is invalid")
        captured[key] = launch
    matched = set()
    mappings = []
    for side in ("lhs", "rhs"):
        last_step = None
        last_pair_step = None
        for program in sorted(pair["programs"][side], key=lambda item: item["step"]):
            if program.get("kind", "ttir") != "ttir":
                raise ValueError("TTIR-only PairSpec contains a non-TTIR program")
            key = (side, program["file"], program["function"])
            if key not in captured or key in matched:
                raise ValueError("PairSpec launch is not a unique captured physical TTIR")
            actual = captured[key]
            work_artifact = recapture.contained(pair_path.parent, program["file"])
            if digest(work_artifact) != actual["sha256"]:
                raise ValueError("generated pair uses changed or synthetic TTIR bytes")
            if last_step is not None and (
                actual["step"] <= last_step or program["step"] <= last_pair_step
            ):
                raise ValueError("TTIR scheduling changes captured launch order")
            last_step = actual["step"]
            last_pair_step = program["step"]
            matched.add(key)
            mappings.append(
                {
                    "id": program["id"],
                    "side": side,
                    "file": program["file"],
                    "function": program["function"],
                    "sha256": actual["sha256"],
                    "pair_step": program["step"],
                    "capture_step": actual["step"],
                    "capture_ordinal": actual["ordinal"],
                }
            )
    if require_complete and matched != set(captured):
        raise ValueError("formal result omits a captured physical TTIR launch")
    return {
        "execution_scope": SCOPE,
        "pair_sha256": digest(pair_path),
        "pair_host_steps": 0,
        "full_output_observation": True,
        "comparison_contract": comparison_contract,
        "boolean_storage_premise": pair["metadata"].get("target", {}).get("boolean_storage"),
        "captured_physical_launch_count": len(captured),
        "pair_physical_launch_count": len(matched),
        "omitted_capture_launches": len(captured) - len(matched),
        "physical_launches": mappings,
    }


def audit_report_scope(report_path, row, pair_path, pair_audit):
    report = read(report_path)
    if report.get("execution_scope") != SCOPE:
        raise ValueError("formal report is not explicitly TTIR-only")
    if report.get("inputs", {}).get("host_operations"):
        raise ValueError("TTIR-only report contains host input records")
    if str(report.get("reason", "")).startswith("HOST_"):
        raise ValueError("excluded host semantics cannot be a TTIR-only proof result")
    expected_contract = pair_audit["comparison_contract"]
    actual_contract = report.get("comparison_contract")
    if actual_contract is None and expected_contract["encoding"] is None:
        # Historical typed reports predate the explicit report field; their
        # hashed PairSpec still has the default typed observation contract.
        actual_contract = expected_contract
    if actual_contract != expected_contract:
        raise ValueError("formal report comparison contract disagrees with actual PairSpec")
    if report["status"] == "PROVED" and (
        (expected_contract["encoding"] == "boolean_numeric_01")
        != (report["reason"] == "LOGICAL_BOOLEAN_OUTPUT_EQUIVALENT")
    ):
        raise ValueError("proved reason disagrees with the output comparison contract")
    facts = report.get("launches", [])
    if any(item.get("kind", "ttir") != "ttir" for item in facts):
        raise ValueError("TTIR-only report contains a host execution step")
    physical = report.get("physical_launch_count")
    execution = report.get("execution_step_count")
    if physical != execution or type(physical) is not int or physical != len(facts):
        raise ValueError("TTIR-only physical/execution launch counts disagree")
    expected = {item["id"]: item for item in pair_audit["physical_launches"]}
    if {item["id"] for item in facts} != set(expected) or len(facts) != len(expected):
        raise ValueError("report execution facts do not cover the actual PairSpec TTIR launches")
    for fact in facts:
        actual = expected[fact["id"]]
        if (
            any(fact.get(name) != actual[name] for name in ("side", "function"))
            or fact.get("file") != Path(actual["file"]).name
            or fact.get("step") != actual["pair_step"]
        ):
            raise ValueError("report launch fact disagrees with captured TTIR identity")
    sources = report.get("inputs", {}).get("programs", [])
    if len(sources) != len(expected) or {item.get("launch_id") for item in sources} != set(
        expected
    ):
        raise ValueError("report TTIR input manifest is incomplete")
    for source in sources:
        actual = expected[source["launch_id"]]
        if (
            source.get("kind") != "ttir"
            or source.get("sha256") != actual["sha256"]
            or source.get("path") != actual["file"]
        ):
            raise ValueError("report program hash is not the captured physical TTIR")
    audit = legacy.audit_report(report_path, row, pair_path)
    if audit["host_steps"]:
        raise ValueError("host step passed the strict report gate")
    return audit | {
        "execution_scope": SCOPE, "host_input_records": 0,
        "comparison_contract": actual_contract,
        "semantics": report.get("semantics", {}),
    }


def export(run, corpus, baseline_path, replay_path=None, composite_history=None, source_root=None):
    run, corpus = run.resolve(), corpus.resolve()
    metadata = read(run / "run.json")
    summary = read(run / "analysis/summary.json")
    if metadata.get("execution_scope") != SCOPE or summary.get("execution_scope") != SCOPE:
        raise ValueError("run and summary must explicitly declare TTIR-only scope")
    if (
        summary.get("selected") != EXPECTED_POPULATION
        or metadata.get("selected") != EXPECTED_POPULATION
    ):
        raise ValueError("TTIR-only export requires the complete fixed 1365-case run")
    source_root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    sources = source_hashes(source_root)
    if not sources or metadata.get("source_sha256") != sources:
        raise ValueError("implementation source differs from the completed run")
    # Reuse frozen corpus/population/completion and report-hash checks. Its host
    # classifications are replaced below; they are never the strict result.
    payload = legacy.export(
        run, corpus, baseline_path, replay_path, allow_same_strict_corpus=True
    )
    suite = read(corpus / "suite.lock.json")
    baseline_data = read(baseline_path)
    baseline = {row["case_id"]: row for row in baseline_data["cases"]}
    strict_baseline = baseline_data.get("execution_scope") == SCOPE
    if strict_baseline and (
        baseline_data.get("format") != "etv-ttir-only-export-v1"
        or baseline_data.get("evidence", {}).get("source_commit") != suite["source_commit"]
    ):
        raise ValueError("strict baseline does not bind to the same pinned source")
    cases = payload["cases"]
    pair_audits, report_count, absent_pairs = [], 0, []
    for row in cases:
        identity = row["case_id"]
        if row.get("execution_scope") != SCOPE:
            raise ValueError("case summary has a different execution scope: " + identity)
        directory = corpus / identity
        capture_path = directory / "capture.json"
        capture = read(capture_path)
        if (
            capture["source"].get("commit") != suite["source_commit"]
            or capture["source"].get("test_id") != row["nodeid"]
        ):
            raise ValueError("captured source revision or case identity changed")
        work = run / "work" / identity
        if digest(work / "capture.json") != digest(capture_path):
            raise ValueError("verification workspace changed the frozen capture")
        if digest(work / "case_status.json") != digest(directory / "case_status.json"):
            raise ValueError("verification workspace changed frozen runtime parameters")
        arguments = read(directory / "case_status.json").get("arguments")
        previous = baseline[identity].get("capture", {}).get("arguments")
        if (
            not isinstance(arguments, dict)
            or not isinstance(previous, dict)
            or recapture.argument_differences(
                recapture.normalized_arguments(previous), recapture.normalized_arguments(arguments)
            )
        ):
            raise ValueError("runtime parameters differ from the original fixed baseline")
        if not (
            row["status"] == "NOT_VERIFIED" and row.get("analysis_error")
        ) and recapture.argument_differences(arguments, row.get("capture", {}).get("arguments")):
            raise ValueError("case summary disagrees with frozen runtime parameters")
        row["category"] = strict_category(row)
        if strict_baseline:
            row["baseline_reported_category"] = row["baseline_category"]
            row["baseline_category"] = strict_category(baseline[identity])
        row["execution_host_operation_count"] = 0
        row["recorded_host_operations_excluded"] = len(capture.get("host_operations", []))
        row["excluded_capture_diagnostics"] = capture.get("diagnostics", [])
        pair_path = work / "pair.json"
        if pair_path.exists():
            pair_audit = audit_pair_scope(
                pair_path, capture, directory, require_complete=row["status"] != "NOT_VERIFIED"
            )
            row["scope_audit"] = pair_audit
            row["comparison_contract"] = pair_audit["comparison_contract"]
            pair_audits.append(pair_audit)
        elif row["status"] != "NOT_VERIFIED":
            raise ValueError("formal result lacks a generated PairSpec")
        else:
            absent_pairs.append(identity)
            row["scope_audit"] = {"execution_scope": SCOPE, "pair_generated": False}
        report_path = run / "analysis" / identity / "report.json"
        if report_path.exists():
            if not pair_path.exists() or row["status"] == "NOT_VERIFIED":
                raise ValueError("unclassified verification report exists for an unverified case")
            row["audit"] = audit_report_scope(report_path, row, pair_path, pair_audit)
            report_count += 1
        elif row["status"] != "NOT_VERIFIED":
            raise ValueError("formal verdict lacks an actual report")
        row["category"] = strict_category(row)
    recaptured = [row for row in cases if "recapture" in row]
    payload["host_iteration"]["groups"] = []
    for category in sorted({row["baseline_category"] for row in recaptured}):
        members = [row for row in recaptured if row["baseline_category"] == category]
        payload["host_iteration"]["groups"].append({
            "baseline_category": category,
            "selected": len(members),
            "enriched": sum(row["recapture"]["enriched"] for row in members),
            "failed_recaptures": sum(not row["recapture"]["enriched"] for row in members),
            "verdicts": counts(members),
            "categories": dict(Counter(row["category"] for row in members)),
            "transitions": dict(Counter(
                row["baseline_status"] + " → " + row["status"] for row in members
            )),
        })
    payload["host_iteration"]["baseline_category_changed"] = sum(
        row["category"] != row["baseline_category"] for row in cases if "recapture" in row
    )
    payload["host_iteration"]["note"] = (
        "采集迭代仅作来源追溯；本报告验证程序只含真实编译 TTIR，宿主事件不进入证明。"
    )
    payload["format"] = "etv-ttir-only-export-v1"
    payload["execution_scope"] = SCOPE
    payload["classification_policy"] = (
        "Actual formal reason first; otherwise current PairSpec diagnostics. "
        "Control-flow syntax is an independent structural annotation, not a cause override."
    )
    payload["baseline_execution_scope"] = baseline_data.get("execution_scope", "original-ttir")
    payload["baseline_label"] = "严格 TTIR-only 上轮基线" if strict_baseline else "原 TTIR 片段基线"
    payload["target_groups"] = target_groups(cases) if strict_baseline else []
    payload["same_frozen_corpus_as_baseline"] = (
        baseline_data["summary"].get("corpus_lock_sha256")
        == payload["summary"].get("corpus_lock_sha256")
    )
    models = {
        json.dumps(row["audit"].get("semantics", {}), sort_keys=True)
        for row in cases if row["status"] != "NOT_VERIFIED"
    }
    payload["floating_models"] = [json.loads(model) for model in sorted(models)]
    payload["comparison_contracts"] = {
        "typed_observable_proved": sum(
            row["status"] == "PROVED"
            and row["comparison_contract"]["encoding"] is None for row in cases
        ),
        "encoded_logical_proved": sum(
            row["status"] == "PROVED"
            and row["comparison_contract"]["encoding"] == "boolean_numeric_01" for row in cases
        ),
        "encoded_pairs": sum(
            row.get("comparison_contract", {}).get("encoding") == "boolean_numeric_01"
            for row in cases
        ),
        "bit_identical_claimed": False,
        "output_dtype_equality_claimed": False,
    }
    payload["unresolved_categories"] = dict(
        Counter(
            row["category"] for row in cases if row["status"] in ("UNKNOWN", "NOT_VERIFIED")
        ).most_common()
    )
    payload["scope_audit"] = {
        "execution_scope": SCOPE,
        "pairs_audited": len(pair_audits),
        "reports_audited": report_count,
        "pair_specs_audited": len(pair_audits),
        "formal_reports_audited": report_count,
        "nonready_count": sum(row["status"] == "NOT_VERIFIED" for row in cases),
        "pair_host_steps": 0,
        "report_host_steps": 0,
        "report_host_inputs": 0,
        "all_report_launch_counts_match": True,
        "all_pair_launches_match_captured_ttir": True,
        "frozen_capture_and_runtime_parameters_unchanged": True,
        "runtime_parameters_match_original_baseline": True,
        "source_files_audited": len(sources),
        "cases_without_generated_pair": absent_pairs,
        "captured_host_events_used_as_semantics": False,
        "whole_operator_equivalence_claimed": False,
        "full_output_observation": True,
    }
    payload["proof_audit"]["execution_scope"] = SCOPE
    payload["proof_audit"]["whole_operator_equivalence_claimed"] = False
    for item in payload["replay"]:
        if item.get("execution_scope") != SCOPE:
            raise ValueError("counterexample replay has no matching TTIR-only scope")
    if composite_history:
        history = read(composite_history)
        if history.get("format") != "etv-host-iteration-export-v1" or {
            row["case_id"] for row in history["cases"]
        } != set(baseline):
            raise ValueError("composite history is not the matching separate host iteration")
        payload["composite_history"] = {
            "source_sha256": digest(composite_history),
            "execution_scope": "ttir-with-host-ops",
            "included_in_ttir_only_totals": False,
            "summary": history["summary"],
            "proof_audit": history["proof_audit"],
            "note": "历史复合程序实验，包含语义化宿主步骤；与本次严格 TTIR-only 主结果分开呈现。",
        }
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-data", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--composite-history", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = export(
        args.run,
        args.corpus,
        args.baseline_data,
        args.replay,
        args.composite_history,
        args.source_root,
    )
    with args.out.open("x") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("summary", "scope_audit", "proof_audit")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
