"""Export evidence from a completed host-op iteration; never invent run results.

Input: run_formal_suite.py output, its immutable enriched corpus, and the previous
web-results.json. Output is data only; the existing HTML and Markdown stay intact.
Optional counterexample replay evidence must bind to the current report hashes.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


STATUSES = ("PROVED", "UNKNOWN", "NOT_VERIFIED", "DISPROVED")
HOST_REASONS = {
    "HOST_NONFINITE_UNSUPPORTED": "宿主非有限浮点值未支持",
    "HOST_DTYPE_UNSUPPORTED": "宿主数据类型未支持",
    "HOST_DTYPE_CONVERSION_UNSUPPORTED": "宿主类型转换未支持",
    "HOST_ALIAS_UNSUPPORTED": "宿主输入输出别名未支持",
    "HOST_OVERLAPPING_OUTPUT_UNSUPPORTED": "宿主重叠输出未支持",
    "HOST_DIVISION_UNSUPPORTED": "宿主除法模式未支持",
    "HOST_OPERATION_UNSUPPORTED": "宿主操作尚未支持",
}
REASONS = {
    "TTIR_CONSTANT_UNSUPPORTED": "非有限浮点常量未支持",
    "FLOAT_DEFINEDNESS_NOT_PROVED": "浮点定义域未闭合",
    "TTIR_OP_UNSUPPORTED": "TTIR 操作未支持",
    "TTIR_REGION_UNSUPPORTED": "复杂 reduction region 未支持",
    "NO_EQUIVALENCE_PROOF": "等式证明未闭合",
    "CAST_EQUIVALENCE_NOT_REWRITTEN": "整数转换等式未闭合",
    "RESOURCE_LIMIT": "求解 / 展开达到资源限制",
    "UNDEFINED_LOAD_REACHABLE": "可达未定义加载",
    "BOTH_SIDES_MULTI_LAUNCH_UNSUPPORTED": "双侧多步执行尚未支持",
}
CONTROL_FLOW = re.compile(
    r"^\s*(?:%[^=]+\s*=\s*)?(?:scf|cf)\.(?:for|if|while|condition|br|cond_br)\b", re.M
)
RANDOM_OPERATOR = re.compile(r"(?:rand|bernoulli|seed|dropout)", re.I)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    result = Counter(row["status"] for row in rows)
    return {status: result[status] for status in STATUSES}


def category(row, capture):
    if row["status"] == "PROVED":
        return "可观察内存等价"
    if row["status"] == "DISPROVED":
        return "声明执行片段不等价"
    if row["control_flow"]:
        return "分支 / 循环暂不支持"
    reason = row.get("reason", "")
    diagnostics = row.get("diagnostics", []) + capture.get("diagnostics", [])
    combined = "\n".join(diagnostics)
    if row.get("analysis_error") or row.get("pair_returncode") not in (None, 0):
        return "验证进程失败 / 超时"
    if "output view needs logical observation projection" in combined:
        return "非连续输出视图的观察投影尚未支持"
    unresolved_random = any(
        event.get("kind") == "unsupported" and RANDOM_OPERATOR.search(event.get("operator", ""))
        for event in capture.get("host_operations", [])
    )
    if unresolved_random or any(
        "random" in text.lower() or "rng" in text.lower() for text in diagnostics
    ):
        return "随机状态或随机算子对应尚未建立"
    if reason in HOST_REASONS:
        return HOST_REASONS[reason]
    if reason and reason.startswith("HOST_"):
        return "宿主语义未覆盖：" + reason
    if "unsupported host operation" in combined:
        return "宿主操作尚未支持"
    if "host event incomplete" in combined or "unsupported host capture format" in combined:
        return "宿主采集记录不完整或不兼容"
    if "storage role has inconsistent element types" in diagnostics:
        return "ABI 输出类型冲突"
    if (
        "duplicate physical endpoints for one role require review" in diagnostics
        or "duplicate host endpoints for one role require review" in diagnostics
    ):
        return "重复物理端点 / 存储配对"
    if "external input lacks a unique declared storage role" in diagnostics:
        return "外部输入映射缺失"
    if "launch selection or grid unsupported" in diagnostics:
        return "多维 grid 尚不支持"
    if "INVALID_PAIRSPEC" in diagnostics:
        # INVALID_PAIRSPEC is broader than a missing output store. Do not infer
        # the old category unless the actual verifier diagnostic identifies it.
        if "observation has no selected store" in str(row.get("diagnostic", "")):
            return "观察契约缺少输出写入"
        return "PairSpec 契约尚未有效建立"
    if reason in REASONS:
        return REASONS[reason]
    return reason or "其他配对 / 契约问题"


def audit_report(path, row, pair_path):
    report = read(path)
    pair = read(pair_path)
    pair_hash = digest(pair_path)
    recorded_inputs = report.get("inputs", {})
    recorded_spec = recorded_inputs.get("spec")
    if (
        pair.get("format") != "etv-pair-v3"
        or pair.get("metadata", {}).get("pair_id") != row["nodeid"]
        or not isinstance(recorded_spec, dict)
        or recorded_spec.get("kind") != "spec"
        or recorded_spec.get("sha256") != pair_hash
    ):
        raise ValueError(
            "verification report is not bound to the expected PairSpec: " + row["case_id"]
        )
    if report["status"] != row["status"] or report["reason"] != row.get("reason"):
        raise ValueError("case summary and actual verification report disagree: " + row["case_id"])
    identity = "exact_pair_id"
    if report["pair_id"] != row["nodeid"]:
        # verify() records the spec hash before parse_pair_spec(). A supported
        # capture with e.g. -Inf can stop during host schema parsing, before the
        # declared pair_id and execution facts are installed in the report.
        # Only this early UNKNOWN path may use the filename stem. Proofs and
        # counterexamples always need an exact declared case identity.
        early_host_schema_stop = (
            report["status"] == "UNKNOWN"
            and report["reason"] in HOST_REASONS
            and report["pair_id"] == pair_path.stem
            and not report.get("launches")
            and not report.get("obligations")
            and not report.get("proof", {}).get("roots")
            and not recorded_inputs.get("programs")
            and not recorded_inputs.get("host_operations")
        )
        if not early_host_schema_stop:
            raise ValueError("verification report case identity mismatch: " + row["case_id"])
        identity = "early_host_schema_unknown_with_bound_spec"
    equalities = report.get("proof", {}).get("roots", [])
    rules = report.get("rewrites", {}).get("admissions", [])
    return {
        "report_sha256": digest(path),
        "pair_sha256": pair_hash,
        "identity_check": identity,
        "reported_pair_id": report["pair_id"],
        "status": report["status"],
        "reason": report["reason"],
        "diagnostic": report.get("diagnostic", ""),
        "soundness": report.get("soundness", {}),
        "obligations": dict(Counter(item["status"] for item in report.get("obligations", []))),
        "root_count": len(equalities),
        "merged_roots": sum(bool(item["merged"]) for item in equalities),
        "rule_admissions": dict(Counter(item["admission"] for item in rules)),
        "used_unverified_rules": [
            item["id"]
            for item in rules
            if item.get("used") and item["admission"] == "admitted_unverified"
        ],
        "physical_launch_count": report.get(
            "physical_launch_count", len(report.get("launches", []))
        ),
        "execution_step_count": report.get("execution_step_count", len(report.get("launches", []))),
        "host_steps": sum(item.get("kind") == "host_op" for item in report.get("launches", [])),
    }


def export(run, corpus, baseline_path, replay_path=None, *, allow_same_strict_corpus=False):
    summary = read(run / "analysis" / "summary.json")
    completed = read(run / "completed.json")
    metadata = read(run / "run.json")
    if completed != {key: value for key, value in summary.items() if key != "cases"}:
        raise ValueError("run completion receipt disagrees with analysis summary")
    frozen = read(corpus / "corpus.lock.json")
    if summary.get("corpus_lock_sha256") != digest(corpus / "corpus.lock.json") or metadata.get(
        "corpus_lock_sha256"
    ) != digest(corpus / "corpus.lock.json"):
        raise ValueError("run used a different frozen corpus")
    for name, expected in frozen["files"].items():
        path = (corpus / name).resolve()
        if (
            not path.is_relative_to(corpus.resolve())
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError("enriched frozen input changed: " + name)
    suite = read(corpus / "suite.lock.json")
    recapture = read(corpus / "recapture-audit.json")
    baseline = read(baseline_path)
    original = {row["case_id"]: row for row in baseline["cases"]}
    selected = set(suite["selected"])
    rows = summary["cases"]
    if (
        len(rows) != len(selected)
        or len(original) != len(selected)
        or {row["nodeid"] for row in rows} != selected
        or {row["case_id"] for row in rows} != set(original)
    ):
        raise ValueError("baseline, frozen suite and actual run populations differ")
    if (
        summary["selected"] != len(rows)
        or metadata["selected"] != len(rows)
        or summary["verdicts"] != dict(Counter(row["status"] for row in rows))
    ):
        raise ValueError("run result totals disagree with actual case rows")
    if recapture["source_commit"] != suite["source_commit"] or recapture["full_selected"] != len(
        rows
    ):
        raise ValueError("recapture audit population or source revision differs")
    targets = {row["case_id"]: row for row in recapture["cases"]}
    if len(targets) != len(recapture["cases"]) or len(targets) != recapture["target_selected"]:
        raise ValueError("recapture target audit contains duplicates or inconsistent totals")
    baseline_lock = baseline["summary"].get("corpus_lock_sha256")
    same_strict_corpus = (
        allow_same_strict_corpus
        and baseline.get("format") == "etv-ttir-only-export-v1"
        and baseline.get("execution_scope") == "ttir-only"
        and baseline_lock == digest(corpus / "corpus.lock.json")
        and baseline.get("evidence", {}).get("source_commit") == suite["source_commit"]
    )
    if baseline_lock != frozen.get("parent_corpus_lock_sha256") and not same_strict_corpus:
        raise ValueError("baseline does not describe the parent frozen corpus")
    retained = {row["case_id"]: row for row in read(corpus / "retained.json")}
    cases = []
    for raw in rows:
        row = dict(raw)
        if row["status"] not in STATUSES:
            raise ValueError("unknown case status")
        identity = row["case_id"]
        if hashlib.sha256(row["nodeid"].encode()).hexdigest()[:16] != identity:
            raise ValueError("case identity mismatch")
        directory = corpus / identity
        capture = read(directory / "capture.json")
        row["control_flow"] = any(
            CONTROL_FLOW.search(path.read_text()) for path in directory.glob("*/*.ttir")
        )
        if row["control_flow"] and row["status"] in ("PROVED", "DISPROVED"):
            raise ValueError("a control-flow case unexpectedly obtained a conclusion")
        row["category"] = category(row, capture)
        row["parameters"] = retained[identity].get("parameters", {})
        row["baseline_status"] = original[identity]["status"]
        row["baseline_category"] = original[identity]["category"]
        row["newly_proved"] = row["status"] == "PROVED" and row["baseline_status"] != "PROVED"
        row["capture_origin"] = targets.get(identity, {}).get(
            "capture_origin", "unchanged_original"
        )
        row["host_operation_count"] = len(capture.get("host_operations", []))
        if identity in targets:
            row["recapture"] = targets[identity]
        if row["status"] != "NOT_VERIFIED":
            row["audit"] = audit_report(
                run / "analysis" / identity / "report.json",
                row,
                run / "work" / identity / "pair.json",
            )
        cases.append(row)
    by_id = {row["case_id"]: row for row in cases}
    if not set(targets) <= set(by_id):
        raise ValueError("recapture targets are absent from the complete run")
    target_rows = [by_id[identity] for identity in targets]
    groups = []
    for group in sorted({row["baseline_category"] for row in target_rows}):
        members = [row for row in target_rows if row["baseline_category"] == group]
        groups.append(
            {
                "baseline_category": group,
                "selected": len(members),
                "enriched": sum(row["recapture"]["enriched"] for row in members),
                "failed_recaptures": sum(not row["recapture"]["enriched"] for row in members),
                "verdicts": counts(members),
                "categories": dict(Counter(row["category"] for row in members)),
                "transitions": dict(
                    Counter(row["baseline_status"] + " → " + row["status"] for row in members)
                ),
            }
        )
    replay = read(replay_path) if replay_path else []
    for item in replay:
        row = by_id[item["case_id"]]
        if (
            row["status"] != "DISPROVED"
            or item.get("report_sha256") != row["audit"]["report_sha256"]
        ):
            raise ValueError("counterexample replay is not bound to the current disproved report")
    if len({item["case_id"] for item in replay}) != len(replay):
        raise ValueError("duplicate replay evidence")
    proved = [row for row in cases if row["status"] == "PROVED"]
    straight = [row for row in cases if not row["control_flow"]]
    return {
        "format": "etv-host-iteration-export-v1",
        "summary": {key: value for key, value in summary.items() if key != "cases"},
        "baseline": baseline["summary"],
        "transitions": dict(
            Counter(row["baseline_status"] + " → " + row["status"] for row in cases)
        ),
        "scope_counts": {
            "control_flow": len(cases) - len(straight),
            "straight_line": len(straight),
            "straight_line_verdicts": counts(straight),
        },
        "unresolved_categories": dict(
            Counter(
                row["category"] for row in cases if row["status"] in ("UNKNOWN", "NOT_VERIFIED")
            ).most_common()
        ),
        "host_iteration": {
            "selected": len(target_rows),
            "enriched": sum(row["enriched"] for row in targets.values()),
            "failed_recaptures": sum(not row["enriched"] for row in targets.values()),
            "groups": groups,
            "target_verdicts": counts(target_rows),
            "baseline_category_changed": sum(
                row["category"] != row["baseline_category"] for row in target_rows
            ),
            "note": "原因迁移或完成采集不代表证明通过；以逐例形式验证状态为准。",
        },
        "proof_audit": {
            "proved_audited": len(proved),
            "trusted_axioms_used": sum(
                len(row["audit"]["soundness"].get("trusted_axioms", [])) for row in proved
            ),
            "proved_with_unverified_rules": sum(
                bool(row["audit"]["used_unverified_rules"]) for row in proved
            ),
            "disproved_reports": sum(row["status"] == "DISPROVED" for row in cases),
            "counterexamples_replayed": sum(bool(item.get("replayed")) for item in replay),
            "whole_operator_numeric_counterexamples_confirmed": None,
            "independent_certificate": False,
        },
        "evidence": {
            "source_commit": suite["source_commit"],
            "corpus_lock_sha256": digest(corpus / "corpus.lock.json"),
            "parent_corpus_lock_sha256": frozen.get("parent_corpus_lock_sha256"),
            "recapture_audit_sha256": digest(corpus / "recapture-audit.json"),
            "baseline_data_sha256": digest(baseline_path),
            "run_metadata_sha256": digest(run / "run.json"),
            "run": metadata,
            "execution_location": "not inferred from filesystem paths; attest separately",
        },
        "replay": replay,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-data", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.run, args.corpus, args.baseline_data, args.replay)
    with args.out.open("x") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("summary", "host_iteration", "proof_audit")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
