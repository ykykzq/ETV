"""Prepare an immutable host-op recapture, then merge actual recordings.

The original corpus is read-only. Re-running a selected test must produce a new
capture with host-op recording enabled; source-derived or old capture files are
never treated as enriched recordings. Failed recaptures retain their original
case and remain explicitly labelled in the audit and full-suite denominator.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


CATEGORIES = ("外部输入映射缺失", "观察契约缺少输出写入")
DIAG_CATEGORY = "对角线片段：输出写入掩码不同"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact missing or escapes corpus: " + relative)
    return path


def verify_frozen(root):
    lock = read(root / "corpus.lock.json")
    for name, expected in lock["files"].items():
        if digest(contained(root, name)) != expected:
            raise ValueError("frozen corpus changed: " + name)
    return lock


def case_id(nodeid):
    return hashlib.sha256(nodeid.encode()).hexdigest()[:16]


def prepare(
    frozen, baseline_data, out, include_diag=False, *, categories=None, capture_control_flow=False
):
    frozen, baseline_data, out = frozen.resolve(), baseline_data.resolve(), out.resolve()
    corpus = verify_frozen(frozen)
    suite = read(frozen / "suite.lock.json")
    baseline = read(baseline_data)
    rows = baseline["cases"]
    selected = suite["selected"]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate frozen nodeids")
    if len(rows) != len(selected) or {r["nodeid"] for r in rows} != set(selected):
        raise ValueError("baseline and frozen suite have different case populations")
    if corpus["source_commit"] != suite["source_commit"]:
        raise ValueError("corpus and suite revisions disagree")
    categories = set(CATEGORIES if categories is None else categories) | (
        {DIAG_CATEGORY} if include_diag else set()
    )
    targets = {r["nodeid"]: r for r in rows if r["category"] in categories}
    target_nodes = [node for node in selected if node in targets]
    if not target_nodes:
        raise ValueError("no target cases in baseline")
    if not capture_control_flow and any(targets[node].get("control_flow") for node in target_nodes):
        raise ValueError("target selection unexpectedly contains control flow")
    target_cases = []
    for node in target_nodes:
        identity = case_id(node)
        if targets[node]["case_id"] != identity:
            raise ValueError("baseline case identity mismatch: " + node)
        original = contained(frozen, identity + "/capture.json")
        target_cases.append(
            {
                "case_id": identity,
                "nodeid": node,
                "baseline_status": targets[node]["status"],
                "baseline_category": targets[node]["category"],
                "original_capture_sha256": digest(original),
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "recaptured").mkdir()
    target_suite = dict(suite, selected=target_nodes)
    target_suite["selection_policy"] = (
        "baseline host input/output gaps; fixed nodeids, no outcome filtering"
    )
    write(out / "target-suite.lock.json", target_suite)
    plan = {
        "format": "etv-host-recapture-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_path": str(frozen),
        "source_commit": suite["source_commit"],
        "original_corpus_lock_sha256": digest(frozen / "corpus.lock.json"),
        "original_suite_lock_sha256": digest(frozen / "suite.lock.json"),
        "baseline_data_sha256": digest(baseline_data),
        "full_selected": len(selected),
        "target_selected": len(target_nodes),
        "target_categories": dict(Counter(r["baseline_category"] for r in target_cases)),
        "control_flow_capture_only": capture_control_flow,
        "control_flow_semantics_supported": False,
        "target_suite_lock_sha256": digest(out / "target-suite.lock.json"),
        "cases": target_cases,
    }
    write(out / "plan.json", plan)
    return plan


def normalized_arguments(value):
    """Preserve runtime configuration, ignoring only a tensor's CUDA ordinal.

    Tensor values are deliberately absent from pytest_capture.encode: the proof
    quantifies over their elements. Shape, dtype, stride and all scalar settings
    (including q/p generated inside a test) still define the frozen test case.
    """
    if isinstance(value, list):
        return [normalized_arguments(item) for item in value]
    if isinstance(value, dict):
        result = {key: normalized_arguments(item) for key, item in value.items()}
        if {"shape", "dtype", "stride", "device"} <= value.keys():
            device = value["device"]
            if isinstance(device, str) and re.fullmatch(r"cuda(?::[0-9]+)?", device):
                result["device"] = "cuda"
        return result
    return value


def argument_differences(original, recaptured, path="arguments"):
    """Type-sensitive differences: False, 0 and 0.0 are distinct configurations."""
    if type(original) is type(recaptured):
        if isinstance(original, dict) and original.keys() == recaptured.keys():
            return [
                difference
                for key in sorted(original)
                for difference in argument_differences(
                    original[key], recaptured[key], path + "." + key
                )
            ]
        if isinstance(original, list) and len(original) == len(recaptured):
            return [
                difference
                for index, (left, right) in enumerate(zip(original, recaptured, strict=True))
                for difference in argument_differences(left, right, f"{path}[{index}]")
            ]
        if isinstance(original, float):
            same = original.hex() == recaptured.hex()
        else:
            same = original == recaptured
        if same:
            return []
    return [{"path": path, "original": original, "recaptured": recaptured}]


def compare_arguments(original_directory, recaptured_directory, nodeid):
    comparison = {
        "policy": "exact runtime arguments; tensor CUDA device ordinal only is normalized",
    }
    values = []
    for label, directory in (
        ("original", original_directory),
        ("recaptured", recaptured_directory),
    ):
        path = directory / "case_status.json"
        if not path.is_file():
            return comparison | {"failure": f"recapture_{label}_arguments_missing"}
        comparison[label + "_case_status_sha256"] = digest(path)
        try:
            status = read(path)
            if status.get("nodeid") != nodeid:
                return comparison | {"failure": f"recapture_{label}_arguments_identity_mismatch"}
            if not isinstance(status.get("arguments"), dict):
                return comparison | {"failure": f"recapture_{label}_arguments_missing"}
            arguments = normalized_arguments(status["arguments"])
            encoded = json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (AttributeError, TypeError, ValueError):
            return comparison | {"failure": f"recapture_{label}_arguments_invalid"}
        comparison[label + "_normalized_arguments_sha256"] = hashlib.sha256(encoded).hexdigest()
        values.append(arguments)
    differences = argument_differences(*values)
    return comparison | {
        "failure": "recapture_arguments_mismatch" if differences else None,
        "differences": differences,
    }


def recording_status(directory, target, source_commit, original_directory):
    """Return explicit failure reasons; identity/pin violations abort the merge."""
    process_path = directory / "process.json"
    if not process_path.exists():
        raise ValueError("recapture has not completed: " + target["nodeid"])
    process = read(process_path)
    if process.get("nodeid") != target["nodeid"] or process.get("case_id") != target["case_id"]:
        raise ValueError("recapture process identity mismatch")
    result = {"process": process, "diagnostics": []}
    if process.get("returncode") != 0 or process.get("timeout"):
        return result | {"failure": "recapture_process_failed"}
    upstream_path = directory / "upstream.json"
    if not upstream_path.exists():
        return result | {"failure": "recapture_upstream_result_missing"}
    upstream = read(upstream_path)
    if upstream.get("nodeid") != target["nodeid"]:
        raise ValueError("recapture upstream identity mismatch")
    result["upstream"] = upstream
    if upstream.get("status") != "passed":
        return result | {"failure": "recapture_upstream_test_failed"}
    capture_path = directory / "capture.json"
    if not capture_path.exists():
        return result | {"failure": "recapture_recording_missing"}
    capture = read(capture_path)
    source = capture.get("source", {})
    if source.get("test_id") != target["nodeid"] or source.get("commit") != source_commit:
        raise ValueError("recapture source revision or test identity mismatch")
    if capture.get("format") != "etv-runtime-capture-v1":
        return result | {"failure": "recapture_format_invalid"}
    if capture.get("host_capture_format") != "etv-host-operations-v1" or not isinstance(
        capture.get("host_operations"), list
    ):
        return result | {"failure": "recapture_host_recording_not_enabled"}
    launches = capture.get("launches", [])
    if {launch.get("side") for launch in launches} != {"lhs", "rhs"}:
        return result | {"failure": "recapture_ttir_sides_incomplete"}
    for launch in launches:
        path = contained(directory, launch["file"])
        if not launch.get("sha256") or digest(path) != launch["sha256"]:
            raise ValueError("recaptured TTIR artifact hash mismatch")
    if digest(capture_path) == target["original_capture_sha256"]:
        return result | {"failure": "recapture_is_unchanged_original"}
    argument_check = compare_arguments(original_directory, directory, target["nodeid"])
    result["runtime_arguments_comparison"] = argument_check
    if argument_check["failure"]:
        return result | {"failure": argument_check["failure"]}
    return result | {
        "failure": None,
        "new_capture_sha256": digest(capture_path),
        "host_operations": len(capture["host_operations"]),
        "host_operation_kinds": dict(Counter(op["kind"] for op in capture["host_operations"])),
        "diagnostics": capture.get("diagnostics", []),
    }


def merge(frozen, plan_dir, out):
    frozen, plan_dir, out = frozen.resolve(), plan_dir.resolve(), out.resolve()
    plan = read(plan_dir / "plan.json")
    if plan.get("format") != "etv-host-recapture-plan-v1":
        raise ValueError("unsupported recapture plan")
    if (
        digest(frozen / "corpus.lock.json") != plan["original_corpus_lock_sha256"]
        or digest(frozen / "suite.lock.json") != plan["original_suite_lock_sha256"]
    ):
        raise ValueError("original corpus or suite lock differs from recapture plan")
    if digest(plan_dir / "target-suite.lock.json") != plan["target_suite_lock_sha256"]:
        raise ValueError("target suite changed after prepare")
    corpus = verify_frozen(frozen)
    suite = read(frozen / "suite.lock.json")
    if (
        corpus["source_commit"] != plan["source_commit"]
        or suite["source_commit"] != plan["source_commit"]
    ):
        raise ValueError("source pin differs from recapture plan")
    audit = []
    # Inspect every recording before creating the derived corpus. Missing process
    # results require completing the run, not dropping cases or reusing silently.
    for target in plan["cases"]:
        directory = plan_dir / "recaptured" / target["case_id"]
        status = recording_status(
            directory,
            target,
            plan["source_commit"],
            frozen / target["case_id"],
        )
        audit.append(
            target
            | status
            | {
                "capture_origin": "retained_original_after_recapture_failure"
                if status["failure"]
                else "new_runtime_host_capture",
                "enriched": status["failure"] is None,
            }
        )
    if out.exists():
        raise FileExistsError("derived corpus output already exists")
    if out.is_relative_to(frozen) or out.is_relative_to(plan_dir / "recaptured"):
        raise ValueError("derived corpus must not be inside an input capture directory")
    shutil.copytree(frozen, out)
    for row in audit:
        if row["enriched"]:
            destination = out / row["case_id"]
            shutil.rmtree(destination)
            shutil.copytree(plan_dir / "recaptured" / row["case_id"], destination)
    summary = {
        "format": "etv-host-recapture-audit-v1",
        "source_commit": plan["source_commit"],
        "full_selected": plan["full_selected"],
        "target_selected": len(audit),
        "enriched": sum(row["enriched"] for row in audit),
        "failed_recaptures": sum(not row["enriched"] for row in audit),
        "unchanged_untargeted": plan["full_selected"] - len(audit),
        "original_corpus_lock_sha256": plan["original_corpus_lock_sha256"],
        "plan_sha256": digest(plan_dir / "plan.json"),
        "cases": audit,
    }
    write(out / "recapture-audit.json", summary)
    # Preserve the selection lock byte for byte: only actual execution artifacts
    # are replaced. Refresh descriptive inventory rows, if present.
    inventory_path = out / "retained.json"
    if inventory_path.exists():
        inventory = read(inventory_path)
        by_id = {row["case_id"]: row for row in audit}
        for row in inventory:
            update = by_id.get(row["case_id"])
            row["capture_origin"] = update["capture_origin"] if update else "unchanged_original"
            if update and update["enriched"]:
                directory = out / row["case_id"]
                row["process"] = update["process"]
                row["upstream_status"] = update["upstream"]["status"]
                row["ttir_files"] = len(list(directory.rglob("*.ttir")))
                if (directory / "case_status.json").exists():
                    row["capture_status"] = read(directory / "case_status.json").get("status")
        write(inventory_path, inventory)
    new_lock = {
        "format": "etv-host-enriched-capture-corpus-v1",
        "source_commit": plan["source_commit"],
        "parent_corpus_lock_sha256": plan["original_corpus_lock_sha256"],
        "cases": [
            dict(row, ttir_files=len(list((out / row["directory"]).rglob("*.ttir"))))
            for row in corpus["cases"]
        ],
        "files": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path != out / "corpus.lock.json"
        },
    }
    write(out / "corpus.lock.json", new_lock)
    verify_frozen(frozen)
    verify_frozen(out)
    if digest(out / "suite.lock.json") != plan["original_suite_lock_sha256"]:
        raise ValueError("derived full-suite population changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    start = commands.add_parser("prepare")
    start.add_argument("--frozen", type=Path, required=True)
    start.add_argument("--baseline-data", type=Path, required=True)
    start.add_argument("--out", type=Path, required=True)
    start.add_argument("--include-diag", action="store_true")
    start.add_argument("--category", action="append")
    start.add_argument(
        "--capture-control-flow", action="store_true",
        help="record provenance for cases containing control flow; does not enable its verification",
    )
    finish = commands.add_parser("merge")
    finish.add_argument("--frozen", type=Path, required=True)
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(
            args.frozen, args.baseline_data, args.out, args.include_diag,
            categories=args.category, capture_control_flow=args.capture_control_flow,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))
        print(
            "Run tools/run_representatives.py with target-suite.lock.json and --out recaptured/ before merge."
        )
    else:
        result = merge(args.frozen, args.plan, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
