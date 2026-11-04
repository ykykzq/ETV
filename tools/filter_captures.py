"""Select complete runtime captures from a fully attempted frozen inventory.

This is capture filtering only: upstream assertion outcomes, output dtype
differences, PairSpec validity, and formal verdicts do not remove a complete
capture. The source inventory and raw captures remain available for audit.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def classify(case, nodeid, source_commit):
    process = read(case / "process.json")
    if process["nodeid"] != nodeid:
        raise ValueError("process identity mismatch: " + nodeid)
    status_path = case / "case_status.json"
    if process.get("timeout"):
        return "capture_failed", "capture_process_timeout", None
    if not status_path.is_file():
        return "capture_failed", "capture_status_missing", None
    status = read(status_path)
    if status["nodeid"] != nodeid:
        raise ValueError("capture identity mismatch: " + nodeid)
    capture_status = status.get("status")
    if capture_status in ("capture_failed", "capture_incomplete"):
        return (
            capture_status,
            status.get("error")
            or "; ".join(status.get("diagnostics", []))
            or "both launch sequences are required",
            status,
        )
    if capture_status != "captured":
        raise ValueError("unrecognized capture status: " + str(capture_status))
    contract = read(case / "capture.json")
    if contract["source"]["test_id"] != nodeid or contract["source"]["commit"] != source_commit:
        raise ValueError("capture source mismatch: " + nodeid)
    launches = contract["launches"]
    if contract.get("diagnostics") or {x["side"] for x in launches} != {"lhs", "rhs"}:
        return "capture_incomplete", "capture contract lacks complete launch mapping", status
    for launch in launches:
        path = (case / launch["file"]).resolve()
        if not path.is_relative_to(case.resolve()) or not launch.get("selected_runtime_launch"):
            raise ValueError("invalid launch artifact: " + nodeid)
        if digest(path) != launch["sha256"]:
            raise ValueError("TTIR hash mismatch: " + nodeid)
    return "captured", None, status


def filter_captures(lock_path, captures, out):
    lock = read(lock_path)
    records = lock["all_cases"]
    identities = [r["nodeid"] for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate frozen node IDs")
    if set(lock["selected"]) != set(identities):
        raise ValueError("filtering requires the entire inventory to be selected")
    rows = []
    for record in records:
        nodeid = record["nodeid"]
        case_id = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        case = captures / case_id
        if not (case / "process.json").is_file():
            raise ValueError("case not attempted; cannot finalize: " + nodeid)
        status, reason, detail = classify(case, nodeid, lock["source_commit"])
        upstream_path = case / "upstream.json"
        rows.append(
            {
                "case_id": case_id,
                "nodeid": nodeid,
                "function": record["function"],
                "parameters": record["parameters"],
                "capture_status": status,
                "retained": status == "captured",
                "reason": reason,
                "ttir_files": len(list(case.rglob("*.ttir"))),
                "upstream_status": read(upstream_path).get("status")
                if upstream_path.exists()
                else "not_reported",
                "process": read(case / "process.json"),
            }
        )
    # Freeze in a fresh destination; never mutate a previously published corpus.
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(lock_path, out / "source-suite.lock.json")
    keep = {r["nodeid"] for r in rows if r["retained"]}
    selected_lock = {
        **lock,
        "format": "etv-capture-filtered-suite-v1",
        "selection_policy": "retain all and only complete two-sided runtime captures; no formal-verdict filtering",
        "source_inventory_count": len(records),
        "source_inventory_sha256": digest(lock_path),
        "all_cases": [r for r in records if r["nodeid"] in keep],
        "selected": sorted(keep),
    }
    write(out / "suite.lock.json", selected_lock)
    write(out / "retained.json", [r for r in rows if r["retained"]])
    write(out / "excluded.json", [r for r in rows if not r["retained"]])
    (out / "selected-nodeids.txt").write_text("\n".join(sorted(keep)) + "\n")
    copied = []
    for row in rows:
        if not row["retained"]:
            continue
        source = captures / row["case_id"]
        target = out / row["case_id"]
        target.mkdir(parents=True)
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if path.is_file() and (
                path.suffix == ".ttir"
                or str(relative)
                in (
                    "capture.json",
                    "case_status.json",
                    "process.json",
                    "upstream.json",
                    "pytest.log",
                    "pytest.xml",
                )
            ):
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        copied.append(
            {
                "nodeid": row["nodeid"],
                "case_id": row["case_id"],
                "directory": row["case_id"],
                "ttir_files": row["ttir_files"],
            }
        )
    summary = {
        "format": "etv-capture-filter-summary-v1",
        "source_commit": lock["source_commit"],
        "inventory": len(rows),
        "attempted": len(rows),
        "retained": len(keep),
        "excluded": len(rows) - len(keep),
        "capture_counts": dict(Counter(r["capture_status"] for r in rows)),
        "retained_ttir_files": sum(r["ttir_files"] for r in rows if r["retained"]),
        "retained_functions": len({r["function"] for r in rows if r["retained"]}),
        "all_functions": len({r["function"] for r in rows}),
        "retained_upstream_status": dict(
            Counter(r["upstream_status"] for r in rows if r["retained"])
        ),
        "source_suite_sha256": digest(lock_path),
        "filtered_suite_sha256": digest(out / "suite.lock.json"),
        "formal_verification": "not_run_for_this_full_corpus",
    }
    write(out / "summary.json", summary)
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(
        out / "corpus.lock.json",
        {
            "format": "etv-filtered-capture-corpus-v1",
            "source_commit": lock["source_commit"],
            "cases": copied,
            "files": files,
        },
    )
    print(
        json.dumps(
            {**summary, "corpus_sha256": digest(out / "corpus.lock.json")}, ensure_ascii=False
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    filter_captures(args.lock, args.captures, args.out)
