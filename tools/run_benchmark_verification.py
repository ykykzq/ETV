#!/usr/bin/env python3
"""Run ETV on every generated benchmark PairSpec observation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _tasks(status_path: Path) -> list[dict[str, Any]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    case_dir = status_path.parents[1]
    base = {
        "nodeid": status.get("nodeid"),
        "case_dir": str(case_dir),
        "status_path": str(status_path),
    }
    if status.get("status") != "generated":
        return [
            {
                **base,
                "kind": "unavailable",
                "reason": status.get("reason", "PairSpec unavailable"),
            }
        ]
    observations = status.get("pairspecs")
    if isinstance(observations, list) and observations:
        return [
            {
                **base,
                **observation,
                "kind": "pairspec",
            }
            for observation in observations
        ]
    # Backward-compatible support for v1 status records.
    return [
        {
            **base,
            "kind": "pairspec",
            "pairspec": status["pairspec"],
            "pair_id": status.get("nodeid"),
        }
    ]


def _result_dir(task: dict[str, Any]) -> Path:
    case_dir = Path(task["case_dir"])
    pair_id = str(task.get("pair_id") or task.get("pairspec") or "observation")
    digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:10]
    leaf = task.get("output_leaf_index")
    reference = task.get("reference_site")
    slug = f"r{reference}-o{leaf}-{digest}" if leaf is not None else digest
    return case_dir / "results" / "formal" / slug


def _run_one(
    task: dict[str, Any],
    *,
    python: Path,
    timeout: int,
    rerun: bool,
) -> dict[str, Any]:
    if task["kind"] == "unavailable":
        return {
            "format": "etv-benchmark-formal-runner-record-v2",
            "complete": True,
            "nodeid": task.get("nodeid"),
            "case_dir": task["case_dir"],
            "status": "NOT_RUN",
            "reason": task.get("reason", "PairSpec unavailable"),
        }
    result_dir = _result_dir(task)
    runner_path = result_dir / "runner.json"
    if not rerun and runner_path.exists():
        try:
            previous = json.loads(runner_path.read_text(encoding="utf-8"))
            if previous.get("complete") is True:
                return previous
        except (OSError, json.JSONDecodeError):
            pass
    spec_path = Path(task["pairspec"])
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "etv.cli",
        "check",
        str(spec_path),
        "--out",
        str(result_dir),
        "--json",
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = None
        output = (error.stdout or "") + "\nTIMEOUT\n"
        timed_out = True
    (result_dir / "etv.stdout.log").write_text(output, encoding="utf-8")
    report_path = result_dir / "report.json"
    if timed_out:
        result_status, reason = "TIMEOUT", f"exceeded {timeout} seconds"
    elif report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result_status = report.get("status", "RUNNER_ERROR")
            reason = report.get("reason")
        except (OSError, json.JSONDecodeError) as error:
            result_status, reason = "RUNNER_ERROR", repr(error)
    else:
        result_status = "RUNNER_ERROR"
        reason = f"ETV exited {returncode} without report.json"
    record = {
        "format": "etv-benchmark-formal-runner-record-v2",
        "complete": True,
        "nodeid": task.get("nodeid"),
        "case_dir": task["case_dir"],
        "pair_id": task.get("pair_id"),
        "pairspec": str(spec_path),
        "mapping": task.get("mapping"),
        "reference_site": task.get("reference_site"),
        "reference_execution": task.get("reference_execution"),
        "output_leaf_index": task.get("output_leaf_index"),
        "output_leaf_path": task.get("output_leaf_path"),
        "result_dir": str(result_dir),
        "returncode": returncode,
        "status": result_status,
        "reason": reason,
        "duration_seconds": time.time() - started,
    }
    _write_json(runner_path, record)
    return record


def _summary(records: list[dict[str, Any]], expected_tasks: int) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in records)
    reasons = Counter(
        str(record.get("reason"))
        for record in records
        if record.get("status") in {"UNKNOWN", "NOT_RUN", "RUNNER_ERROR", "TIMEOUT"}
    )
    case_statuses: dict[str, Counter[str]] = {}
    for record in records:
        case_statuses.setdefault(str(record.get("nodeid")), Counter())[record["status"]] += 1
    case_outcomes = Counter(
        "+".join(
            f"{status}:{count}" for status, count in sorted(statuses.items())
        )
        for statuses in case_statuses.values()
    )
    return {
        "format": "etv-benchmark-formal-summary-v2",
        "updated_at_unix": time.time(),
        "expected_tasks": expected_tasks,
        "completed_tasks": len(records),
        "counts": dict(sorted(statuses.items())),
        "case_counts": {
            "cases": len(case_statuses),
            "with_etv_results": sum(
                any(status != "NOT_RUN" for status in values)
                for values in case_statuses.values()
            ),
            "with_conclusive_results": sum(
                any(status in {"PROVED", "DISPROVED"} for status in values)
                for values in case_statuses.values()
            ),
            "with_disproved_results": sum(
                "DISPROVED" in values for values in case_statuses.values()
            ),
            "outcomes": dict(sorted(case_outcomes.items())),
        },
        "unknown_or_not_run_reasons": dict(sorted(reasons.items())),
        "disproved": [
            record for record in records if record.get("status") == "DISPROVED"
        ],
        "records": sorted(
            records,
            key=lambda item: (
                str(item.get("nodeid")),
                str(item.get("pair_id")),
            ),
        ),
    }


def _remove_legacy_case_results(status_paths: list[Path]) -> None:
    legacy_names = {
        "certificate.json",
        "etv.stdout.log",
        "partition.json",
        "report.json",
        "runner.json",
    }
    for status_path in status_paths:
        formal_dir = status_path.parents[1] / "results" / "formal"
        for name in legacy_names:
            (formal_dir / name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--python", type=Path, default=Path(".venv-etv/bin/python"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve()
    python = args.python if args.python.is_absolute() else Path.cwd() / args.python
    pairspec_summary_path = benchmark_root / "pairspec-summary.json"
    if pairspec_summary_path.exists():
        pairspec_summary = json.loads(
            pairspec_summary_path.read_text(encoding="utf-8")
        )
        status_paths = sorted(
            Path(record["case_dir"]) / "pairspec" / "status.json"
            for record in pairspec_summary.get("records", [])
        )
    else:
        status_paths = sorted(benchmark_root.glob("*/*/pairspec/status.json"))
    _remove_legacy_case_results(status_paths)
    tasks = [task for path in status_paths for task in _tasks(path)]
    records: list[dict[str, Any]] = []
    summary_path = benchmark_root / "formal-summary.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                _run_one,
                task,
                python=python,
                timeout=args.timeout,
                rerun=args.rerun,
            )
            for task in tasks
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            _write_json(summary_path, _summary(records, len(tasks)))
            print(
                f"[{index}/{len(tasks)}] {record['status']} "
                f"{record.get('reason')} {record.get('pair_id') or record.get('nodeid')}",
                flush=True,
            )
    return 1 if any(record["status"] == "DISPROVED" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
