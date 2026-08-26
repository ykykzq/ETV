#!/usr/bin/env python3
"""Run ETV on every generated benchmark PairSpec and summarize all outcomes."""

from __future__ import annotations

import argparse
import concurrent.futures
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


def _run_one(
    status_path: Path,
    *,
    python: Path,
    timeout: int,
    rerun: bool,
) -> dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    case_dir = status_path.parents[1]
    result_dir = case_dir / "results" / "formal"
    runner_path = result_dir / "runner.json"
    if status.get("status") != "generated":
        for name in (
            "runner.json",
            "report.json",
            "report.md",
            "etv.stdout.log",
            "etv.log",
        ):
            (result_dir / name).unlink(missing_ok=True)
        return {
            "nodeid": status.get("nodeid"),
            "case_dir": str(case_dir),
            "status": "NOT_RUN",
            "reason": status.get("reason", "PairSpec unavailable"),
        }
    if not rerun and runner_path.exists():
        try:
            previous = json.loads(runner_path.read_text(encoding="utf-8"))
            if previous.get("complete") is True:
                return previous
        except (OSError, json.JSONDecodeError):
            pass
    spec_path = Path(status["pairspec"])
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
        "format": "etv-benchmark-formal-runner-record-v1",
        "complete": True,
        "nodeid": status.get("nodeid"),
        "case_dir": str(case_dir),
        "pairspec": str(spec_path),
        "returncode": returncode,
        "status": result_status,
        "reason": reason,
        "duration_seconds": time.time() - started,
    }
    _write_json(runner_path, record)
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in records)
    reasons = Counter(
        str(record.get("reason"))
        for record in records
        if record.get("status") in {"UNKNOWN", "NOT_RUN", "RUNNER_ERROR", "TIMEOUT"}
    )
    return {
        "format": "etv-benchmark-formal-summary-v1",
        "updated_at_unix": time.time(),
        "counts": dict(sorted(statuses.items())),
        "unknown_or_not_run_reasons": dict(sorted(reasons.items())),
        "disproved": [
            record for record in records if record.get("status") == "DISPROVED"
        ],
        "records": sorted(records, key=lambda item: str(item.get("nodeid"))),
    }


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
    records: list[dict[str, Any]] = []
    summary_path = benchmark_root / "formal-summary.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                _run_one,
                path,
                python=python,
                timeout=args.timeout,
                rerun=args.rerun,
            )
            for path in status_paths
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            summary = _summary(records)
            _write_json(summary_path, summary)
            print(
                f"[{index}/{len(status_paths)}] {record['status']} "
                f"{record.get('reason')} {record.get('nodeid')}",
                flush=True,
            )
    return 1 if any(record["status"] == "DISPROVED" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
