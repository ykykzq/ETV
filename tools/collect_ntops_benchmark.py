#!/usr/bin/env python3
"""Collect test-specialized ntops/TorchInductor TTIR pairs from pytest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
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


def _operator(nodeid: str) -> str:
    match = re.search(r"(?:^|/)test_([^/]+)\.py::", nodeid)
    if match is None:
        raise ValueError(f"cannot infer operator from node ID: {nodeid}")
    return match.group(1)


def _case_name(nodeid: str) -> str:
    test_part = nodeid.split("::", 1)[1]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", test_part).strip("-.")
    digest = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:100]}--{digest}"


def _pytest_node(nodeid: str) -> str:
    marker = "tests/"
    index = nodeid.find(marker)
    return nodeid[index:] if index >= 0 else nodeid


def _collect_nodes(python: Path, ntops_repo: Path) -> list[str]:
    test_files = sorted((ntops_repo / "tests").glob("test_*.py"))

    def collect_file(test_file: Path) -> list[str]:
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                str(test_file.relative_to(ntops_repo)),
            ],
            cwd=ntops_repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        nodes = [line.strip() for line in result.stdout.splitlines() if "::" in line]
        if result.returncode != 0:
            raise RuntimeError(
                f"pytest collection failed for {test_file} with exit code "
                f"{result.returncode}:\n{result.stdout}"
            )
        return nodes

    nodes: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for collected in executor.map(collect_file, test_files):
            nodes.extend(collected)
    if not nodes:
        raise RuntimeError("isolated pytest collection produced no test nodes")
    return nodes


def _junit_outcome(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "runner_error", "pytest did not write JUnit XML"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return "runner_error", repr(error)
    case = root.find(".//testcase")
    if case is None:
        return "runner_error", "JUnit XML contains no testcase"
    for tag, outcome in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        child = case.find(tag)
        if child is not None:
            return outcome, child.get("message") or (child.text or "").strip() or None
    return "passed", None


def _summarize(records: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any]:
    outcomes = Counter(record["outcome"] for record in records)
    lhs = Counter(record["lhs_status"] for record in records)
    rhs = Counter(record["rhs_status"] for record in records)
    return {
        "format": "etv-benchmark-collection-summary-v1",
        "updated_at_unix": time.time(),
        "inventory": inventory,
        "processed": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "lhs": dict(sorted(lhs.items())),
        "rhs": dict(sorted(rhs.items())),
        "records": sorted(records, key=lambda item: item["nodeid"]),
    }


def _manifest_status(case_dir: Path) -> tuple[str, str]:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        return "not_reached", "not_reached"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "manifest_invalid", "manifest_invalid"
    lhs_launches = manifest.get("lhs_launches", [])
    if any(item.get("status") == "captured" for item in lhs_launches):
        lhs_status = "captured"
    elif any(item.get("status") == "failed" for item in lhs_launches):
        lhs_status = "compile_failed"
    else:
        lhs_status = "not_captured"
    references = manifest.get("rhs_references", [])
    if any(item.get("ttir_status") == "captured" for item in references):
        rhs_status = "captured"
    elif any(item.get("status") == "failed" for item in references):
        rhs_status = "reference_compile_failed"
    elif references and all(
        item.get("ttir_status") == "no_triton_kernel_launched" for item in references
    ):
        rhs_status = "no_triton_kernel"
    elif manifest.get("reference_executions", 0) == 0:
        rhs_status = "reference_not_intercepted"
    else:
        rhs_status = "ttir_unavailable"
    return lhs_status, rhs_status


def _run_one(
    nodeid: str,
    *,
    python: Path,
    ntops_repo: Path,
    benchmark_root: Path,
    tools_dir: Path,
    gpu: str,
    timeout: int,
    rerun: bool,
) -> dict[str, Any]:
    operator = _operator(nodeid)
    case_dir = benchmark_root / operator / _case_name(nodeid)
    runner_path = case_dir / "runner.json"
    if not rerun and runner_path.exists():
        try:
            previous = json.loads(runner_path.read_text(encoding="utf-8"))
            if previous.get("complete") is True:
                return previous
        except (OSError, json.JSONDecodeError):
            pass
    case_dir.mkdir(parents=True, exist_ok=True)
    result_dir = case_dir / "results"
    result_dir.mkdir(exist_ok=True)
    junit_path = result_dir / "pytest.xml"
    stdout_path = result_dir / "pytest.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(tools_dir), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            "ETV_BENCHMARK_CASE_DIR": str(case_dir),
            "TRITON_CACHE_DIR": str(case_dir / "lhs" / "triton-cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(case_dir / "rhs" / "torchinductor-cache"),
            "CUDA_VISIBLE_DEVICES": gpu,
        }
    )
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "capture_benchmark_test",
        f"--junitxml={junit_path}",
        _pytest_node(nodeid),
    ]
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=ntops_repo,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        output = (error.stdout or "") + "\nTIMEOUT\n"
    stdout_path.write_text(output, encoding="utf-8")
    outcome, reason = (
        ("timeout", f"exceeded {timeout} seconds")
        if timed_out
        else _junit_outcome(junit_path)
    )
    lhs_status, rhs_status = _manifest_status(case_dir)
    record = {
        "format": "etv-benchmark-runner-record-v1",
        "complete": True,
        "nodeid": nodeid,
        "operator": operator,
        "case_dir": str(case_dir),
        "gpu": gpu,
        "command": command,
        "returncode": returncode,
        "outcome": outcome,
        "reason": reason,
        "lhs_status": lhs_status,
        "rhs_status": rhs_status,
        "duration_seconds": time.time() - started,
    }
    _write_json(runner_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntops-repo", type=Path, default=Path("build/upstream/ntops"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--python", type=Path, default=Path(".venv-nt025/bin/python"))
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    ntops_repo = args.ntops_repo.resolve()
    benchmark_root = args.benchmark_root.resolve()
    python = args.python if args.python.is_absolute() else Path.cwd() / args.python
    tools_dir = Path(__file__).resolve().parent
    nodes = args.node or _collect_nodes(python, ntops_repo)
    if args.operator:
        selected = set(args.operator)
        nodes = [node for node in nodes if _operator(node) in selected]
    if args.limit is not None:
        nodes = nodes[: args.limit]
    operators = Counter(_operator(node) for node in nodes)
    inventory = {
        "nodes": len(nodes),
        "operators": len(operators),
        "tests_by_operator": dict(sorted(operators.items())),
    }
    _write_json(benchmark_root / "inventory.json", inventory)
    gpu_devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_devices:
        parser.error("--gpus must contain at least one device")
    jobs = max(1, min(args.jobs, len(gpu_devices)))
    records: list[dict[str, Any]] = []
    summary_path = benchmark_root / "collection-summary.json"

    def submit(index_and_node: tuple[int, str]) -> dict[str, Any]:
        index, nodeid = index_and_node
        return _run_one(
            nodeid,
            python=python,
            ntops_repo=ntops_repo,
            benchmark_root=benchmark_root,
            tools_dir=tools_dir,
            gpu=gpu_devices[index % len(gpu_devices)],
            timeout=args.timeout,
            rerun=args.rerun,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(submit, item): item
            for item in enumerate(nodes)
        }
        for completed_index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            _write_json(summary_path, _summarize(records, inventory))
            print(
                f"[{completed_index}/{len(nodes)}] {record['outcome']} "
                f"lhs={record['lhs_status']} rhs={record['rhs_status']} "
                f"{record['nodeid']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
