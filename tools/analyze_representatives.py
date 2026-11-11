"""Audit every frozen representative through the public pairing/verification CLI."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--execution-scope", choices=("ttir-only", "ttir-with-host-ops"), default="ttir-only"
    )
    args = parser.parse_args()
    frozen = read(args.captures / "corpus.lock.json", {})

    def check_frozen():
        for file, digest in frozen.get("files", {}).items():
            if hashlib.sha256((args.captures / file).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen corpus changed: " + file)

    check_frozen()
    lock = read(args.lock)
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=True)
    python = str(args.python.absolute())
    command = [python, "-m", "etv"]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    started = time.monotonic()

    def analyze(nodeid):
        case_id = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
        case = args.captures.absolute() / case_id
        row = {
            "case_id": case_id,
            "nodeid": nodeid,
            "process": read(case / "process.json"),
            "upstream": read(case / "upstream.json"),
            "capture": read(case / "case_status.json"),
            "status": "NOT_VERIFIED",
            "execution_scope": args.execution_scope,
        }
        report_dir = out / case_id
        report_dir.mkdir(exist_ok=True)
        start = time.monotonic()
        if (case / "capture.json").exists():
            with (report_dir / "pair.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        python,
                        "-m",
                        "etv_bench.cli",
                        "pair",
                        str(case / "capture.json"),
                        "--verifier",
                        shlex.join(command),
                        "--execution-scope",
                        args.execution_scope,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    row["pair_returncode"] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    row["analysis_error"] = "pairing_or_verification_timeout"
            manifest = read(case / "manifest.json", {})
            provenance = read(case / "provenance.json", {})
            row["pair_status"] = manifest.get("status", "failed")
            row["diagnostics"] = provenance.get("diagnostics", [])
            if row.get("pair_returncode") == 0 and manifest.get("status") == "ready":
                report = read(case / "validation" / "report.json", {})
                if report.get("execution_scope") != args.execution_scope:
                    raise ValueError("verification report scope differs from requested scope")
                row.update(
                    status=report.get("status", "NOT_VERIFIED"),
                    reason=report.get("reason"),
                    diagnostic=report.get("diagnostic"),
                    counterexample=report.get("counterexample"),
                )
                (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            # These snapshots are produced by the official parser, not by text matching.
            names = Counter()

            def visit(op):
                names[op["name"]] += 1
                for region in op["regions"]:
                    for block in region:
                        for child in block["operations"]:
                            visit(child)

            for path in case.glob("snapshot-*.json"):
                visit(read(path)["function"])
            row["operations"] = dict(names)
        row["seconds"] = round(time.monotonic() - start, 3)
        (report_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        print(
            json.dumps(
                {k: row.get(k) for k in ("nodeid", "status", "reason", "pair_status", "seconds")}
            ),
            flush=True,
        )
        return row

    def audited(nodeid):
        try:
            return analyze(nodeid)
        except Exception:
            # An infrastructure error must retain the case in the denominator.
            case_id = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
            row = {
                "nodeid": nodeid,
                "case_id": case_id,
                "status": "NOT_VERIFIED",
                "execution_scope": args.execution_scope,
                "analysis_error": traceback.format_exc(),
            }
            report_dir = out / case_id
            report_dir.mkdir(exist_ok=True)
            (report_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(audited, lock["selected"]))
    check_frozen()
    summary = {
        "selected": len(rows),
        "execution_scope": args.execution_scope,
        "verdicts": dict(Counter(r["status"] for r in rows)),
        "reasons": dict(Counter(r.get("reason") for r in rows)),
        "wall_seconds": round(time.monotonic() - started, 3),
        "corpus_lock_sha256": hashlib.sha256(
            (args.captures / "corpus.lock.json").read_bytes()
        ).hexdigest()
        if frozen
        else None,
        "cases": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
