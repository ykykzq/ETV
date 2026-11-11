"""Run an immutable captured suite in a fresh derived workspace with progress."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def hashes():
    result = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("src/etv", "src/etv_bench", "tools")
        for p in sorted(Path(folder).rglob("*.py"))
    }
    result.update({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("native").rglob("*"))
        if p.is_file() and p.suffix in {".cpp", ".h", ".hpp"}
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--execution-scope", choices=("ttir-only", "ttir-with-host-ops"), default="ttir-only"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lock = json.loads((args.frozen / "suite.lock.json").read_text())
    source = hashes()
    started = time.time()
    work = args.out / "work"
    shutil.copytree(args.frozen, work)
    command = [
        str(args.python),
        "tools/analyze_representatives.py",
        str(work / "suite.lock.json"),
        "--captures",
        str(work),
        "--python",
        str(args.python),
        "--out",
        str(args.out / "analysis"),
        "--workers",
        str(args.workers),
        "--timeout",
        "180",
        "--execution-scope",
        args.execution_scope,
    ]
    with (args.out / "analysis.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write(
            args.out / "run.json",
            {
                "command": command,
                "pid": process.pid,
                "started": started,
                "source_sha256": source,
                "selected": len(lock["selected"]),
                "execution_scope": args.execution_scope,
                "corpus_lock_sha256": hashlib.sha256(
                    (args.frozen / "corpus.lock.json").read_bytes()
                ).hexdigest(),
            },
        )
        while True:
            done = process.poll()
            rows = []
            for path in (args.out / "analysis").glob("*/summary.json"):
                try:
                    rows.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    pass
            write(
                args.out / "progress.json",
                {
                    "completed": len(rows),
                    "total": len(lock["selected"]),
                    "verdicts": dict(Counter(r["status"] for r in rows)),
                    "returncode": done,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
            if done is not None:
                break
            time.sleep(5)
    if done != 0:
        raise SystemExit(done)
    assert source == hashes(), "implementation changed during experiment"
    summary = json.loads((args.out / "analysis/summary.json").read_text())
    assert summary["selected"] == len(lock["selected"])
    assert {r["nodeid"] for r in summary["cases"]} == set(lock["selected"])
    write(args.out / "completed.json", {k: v for k, v in summary.items() if k != "cases"})


if __name__ == "__main__":
    main()
