"""Run every frozen representative in an isolated GPU subprocess.

The lock contains the full collected inventory. Failures and timeouts remain in
the denominator; the input tests and their parametrization are never rewritten.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    source = args.source.resolve()
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != lock["source_commit"]:
        raise ValueError("upstream revision differs from frozen inventory")
    for file, digest in lock["files"].items():
        if hashlib.sha256((source / file).read_bytes()).hexdigest() != digest:
            raise ValueError("upstream source differs: " + file)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = str(args.python.resolve())
    gpus = args.gpus.split(",")
    selected = lock["selected"]
    started = time.time()

    def worker(worker_id):
        for index in range(worker_id, len(selected), len(gpus)):
            nodeid = selected[index]
            case_id = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
            directory = out / case_id
            directory.mkdir(exist_ok=True)
            result_path = directory / "process.json"
            if result_path.exists():
                continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpus[worker_id],
                OMP_NUM_THREADS="1",
                TORCHINDUCTOR_COMPILE_THREADS="1",
                PYTHONPATH=os.pathsep.join(
                    [
                        str(Path(__file__).resolve().parents[1] / "src"),
                        str(source / "src"),
                        str(source),
                    ]
                ),
            )
            command = [
                python,
                "-m",
                "pytest",
                nodeid,
                "-q",
                "-p",
                "etv_bench.pytest_capture",
                "--etv-capture-root",
                str(out),
                "--junitxml",
                str(directory / "pytest.xml"),
            ]
            start = time.monotonic()
            with (directory / "pytest.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
            result = dict(
                nodeid=nodeid,
                case_id=case_id,
                returncode=code,
                timeout=timed_out,
                seconds=round(time.monotonic() - start, 3),
                gpu=gpus[worker_id],
            )
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    print(
        json.dumps({"selected": len(selected), "wall_seconds": time.time() - started}), flush=True
    )


if __name__ == "__main__":
    main()
