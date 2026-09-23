"""Execute documented commands and assert report contracts without a GPU."""

import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "etv"]
    cases = [
        ("add/pair.json", "PROVED", 0),
        ("add/pair_parametric.json", "PROVED", 0),
        ("compute_mismatch/pair.json", "DISPROVED", 1),
        ("mask_mismatch/pair.json", "DISPROVED", 1),
        ("address_mismatch/pair.json", "DISPROVED", 1),
        ("multilaunch/pair.json", "PROVED", 0),
        ("unsupported/pair.json", "UNKNOWN", 2),
    ]
    for name, status, code in cases:
        output = root / "build/smoke" / name.removesuffix(".json")
        process = subprocess.run(
            command + ["verify", str(root / "examples" / name), "--out", str(output), "--json"],
            capture_output=True,
            text=True,
        )
        report = json.loads(process.stdout)
        assert process.returncode == code and report["status"] == status, report
        assert json.loads((output / "report.json").read_text()) == report
        assert (output / "report.md").is_file()
    subprocess.run(
        command
        + [
            "parse",
            str(root / "examples/multilaunch/lhs.ttir"),
            "--function",
            "lhs",
            "--out",
            str(root / "build/smoke/snapshot.json"),
        ],
        check=True,
    )
    subprocess.run(
        command + ["rules", "check", str(root / "examples/rules/fadd_zero.json"), "--json"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "etv_bench.cli",
            "run",
            str(root / "examples/add/manifest.json"),
            "--out",
            str(root / "build/smoke/benchmark"),
            "--verifier",
            f"{sys.executable} -m etv",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("README smoke commands passed")


if __name__ == "__main__":
    main()
