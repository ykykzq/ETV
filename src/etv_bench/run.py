"""Benchmark execution consumes only the public CLI and report.json."""

import json
import hashlib
from pathlib import Path
import subprocess

from .pair import _contained, _read


def run_case(manifest: Path, output: Path, command: list[str]) -> dict:
    data = _read(manifest)
    if data.get("format") != "etv-benchmark-case-v1" or data["status"] != "ready":
        raise ValueError("case is not ready")
    pair = _contained(manifest.parent, data["pair"])
    provenance = _read(_contained(manifest.parent, data["provenance"]))
    reviews = provenance.get("review", [])
    expected_pair = (
        (reviews[-1].get("pair_sha256") if reviews else None)
        or provenance.get("pair_sha256")
        or provenance.get("migrated_pair_sha256")
    )
    if expected_pair and hashlib.sha256(pair.read_bytes()).hexdigest() != expected_pair:
        raise ValueError("PairSpec changed without a provenance review entry")
    for artifact in data["artifacts"]:
        path = _contained(manifest.parent, artifact["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("artifact hash mismatch: " + artifact["path"])
    old_report = (
        (output / "report.json").read_bytes() if (output / "report.json").exists() else None
    )
    process = subprocess.run(
        command + ["verify", str(pair), "--out", str(output), "--json"],
        text=True,
        capture_output=True,
        timeout=300,
    )
    raw = (output / "report.json").read_bytes()
    if raw == old_report:
        raise ValueError("verifier did not produce a fresh report")
    report = json.loads(raw)
    if process.returncode != {"PROVED": 0, "DISPROVED": 1, "UNKNOWN": 2}.get(report["status"]):
        raise ValueError("verifier exit code and report disagree")
    return report
