"""Append review evidence without changing the original capture."""

import hashlib
from pathlib import Path
import subprocess

from .pair import _contained, _read, _write


def review_case(case: Path, note: str, command: list[str]) -> dict:
    manifest = _read(case / "manifest.json")
    pair = _contained(case, manifest["pair"])
    provenance_path = _contained(case, manifest["provenance"])
    provenance = _read(provenance_path)
    for artifact in manifest["artifacts"]:
        if (
            hashlib.sha256(_contained(case, artifact["path"]).read_bytes()).hexdigest()
            != artifact["sha256"]
        ):
            raise ValueError("artifact hash mismatch")
    validation = case / "review_validation"
    process = subprocess.run(
        command + ["verify", str(pair), "--out", str(validation), "--json"],
        text=True,
        capture_output=True,
        timeout=300,
    )
    import json

    report = json.loads(process.stdout)
    invalid = report["reason"] in {
        "INVALID_PAIRSPEC",
        "INPUT_NOT_FOUND",
        "FILE_HASH_MISMATCH",
        "ABI_ROLE_MISSING",
        "TTIR_PARSE_FAILED",
        "TTIR_VERIFY_FAILED",
        "TTIR_VERSION_MISMATCH",
        "INTERNAL_ERROR",
    }
    entry = {
        "status": "reviewed",
        "note": note,
        "pair_sha256": hashlib.sha256(pair.read_bytes()).hexdigest(),
        "validation": {"status": report["status"], "reason": report["reason"]},
    }
    provenance.setdefault("review", []).append(entry)
    _write(provenance_path, provenance)
    manifest["status"] = "needs_review" if invalid else "ready"
    manifest["generation"]["mode"] = "semiautomatic"
    _write(case / "manifest.json", manifest)
    return entry
