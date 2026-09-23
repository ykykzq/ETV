"""Preserve verdicts and reason codes when aggregating reports."""

from collections import Counter
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text()) for path in sorted(paths)]
    return {
        "cases": [
            {"pair_id": r["pair_id"], "status": r["status"], "reason": r["reason"]} for r in reports
        ],
        "statuses": dict(sorted(Counter(r["status"] for r in reports).items())),
        "reasons": dict(sorted(Counter(r["reason"] for r in reports).items())),
    }
