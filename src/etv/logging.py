"""Run-correlated standard-library diagnostics."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                "run_id": getattr(record, "run_id", ""),
                "pair_id": getattr(record, "pair_id", ""),
                "phase": getattr(record, "phase", ""),
                "fields": getattr(record, "fields", {}),
            },
            sort_keys=True,
        )


def configure(jsonl: bool = False, file: Path | None = None) -> None:
    logger = logging.getLogger("etv")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = (
        JSONFormatter()
        if jsonl
        else logging.Formatter("%(levelname)s %(run_id)s %(pair_id)s %(phase)s %(message)s")
    )
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    logger.addHandler(stderr)
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def event(run_id: str, pair_id: str, phase: str, name: str, **fields: Any) -> None:
    logging.getLogger("etv").info(
        name, extra={"run_id": run_id, "pair_id": pair_id, "phase": phase, "fields": fields}
    )
