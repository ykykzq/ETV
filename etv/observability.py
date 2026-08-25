"""Structured runtime logging for ETV.

The verification report is the durable proof/audit artifact.  This module is
for operational events while a run is executing: it adds a run identifier,
optional pair/phase context, and can emit either human-readable lines or JSON
Lines without writing to stdout.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


LOGGER_NAME = "etv"
_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "etv_log_context", default={}
)


def new_run_id() -> str:
    """Return a short identifier suitable for correlating one verification run."""

    return uuid.uuid4().hex[:12]


def context_values() -> Mapping[str, Any]:
    """Return the current immutable-by-convention logging context."""

    return dict(_CONTEXT.get())


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Temporarily add fields to every ETV log record in this execution context."""

    current = dict(_CONTEXT.get())
    current.update({key: value for key, value in values.items() if value is not None})
    token = _CONTEXT.set(current)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(context_values())
        event = getattr(record, "etv_event", None)
        if event is not None:
            payload["event"] = event
        fields = getattr(record, "etv_fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = context_values()
        context_text = " ".join(
            f"{key}={context[key]}" for key in ("run_id", "pair_id", "phase") if key in context
        )
        event = getattr(record, "etv_event", None)
        fields = getattr(record, "etv_fields", None)
        suffix = ""
        if event:
            suffix += f" event={event}"
        if fields:
            suffix += " fields=" + json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
        if context_text:
            context_text = " [" + context_text + "]"
        return f"{record.levelname} {record.name}{context_text}: {record.getMessage()}{suffix}"


def _level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    normalized = value.upper()
    resolved = logging.getLevelName(normalized)
    if not isinstance(resolved, int):
        raise ValueError(f"unknown log level {value!r}")
    return resolved


def configure_logging(
    level: str | int = "WARNING",
    *,
    log_file: Optional[Path] = None,
    log_format: str = "text",
    run_id: Optional[str] = None,
    file_level: Optional[str | int] = None,
) -> str:
    """Configure ETV handlers and return the run id used by this process.

    Only handlers installed by this function are replaced.  This keeps library
    callers' unrelated logging configuration intact while making repeated CLI
    invocations in one Python process deterministic.
    """

    if log_format not in {"text", "json"}:
        raise ValueError("log_format must be 'text' or 'json'")
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(_level(level))
    root.propagate = False
    for handler in list(root.handlers):
        if getattr(handler, "_etv_handler", False):
            root.removeHandler(handler)
            handler.close()

    formatter: logging.Formatter = (
        _JsonFormatter() if log_format == "json" else _TextFormatter()
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream._etv_handler = True  # type: ignore[attr-defined]
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        if file_level is not None:
            file_handler.setLevel(_level(file_level))
        file_handler.setFormatter(formatter)
        file_handler._etv_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    selected = run_id or new_run_id()
    current = dict(_CONTEXT.get())
    current["run_id"] = selected
    _CONTEXT.set(current)
    return selected


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Emit one structured event without exposing secrets or payloads by default."""

    logger.log(
        level,
        message,
        extra={"etv_event": event, "etv_fields": fields},
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger below the ETV namespace."""

    qualified = (
        name
        if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}.")
        else f"{LOGGER_NAME}.{name}"
    )
    return logging.getLogger(qualified)
