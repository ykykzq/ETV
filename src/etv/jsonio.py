"""Strict JSON input and deterministic, content-addressed output boundaries."""

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

from .errors import InputError

JSON: TypeAlias = None | bool | int | float | str | list["JSON"] | dict[str, "JSON"]


def _object(pairs: list[tuple[str, JSON]]) -> dict[str, JSON]:
    result: dict[str, JSON] = {}
    for key, value in pairs:
        if key in result:
            raise InputError("INVALID_PAIRSPEC", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> JSON:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_object)
    except OSError as exc:
        raise InputError("INPUT_NOT_FOUND", str(path)) from exc
    except (ValueError, UnicodeError) as exc:
        raise InputError("INVALID_PAIRSPEC", str(exc)) from exc


def digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise InputError("INPUT_NOT_FOUND", str(path)) from exc


def resolve_file(root: Path, name: str, allow_external: bool = False) -> Path:
    path = (root / name).resolve()
    if not allow_external and not path.is_relative_to(root.resolve()):
        raise InputError("INVALID_PAIRSPEC", f"path escapes artifact root: {name}")
    if not path.is_file():
        raise InputError("INPUT_NOT_FOUND", name)
    return path


def obj(value: JSON, required: str, optional: str = "") -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise InputError("INVALID_PAIRSPEC", "expected object")
    missing = set(required.split()) - value.keys()
    extra = value.keys() - set((required + " " + optional).split())
    if missing or extra:
        raise InputError("INVALID_PAIRSPEC", f"missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def mapping(value: JSON) -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise InputError("INVALID_PAIRSPEC", "expected object mapping")
    return value


def array(value: JSON) -> list[JSON]:
    if not isinstance(value, list):
        raise InputError("INVALID_PAIRSPEC", "expected array")
    return value


def string(value: JSON) -> str:
    if not isinstance(value, str) or not value:
        raise InputError("INVALID_PAIRSPEC", "expected nonempty string")
    return value


def integer(value: JSON, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise InputError("INVALID_PAIRSPEC", "expected integer in range")
    return value


def boolean(value: JSON) -> bool:
    if type(value) is not bool:
        raise InputError("INVALID_PAIRSPEC", "expected boolean")
    return value
