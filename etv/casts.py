"""Shared integer-cast metadata and concrete bit-vector semantics."""

from __future__ import annotations

import re
from typing import Any, Tuple

INTEGER_CAST_OPS = frozenset({"sext", "zext", "trunc", "index_cast", "index_castui"})
INT_TO_FLOAT_CAST_OPS = frozenset({"sitofp", "uitofp"})
TTIR_INTEGER_CAST_OPS = {
    "arith.extsi": "sext",
    "arith.extui": "zext",
    "arith.trunci": "trunc",
    "arith.index_cast": "index_cast",
    "arith.index_castui": "index_castui",
}
TTIR_INT_TO_FLOAT_CAST_OPS = {
    "arith.sitofp": "sitofp",
    "arith.uitofp": "uitofp",
}

_INTEGER_TYPE = re.compile(r"^i([1-9][0-9]*)$")


def integer_element_type(type_text: str) -> str:
    """Return the scalar integer/index element type from an MLIR type string."""

    matches = re.findall(r"(?:^|x)(i[1-9][0-9]*|index)(?=[,>]|$)", type_text)
    if len(matches) != 1:
        raise ValueError(
            f"cannot determine one integer element type from {type_text!r}"
        )
    return matches[0]


def normalize_cast_data(data: Any) -> Tuple[str, str]:
    """Validate and normalize the hashable (source type, result type) payload."""

    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("integer cast data must be [source_type, result_type]")
    source, result = data
    if not isinstance(source, str) or not isinstance(result, str):
        raise ValueError("integer cast source and result types must be strings")
    for name in (source, result):
        if name != "index" and _INTEGER_TYPE.fullmatch(name) is None:
            raise ValueError(f"unsupported integer cast type {name!r}")
    return source, result


def integer_width(type_name: str) -> int:
    """Return an explicit integer width; target-dependent index is rejected."""

    if type_name == "index":
        raise ValueError("the target-dependent MLIR index width is not declared")
    match = _INTEGER_TYPE.fullmatch(type_name)
    if match is None:
        raise ValueError(f"unsupported integer type {type_name!r}")
    return int(match.group(1))


def unsigned_wrap(value: int, width: int) -> int:
    return value % (1 << width)


def signed_wrap(value: int, width: int) -> int:
    unsigned = unsigned_wrap(value, width)
    sign = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned >= sign else unsigned


def apply_integer_cast(op: str, value: int | bool, data: Any) -> int | bool:
    """Apply an integer cast without ever treating it as an implicit identity."""

    if op not in INTEGER_CAST_OPS:
        raise ValueError(f"unsupported integer cast operation {op!r}")
    source, result = normalize_cast_data(data)
    source_width = integer_width(source)
    result_width = integer_width(result)
    raw = int(value)

    if op == "sext":
        if result_width <= source_width:
            raise ValueError("sext requires a wider result type")
        casted = signed_wrap(raw, source_width)
    elif op == "zext":
        if result_width <= source_width:
            raise ValueError("zext requires a wider result type")
        casted = unsigned_wrap(raw, source_width)
    elif op == "trunc":
        if result_width >= source_width:
            raise ValueError("trunc requires a narrower result type")
        casted = signed_wrap(raw, result_width)
    else:
        raise ValueError("index casts require an explicit target index width")

    return bool(unsigned_wrap(casted, 1)) if result == "i1" else casted
