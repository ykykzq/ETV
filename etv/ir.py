"""Backend-neutral semantic IR used after TTIR lifting and before proof search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class Sort(str, Enum):
    INT = "int"
    BOOL = "bool"
    FLOAT = "abstract_float"


@dataclass(frozen=True)
class Expr:
    """A typed, immutable expression in ETV's common semantic IR."""

    op: str
    args: Tuple["Expr", ...] = ()
    data: Any = None
    sort: Sort = Sort.FLOAT

    def to_json(self) -> Any:
        if self.op == "const_int":
            return int(self.data)
        if self.op == "const_bool":
            return bool(self.data)
        if self.op == "const_float":
            numerator, denominator = self.data
            if denominator == 1:
                return {"float": str(numerator)}
            return {"float": f"{numerator}/{denominator}"}
        if self.op == "var":
            return {"var": self.data}
        if self.op == "scalar":
            return {"scalar": self.data}
        if self.op == "load":
            return {
                "op": "load",
                "block": self.data,
                "offset": self.args[0].to_json(),
                "mask": self.args[1].to_json(),
                "default": self.args[2].to_json(),
            }
        result: Dict[str, Any] = {
            "op": self.op,
            "args": [arg.to_json() for arg in self.args],
        }
        if self.data is not None:
            result["data"] = self.data
        return result

    def render(self) -> str:
        if self.op == "const_int":
            return str(self.data)
        if self.op == "const_bool":
            return "true" if self.data else "false"
        if self.op == "const_float":
            numerator, denominator = self.data
            return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
        if self.op in {"var", "scalar", "input"}:
            return f"{self.op}({self.data})"
        if self.op == "read":
            return f"read({self.data}, {self.args[0].render()})"
        if self.op == "load":
            return f"load({self.data}, {', '.join(arg.render() for arg in self.args)})"
        if self.op == "observe_store":
            return (
                f"observe_store({self.data}, "
                f"{', '.join(arg.render() for arg in self.args)})"
            )
        if self.op in {
            "sext",
            "zext",
            "trunc",
            "index_cast",
            "index_castui",
            "sitofp",
            "uitofp",
        }:
            source, result = self.data
            return f"{self.op}[{source}->{result}]({self.args[0].render()})"
        if not self.args:
            return self.op if self.data is None else f"{self.op}({self.data})"
        return f"{self.op}({', '.join(arg.render() for arg in self.args)})"


def int_const(value: int) -> Expr:
    return Expr("const_int", data=int(value), sort=Sort.INT)


def bool_const(value: bool) -> Expr:
    return Expr("const_bool", data=bool(value), sort=Sort.BOOL)


def float_const(value: Fraction) -> Expr:
    return Expr(
        "const_float",
        data=(value.numerator, value.denominator),
        sort=Sort.FLOAT,
    )


@dataclass(frozen=True)
class StoreTemplate:
    """One symbolic store in a lifted program."""

    block: str
    logical_index: Expr
    offset: Expr
    mask: Expr
    value: Expr


@dataclass(frozen=True)
class Program:
    """A program in ETV's proof-oriented semantic IR."""

    name: str
    source: Path
    programs: Expr
    lanes: int
    stores: Tuple[StoreTemplate, ...]
    frontend: str
    frontend_version: Optional[str] = None
