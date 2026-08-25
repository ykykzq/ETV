"""Typed data model shared by the ETV front end and proof pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .rules import Rule


class Status(str, Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"


class ProofLevel(str, Enum):
    BOUNDED_EXHAUSTIVE = "BOUNDED_EXHAUSTIVE"
    STRUCTURAL = "STRUCTURAL"
    ALGEBRAIC = "ALGEBRAIC"
    CONGRUENCE = "CONGRUENCE"
    TRUSTED_AXIOM = "TRUSTED_AXIOM"
    EMPIRICAL = "EMPIRICAL"


class Sort(str, Enum):
    INT = "int"
    BOOL = "bool"
    FLOAT = "abstract_float"


class InputError(ValueError):
    """Raised when an input artifact violates the strict ETV schema."""

    def __init__(self, message: str, code: str = "INVALID_INPUT") -> None:
        super().__init__(message)
        self.code = code


class UnsupportedSemantics(RuntimeError):
    """Raised when a valid artifact uses semantics outside the MVP."""

    def __init__(self, message: str, code: str = "UNSUPPORTED_SEMANTICS") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Expr:
    """A typed, immutable Semantic TTIR expression."""

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
    block: str
    logical_index: Expr
    offset: Expr
    mask: Expr
    value: Expr


@dataclass(frozen=True)
class Program:
    name: str
    source: Path
    programs: Expr
    lanes: int
    stores: Tuple[StoreTemplate, ...]
    frontend: str = "semantic_json"
    frontend_version: Optional[str] = None


@dataclass(frozen=True)
class FrontendSpec:
    """How one side of a PairSpec is parsed and launched."""

    kind: str = "auto"
    function: Optional[str] = None
    programs: Optional[Expr] = None


@dataclass(frozen=True)
class RoleEndpoint:
    kind: str
    name: str
    index: int = 0


@dataclass(frozen=True)
class RolePair:
    logical: str
    lhs: RoleEndpoint
    rhs: RoleEndpoint


@dataclass(frozen=True)
class FactContext:
    bindings: Mapping[str, int]
    assumptions: Tuple[str, ...]
    disjoint_groups: Tuple[frozenset, ...]
    side_bindings: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def disjoint(self, lhs: str, rhs: str) -> bool:
        if lhs == rhs:
            return False
        return any(lhs in group and rhs in group for group in self.disjoint_groups)

    def for_side(self, side: str) -> "FactContext":
        if side not in {"lhs", "rhs"}:
            raise ValueError(f"invalid pair side {side!r}")
        merged = dict(self.bindings)
        merged.update(self.side_bindings.get(side, {}))
        return FactContext(
            bindings=merged,
            assumptions=self.assumptions,
            disjoint_groups=self.disjoint_groups,
        )


@dataclass(frozen=True)
class Limits:
    max_iterations: int = 8
    max_enodes: int = 20_000
    timeout_ms: int = 5_000


@dataclass(frozen=True)
class Contract:
    output_role: str
    output_numel: Expr
    require_full_coverage: bool
    require_disjoint: Tuple[str, ...]


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    source: Path
    lhs_path: Path
    rhs_path: Path
    semantic_mode: str
    roles: Tuple[RolePair, ...]
    facts: FactContext
    contract: Contract
    limits: Limits
    lhs_frontend: FrontendSpec = FrontendSpec()
    rhs_frontend: FrontendSpec = FrontendSpec()
    rewrite_rules: Tuple["Rule", ...] = ()

    def role(self, logical: str) -> RolePair:
        for role in self.roles:
            if role.logical == logical:
                return role
        raise InputError(f"missing role mapping for {logical!r}", "MISSING_ROLE")


@dataclass(frozen=True)
class LaneRecord:
    pid: int
    lane: int
    logical_index: int
    active: bool
    output_role: str
    offset: Optional[int] = None
    value: Optional[Expr] = None


@dataclass(frozen=True)
class Evaluation:
    program: Program
    program_count: int
    records: Tuple[LaneRecord, ...]

    @property
    def active(self) -> Tuple[LaneRecord, ...]:
        return tuple(record for record in self.records if record.active)


def pairwise(values: Sequence[str]) -> Iterable[Tuple[str, str]]:
    for index, lhs in enumerate(values):
        for rhs in values[index + 1 :]:
            yield lhs, rhs
