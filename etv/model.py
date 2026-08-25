"""Typed data model shared by the ETV front end and proof pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# Compatibility re-exports; proof modules import these definitions from etv.ir.
from .ir import (
    Expr,
    Program,
    Sort,
    StoreTemplate,
    bool_const,
    float_const,
    int_const,
)

if TYPE_CHECKING:
    from .rules import Rule


class Status(str, Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"


class ProofLevel(str, Enum):
    BOUNDED_EXHAUSTIVE = "BOUNDED_EXHAUSTIVE"
    PARAMETRIC_SMT = "PARAMETRIC_SMT"
    STRUCTURAL = "STRUCTURAL"
    ALGEBRAIC = "ALGEBRAIC"
    CONGRUENCE = "CONGRUENCE"
    DECOMPOSITION = "DECOMPOSITION"
    TRUSTED_AXIOM = "TRUSTED_AXIOM"
    EMPIRICAL = "EMPIRICAL"


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
class FrontendSpec:
    """How one side of a PairSpec is parsed and launched."""

    kind: str = "ttir"
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
class Parameter:
    """A universally quantified signed-i32 parameter and its declared domain."""

    minimum: int = -(2**31)
    maximum: int = 2**31 - 1


@dataclass(frozen=True)
class FactContext:
    bindings: Mapping[str, int | Expr]
    assumptions: Tuple[str, ...]
    disjoint_groups: Tuple[frozenset, ...]
    side_bindings: Mapping[str, Mapping[str, int | Expr]] = field(default_factory=dict)
    parameters: Mapping[str, Parameter] = field(default_factory=dict)
    constraints: Tuple[Expr, ...] = ()

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
            parameters=self.parameters,
            constraints=self.constraints,
        )


@dataclass(frozen=True)
class Limits:
    max_iterations: int = 8
    max_enodes: int = 20_000
    timeout_ms: int = 5_000


@dataclass(frozen=True)
class RulePolicy:
    """Controls proof admission independently from later rule application."""

    algebraic_validation: str = "required"
    non_algebraic_validation: str = "trusted"


@dataclass(frozen=True)
class LLMConfig:
    """Optional, auditable LLM assistance. Credentials come from the environment."""

    enabled: bool = False
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    select_nodes: bool = True
    generate_rules: bool = True
    max_candidates: int = 8
    timeout_ms: int = 30_000


@dataclass(frozen=True)
class PartitionConfig:
    """Controls LLM-proposed, machine-checked paired subgraph decomposition."""

    enabled: bool = False
    min_partitions: int = 2
    max_partitions: int = 16


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
    rule_policy: RulePolicy = RulePolicy()
    llm: LLMConfig = LLMConfig()
    partition: PartitionConfig = PartitionConfig()

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
