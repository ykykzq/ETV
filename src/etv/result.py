"""Immutable verification facts, independent of presentation."""

from dataclasses import dataclass
from typing import Literal

from .ir import Expr
from .pairspec import Predicate

Status = Literal["PROVED", "DISPROVED", "UNKNOWN"]


@dataclass(frozen=True)
class InputRecord:
    path: str
    sha256: str
    kind: str
    launch_id: str = ""


@dataclass(frozen=True)
class Obligation:
    id: str
    kind: str
    status: str
    dependencies: tuple[str, ...] = ()
    detail: str = ""
    model: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Counterexample:
    index: str
    parameters: tuple[tuple[str, str], ...]
    lhs_offset: str | None
    rhs_offset: str | None
    lhs_mask: bool | None
    rhs_mask: bool | None
    lhs_value: str | None
    rhs_value: str | None
    replay: str


@dataclass(frozen=True)
class RuleFact:
    id: str
    source: str
    sha256: str
    validator: str
    admission: str
    requires: tuple[str, ...] = ()
    matches: int = 0
    used: bool = False
    detail: str = ""
    lhs: Expr | None = None
    rhs: Expr | None = None


@dataclass(frozen=True)
class EqualityFact:
    id: str
    lhs: Expr
    rhs: Expr
    index: str
    merged: bool
    dependencies: tuple[str, ...]
    iterations: int
    enodes: int


@dataclass(frozen=True)
class PartitionFact:
    id: str
    lhs_nodes: tuple[str, ...]
    rhs_nodes: tuple[str, ...]
    dependencies: tuple[str, ...]
    status: str
    detail: str = ""


@dataclass(frozen=True)
class LaunchFact:
    id: str
    side: str
    step: int
    stores: int
    file: str
    function: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationReport:
    run_id: str
    pair_id: str
    status: Status
    reason: str
    inputs: tuple[InputRecord, ...] = ()
    index_bits: int | None = None
    versions: tuple[tuple[str, str], ...] = ()
    predicates: tuple[Predicate, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    rewrites: tuple[RuleFact, ...] = ()
    equalities: tuple[EqualityFact, ...] = ()
    partitions: tuple[PartitionFact, ...] = ()
    launches: tuple[LaunchFact, ...] = ()
    counterexample: Counterexample | None = None
    unsupported: tuple[str, ...] = ()
    trusted_axioms: tuple[str, ...] = ()
    diagnostic: str = ""
