"""Audited algebraic rules admitted by the MVP equality engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .model import ProofLevel, Sort


@dataclass(frozen=True)
class Pattern:
    variable: Optional[str] = None
    op: Optional[str] = None
    args: Tuple["Pattern", ...] = ()
    data: Any = None
    match_data: bool = False
    sort: Sort = Sort.FLOAT


def var(name: str) -> Pattern:
    return Pattern(variable=name)


def node(op: str, *args: Pattern, data: Any = None, match_data: bool = False) -> Pattern:
    return Pattern(op=op, args=tuple(args), data=data, match_data=match_data)


def number(numerator: int, denominator: int = 1) -> Pattern:
    return node(
        "const_float",
        data=(numerator, denominator),
        match_data=True,
    )


@dataclass(frozen=True)
class Rule:
    rule_id: str
    lhs: Pattern
    rhs: Pattern
    evidence: ProofLevel
    validator: str
    statement: str
    requires: Tuple[str, ...] = ("semantic_mode == abstract_float",)

    def to_json(self) -> dict:
        return {
            "id": self.rule_id,
            "statement": self.statement,
            "evidence": self.evidence.value,
            "validator": self.validator,
            "requires": list(self.requires),
            "status": "accepted",
        }


def builtin_rules() -> Tuple[Rule, ...]:
    a, b, c = var("a"), var("b"), var("c")
    schema = "z3_real_unsat"
    return (
        Rule("fadd_comm", node("fadd", a, b), node("fadd", b, a), ProofLevel.ALGEBRAIC, schema, "a + b == b + a"),
        Rule(
            "fadd_assoc",
            node("fadd", node("fadd", a, b), c),
            node("fadd", a, node("fadd", b, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "(a + b) + c == a + (b + c)",
        ),
        Rule("fmul_comm", node("fmul", a, b), node("fmul", b, a), ProofLevel.ALGEBRAIC, schema, "a * b == b * a"),
        Rule(
            "fmul_assoc",
            node("fmul", node("fmul", a, b), c),
            node("fmul", a, node("fmul", b, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "(a * b) * c == a * (b * c)",
        ),
        Rule("fadd_zero", node("fadd", a, number(0)), a, ProofLevel.ALGEBRAIC, schema, "a + 0 == a"),
        Rule("fmul_one", node("fmul", a, number(1)), a, ProofLevel.ALGEBRAIC, schema, "a * 1 == a"),
        Rule("fmul_zero", node("fmul", a, number(0)), number(0), ProofLevel.ALGEBRAIC, schema, "a * 0 == 0"),
        Rule("fdiv_one", node("fdiv", a, number(1)), a, ProofLevel.ALGEBRAIC, schema, "a / 1 == a"),
        Rule(
            "fsub_def",
            node("fsub", a, b),
            node("fadd", a, node("fneg", b)),
            ProofLevel.ALGEBRAIC,
            schema,
            "a - b == a + (-b)",
        ),
        Rule(
            "fneg_involution",
            node("fneg", node("fneg", a)),
            a,
            ProofLevel.ALGEBRAIC,
            schema,
            "-(-a) == a",
        ),
        Rule(
            "fma_def",
            node("fma", a, b, c),
            node("fadd", node("fmul", a, b), c),
            ProofLevel.ALGEBRAIC,
            schema,
            "fma(a,b,c) == a*b+c in ABSTRACT_FLOAT mode",
        ),
    )
