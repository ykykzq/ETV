"""Rewrite declarations and fact gates for the egglog equality engine."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional, Tuple

from .model import Expr, FactContext, ProofLevel, Sort


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


def node(
    op: str,
    *args: Pattern,
    data: Any = None,
    match_data: bool = False,
    sort: Sort = Sort.FLOAT,
) -> Pattern:
    return Pattern(op=op, args=tuple(args), data=data, match_data=match_data, sort=sort)


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
    kind: str = "algebraic"
    source: str = "builtin"
    fact_requirements: Tuple["FactRequirement", ...] = ()
    generated_by: str = "human"
    generator: Optional[str] = None
    prompt_sha256: Optional[str] = None

    def to_json(self) -> dict:
        value = {
            "id": self.rule_id,
            "statement": self.statement,
            "evidence": self.evidence.value,
            "validator": self.validator,
            "kind": self.kind,
            "source": self.source,
            "requires": list(self.requires),
            "fact_requirements": [
                requirement.to_json() for requirement in self.fact_requirements
            ],
            "provenance": {"generated_by": self.generated_by},
            "status": "declared",
        }
        if self.generator is not None:
            value["provenance"]["generator"] = self.generator
        if self.prompt_sha256 is not None:
            value["provenance"]["prompt_sha256"] = self.prompt_sha256
        return value


@dataclass(frozen=True)
class FactRequirement:
    kind: str
    name: Optional[str] = None
    value: Any = None
    roles: Tuple[str, ...] = ()
    expression: Optional[Expr] = None

    def evaluate(self, facts: FactContext) -> bool:
        if self.kind == "binding_equals":
            return (
                self.name in facts.bindings and facts.bindings[self.name] == self.value
            )
        if self.kind == "assumption":
            return self.value in facts.assumptions
        if self.kind == "disjoint":
            return all(
                facts.disjoint(lhs, rhs) for lhs, rhs in combinations(self.roles, 2)
            )
        if self.kind == "constraint":
            return self.expression in facts.constraints
        return False

    def to_json(self) -> dict:
        value = {"kind": self.kind}
        if self.name is not None:
            value["name"] = self.name
        if self.kind == "assumption":
            value["text"] = self.value
        elif self.value is not None:
            value["value"] = self.value
        if self.roles:
            value["roles"] = list(self.roles)
        if self.expression is not None:
            value["expression"] = self.expression.to_json()
        return value


def pattern_variables(pattern: Pattern) -> frozenset[str]:
    if pattern.variable is not None:
        return frozenset((pattern.variable,))
    variables = set()
    for argument in pattern.args:
        variables.update(pattern_variables(argument))
    return frozenset(variables)


def render_pattern(pattern: Pattern) -> str:
    if pattern.variable is not None:
        return f"${pattern.variable}"
    if pattern.op == "const_float":
        numerator, denominator = pattern.data
        return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
    suffix = "" if pattern.data is None else f"[{pattern.data!r}]"
    if not pattern.args:
        return f"{pattern.op}{suffix}"
    return f"{pattern.op}{suffix}({', '.join(render_pattern(arg) for arg in pattern.args)})"


def pattern_from_expr(expression: Expr) -> Pattern:
    """Build an exact ground pattern for an internal expression."""

    return node(
        expression.op,
        *(pattern_from_expr(argument) for argument in expression.args),
        data=expression.data,
        match_data=True,
        sort=expression.sort,
    )


def builtin_rules() -> Tuple[Rule, ...]:
    a, b, c = var("a"), var("b"), var("c")
    schema = "z3_real_unsat"
    return (
        Rule(
            "fadd_comm",
            node("fadd", a, b),
            node("fadd", b, a),
            ProofLevel.ALGEBRAIC,
            schema,
            "a + b == b + a",
        ),
        Rule(
            "fadd_assoc",
            node("fadd", node("fadd", a, b), c),
            node("fadd", a, node("fadd", b, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "(a + b) + c == a + (b + c)",
        ),
        Rule(
            "fmul_comm",
            node("fmul", a, b),
            node("fmul", b, a),
            ProofLevel.ALGEBRAIC,
            schema,
            "a * b == b * a",
        ),
        Rule(
            "fmul_assoc",
            node("fmul", node("fmul", a, b), c),
            node("fmul", a, node("fmul", b, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "(a * b) * c == a * (b * c)",
        ),
        Rule(
            "fadd_zero",
            node("fadd", a, number(0)),
            a,
            ProofLevel.ALGEBRAIC,
            schema,
            "a + 0 == a",
        ),
        Rule(
            "fmul_one",
            node("fmul", a, number(1)),
            a,
            ProofLevel.ALGEBRAIC,
            schema,
            "a * 1 == a",
        ),
        Rule(
            "fmul_zero",
            node("fmul", a, number(0)),
            number(0),
            ProofLevel.ALGEBRAIC,
            schema,
            "a * 0 == 0",
        ),
        Rule(
            "fdiv_one",
            node("fdiv", a, number(1)),
            a,
            ProofLevel.ALGEBRAIC,
            schema,
            "a / 1 == a",
        ),
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
            "fsub_zero",
            node("fsub", a, number(0)),
            a,
            ProofLevel.ALGEBRAIC,
            schema,
            "a - 0 == a",
        ),
        Rule(
            "fsub_self",
            node("fsub", a, a),
            number(0),
            ProofLevel.ALGEBRAIC,
            schema,
            "a - a == 0",
        ),
        Rule(
            "fadd_inverse",
            node("fadd", a, node("fneg", a)),
            number(0),
            ProofLevel.ALGEBRAIC,
            schema,
            "a + (-a) == 0",
        ),
        Rule(
            "fneg_zero",
            node("fneg", number(0)),
            number(0),
            ProofLevel.ALGEBRAIC,
            schema,
            "-0 == 0",
        ),
        Rule(
            "fmul_add_distrib",
            node("fmul", a, node("fadd", b, c)),
            node("fadd", node("fmul", a, b), node("fmul", a, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "a * (b + c) == a*b + a*c",
        ),
        Rule(
            "fmul_sub_distrib",
            node("fmul", a, node("fsub", b, c)),
            node("fsub", node("fmul", a, b), node("fmul", a, c)),
            ProofLevel.ALGEBRAIC,
            schema,
            "a * (b - c) == a*b - a*c",
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
