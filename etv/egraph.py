"""Pinned egglog adapter for unified equality saturation."""

from __future__ import annotations

import json
import time
from collections import Counter
from importlib.metadata import version
from typing import Iterable, Mapping

from egglog import EGraph as EgglogGraph
from egglog import Expr as EgglogExpr
from egglog import StringLike, eq, rule, ruleset, union, var
from egglog.bindings import EggSmolError

from .ir import Expr
from .model import Limits
from .rules import Pattern, Rule, pattern_variables


class Term(EgglogExpr):
    """Uniform egglog term language for typed ETV expressions."""

    @classmethod
    def node0(cls, op: StringLike, data: StringLike, sort: StringLike) -> "Term": ...

    @classmethod
    def node1(
        cls,
        op: StringLike,
        arg0: "Term",
        data: StringLike,
        sort: StringLike,
    ) -> "Term": ...

    @classmethod
    def node2(
        cls,
        op: StringLike,
        arg0: "Term",
        arg1: "Term",
        data: StringLike,
        sort: StringLike,
    ) -> "Term": ...

    @classmethod
    def node3(
        cls,
        op: StringLike,
        arg0: "Term",
        arg1: "Term",
        arg2: "Term",
        data: StringLike,
        sort: StringLike,
    ) -> "Term": ...


def _data(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _term(op: str, args: tuple[Term, ...], data: object, sort: str) -> Term:
    encoded = _data(data)
    if len(args) == 0:
        return Term.node0(op, encoded, sort)
    if len(args) == 1:
        return Term.node1(op, args[0], encoded, sort)
    if len(args) == 2:
        return Term.node2(op, args[0], args[1], encoded, sort)
    if len(args) == 3:
        return Term.node3(op, args[0], args[1], args[2], encoded, sort)
    raise ValueError(f"egglog adapter does not support arity {len(args)} for {op!r}")


def _expr_term(expression: Expr) -> Term:
    return _term(
        expression.op,
        tuple(_expr_term(argument) for argument in expression.args),
        expression.data,
        expression.sort.value,
    )


def _pattern_term(pattern: Pattern, variables: Mapping[str, Term]) -> Term:
    if pattern.variable is not None:
        return variables[pattern.variable]
    return _term(
        pattern.op or "",
        tuple(_pattern_term(argument, variables) for argument in pattern.args),
        pattern.data,
        pattern.sort.value,
    )


def _egglog_rule(declaration: Rule):
    variables = {
        name: var(name, Term)
        for name in sorted(
            pattern_variables(declaration.lhs) | pattern_variables(declaration.rhs)
        )
    }
    root = var("__etv_root", Term)
    lhs = _pattern_term(declaration.lhs, variables)
    rhs = _pattern_term(declaration.rhs, variables)
    return rule(eq(root).to(lhs), name=declaration.rule_id).then(union(root).with_(rhs))


class EGraph:
    """ETV-facing wrapper around egglog 13.2.0."""

    def __init__(self) -> None:
        self._graph = EgglogGraph()
        self._next_root = 0
        self._next_ruleset = 0
        self.merge_log: list[dict] = []

    @property
    def backend(self) -> dict:
        return {"name": "egglog", "version": version("egglog")}

    def add_expr(self, expression: Expr) -> Term:
        name = f"etv_root_{self._next_root}"
        self._next_root += 1
        return self._graph.let(name, _expr_term(expression))

    def rebuild(self) -> int:
        # egglog rebuilds after each ruleset iteration and after registration.
        return 0

    def union(
        self, lhs: Term, rhs: Term, reason: Mapping[str, object]
    ) -> tuple[Term, bool]:
        already_equivalent = self.equivalent(lhs, rhs)
        self._graph.register(union(lhs).with_(rhs))
        if not already_equivalent:
            entry = {"kind": "explicit_union", **dict(reason)}
            self.merge_log.append(entry)
        return lhs, not already_equivalent

    def equivalent(self, lhs: Term, rhs: Term) -> bool:
        try:
            self._graph.check(eq(lhs).to(rhs))
        except EggSmolError:
            return False
        return True

    def _counts(self) -> tuple[int, int]:
        enodes = sum(size for _, size in self._graph.all_function_sizes())
        try:
            serialized = json.loads(self._graph._serialize().to_json())
            eclasses = sum(
                1
                for item in serialized.get("class_data", {}).values()
                if str(item.get("type", "")).endswith(".Term")
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            eclasses = 0
        return enodes, eclasses

    @property
    def enode_count(self) -> int:
        return self._counts()[0]

    @property
    def eclass_count(self) -> int:
        return self._counts()[1]

    def saturate(self, declarations: Iterable[Rule], limits: Limits) -> dict:
        admitted = tuple(declarations)
        counts: Counter[str] = Counter()
        started = time.monotonic()
        iterations = 0
        stop_reason = "SATURATED"
        iteration_trace: list[dict] = []

        enodes, eclasses = self._counts()
        if enodes > limits.max_enodes:
            return self._stats(
                iterations, enodes, eclasses, counts, "ENODE_LIMIT", iteration_trace
            )
        if not admitted:
            return self._stats(
                iterations,
                enodes,
                eclasses,
                counts,
                "NO_ADMITTED_RULES",
                iteration_trace,
            )

        unified = ruleset(name=f"etv_unified_{self._next_ruleset}")
        self._next_ruleset += 1
        for declaration in admitted:
            unified.register(_egglog_rule(declaration))

        for iteration in range(limits.max_iterations):
            iterations = iteration + 1
            before_enodes, before_eclasses = self._counts()
            report = self._graph.run(1, ruleset=unified)
            iteration_matches: Counter[str] = Counter()
            for rule_decl, match_count in report.num_matches_per_rule.items():
                rule_id = getattr(rule_decl, "name", None)
                if rule_id is not None:
                    counts[rule_id] += match_count
                    iteration_matches[rule_id] += match_count
            enodes, eclasses = self._counts()
            iteration_trace.append(
                {
                    "iteration": iterations,
                    "before": {
                        "enodes": before_enodes,
                        "eclasses": before_eclasses,
                    },
                    "after": {"enodes": enodes, "eclasses": eclasses},
                    "rule_matches": {
                        rule_id: iteration_matches[rule_id]
                        for rule_id in sorted(iteration_matches)
                        if iteration_matches[rule_id]
                    },
                    "updated": bool(report.updated),
                }
            )
            if enodes > limits.max_enodes:
                stop_reason = "ENODE_LIMIT"
                break
            if (time.monotonic() - started) * 1000 > limits.timeout_ms:
                stop_reason = "TIME_LIMIT"
                break
            if not report.updated:
                stop_reason = "SATURATED"
                break
        else:
            stop_reason = "ITERATION_LIMIT"

        for rule_id in sorted(counts):
            if counts[rule_id]:
                self.merge_log.append(
                    {
                        "kind": "egglog_rule_matches",
                        "rule_id": rule_id,
                        "matches": counts[rule_id],
                    }
                )

        return self._stats(
            iterations, enodes, eclasses, counts, stop_reason, iteration_trace
        )

    def _stats(
        self,
        iterations: int,
        enodes: int,
        eclasses: int,
        counts: Mapping[str, int],
        stop_reason: str,
        iteration_trace: list[dict],
    ) -> dict:
        ordered_counts = {
            rule_id: counts[rule_id] for rule_id in sorted(counts) if counts[rule_id]
        }
        return {
            "backend": self.backend,
            "iterations": iterations,
            "enodes": enodes,
            "eclasses": eclasses,
            "rule_matches": ordered_counts,
            "rule_applications": ordered_counts,
            "stop_reason": stop_reason,
            "iteration_trace": iteration_trace,
        }
