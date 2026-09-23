"""Bounded official egglog equality saturation in an isolated worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing
from multiprocessing.connection import Connection
import time

import egglog as eg

from .ir import Expr, RootPair
from .pairspec import Limits
from .report import serialize
from .rewrites import RuleRegistry


class Term(eg.Expr):
    def __init__(self, op: eg.String, typ: eg.String, attrs: eg.String) -> None: ...

    @classmethod
    def unary(cls, op: eg.String, typ: eg.String, attrs: eg.String, a: Term) -> Term:  # type: ignore[empty-body]
        ...

    @classmethod
    def binary(cls, op: eg.String, typ: eg.String, attrs: eg.String, a: Term, b: Term) -> Term:  # type: ignore[empty-body]
        ...

    @classmethod
    def ternary(  # type: ignore[empty-body]
        cls, op: eg.String, typ: eg.String, attrs: eg.String, a: Term, b: Term, c: Term
    ) -> Term: ...


@dataclass(frozen=True)
class EqSatResult:
    merged: tuple[bool, ...]
    iterations: int
    enodes: int
    applications: tuple[tuple[str, int], ...]
    reason: str = ""


def _encode(expr: Expr, cache: dict[Expr, Term]) -> Term:
    if expr in cache:
        return cache[expr]
    if expr.op == "pvar":
        result = eg.var(str(expr.attr("name")), Term)
    else:
        prefix = (
            eg.String(expr.op),
            eg.String(str(expr.type)),
            eg.String(json.dumps(serialize(expr.attrs), sort_keys=True)),
        )
        args = tuple(_encode(arg, cache) for arg in expr.args)
        if not args:
            result = Term(*prefix)
        elif len(args) == 1:
            result = Term.unary(*prefix, args[0])
        elif len(args) == 2:
            result = Term.binary(*prefix, args[0], args[1])
        elif len(args) == 3:
            result = Term.ternary(*prefix, args[0], args[1], args[2])
        else:
            raise ValueError("unsupported term arity")
    cache[expr] = result
    return result


def _run(roots: tuple[RootPair, ...], registry: RuleRegistry, limits: Limits) -> EqSatResult:
    graph = eg.EGraph()
    cache: dict[Expr, Term] = {}
    goals = [(_encode(root.lhs, cache), _encode(root.rhs, cache)) for root in roots]
    for i, (lhs, rhs) in enumerate(goals):
        graph.let(f"lhs_{i}", lhs)
        graph.let(f"rhs_{i}", rhs)
    encoded_rules = [
        (r.id, eg.rewrite(_encode(r.lhs, {})).to(_encode(r.rhs, {}))) for r in registry.rules
    ]
    rule_set = (
        eg.ruleset(*(r for _, r in encoded_rules), name="etv_run")
        if encoded_rules
        else eg.ruleset(name="etv_run")
    )
    applications = {r.id: 0 for r in registry.rules}
    iterations = 0
    start = time.monotonic()
    reason = ""
    while True:
        enodes = sum(n for _, n in graph.all_function_sizes())
        merged = tuple(graph.check_bool(eg.eq(lhs).to(rhs)) for lhs, rhs in goals)
        if enodes > limits.max_enodes:
            reason = "RESOURCE_LIMIT"
            merged = tuple(False for _ in goals)
            break
        if all(merged):
            break
        if iterations >= limits.max_iterations:
            reason = "RESOURCE_LIMIT"
            break
        if (time.monotonic() - start) * 1000 >= limits.egraph_timeout_ms:
            reason = "RESOURCE_LIMIT"
            break
        report = graph.run(1, ruleset=rule_set)
        iterations += 1
        for name, rule in encoded_rules:
            applications[name] += sum(
                count
                for declaration, count in report.num_matches_per_rule.items()
                if declaration == rule.decl
            )
        if not report.updated:
            merged = tuple(graph.check_bool(eg.eq(lhs).to(rhs)) for lhs, rhs in goals)
            enodes = sum(n for _, n in graph.all_function_sizes())
            break
    return EqSatResult(merged, iterations, enodes, tuple(sorted(applications.items())), reason)


def _worker(
    connection: Connection, roots: tuple[RootPair, ...], registry: RuleRegistry, limits: Limits
) -> None:
    try:
        connection.send(_run(roots, registry, limits))
    except Exception as exc:
        connection.send(
            EqSatResult(
                tuple(False for _ in roots),
                0,
                0,
                (),
                "INTERNAL_ERROR: " + type(exc).__name__ + ": " + str(exc),
            )
        )
    finally:
        connection.close()


def saturate_roots(roots: tuple[RootPair, ...], rules: RuleRegistry, limits: Limits) -> EqSatResult:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, roots, rules, limits))
    process.start()
    send.close()
    try:
        if receive.poll(limits.egraph_timeout_ms / 1000):
            try:
                return receive.recv()
            except EOFError:
                return EqSatResult(tuple(False for _ in roots), 0, 0, (), "INTERNAL_ERROR")
        return EqSatResult(tuple(False for _ in roots), 0, 0, (), "RESOURCE_LIMIT")
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receive.close()


def saturate(lhs: Expr, rhs: Expr, rules: RuleRegistry, limits: Limits) -> EqSatResult:
    from .ir import const

    return saturate_roots((RootPair(lhs, rhs, const(0)),), rules, limits)
