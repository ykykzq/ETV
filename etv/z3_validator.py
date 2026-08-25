"""Z3 admission gate for equality rules used by the e-graph."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Tuple

import z3

from .model import PairSpec
from .rules import Pattern, Rule, builtin_rules

TRUSTED_RULE_WARNING = (
    "This predicate-gated rewrite is assumed equivalent without SMT or formal validation; "
    "an incorrect PairSpec predicate or rewrite can make PROVED unsound."
)
UNVERIFIED_ALGEBRAIC_WARNING = (
    "This algebraic rewrite was admitted by PairSpec policy without a successful proof; "
    "an incorrect rewrite can make PROVED unsound."
)


def _term(pattern: Pattern, variables: Dict[str, Any]) -> Any:
    if pattern.variable is not None:
        return variables.setdefault(pattern.variable, z3.Real(pattern.variable))
    if pattern.op == "const_float":
        numerator, denominator = pattern.data
        return z3.RealVal(numerator) / z3.RealVal(denominator)
    args = [_term(arg, variables) for arg in pattern.args]
    if pattern.op == "fadd":
        return args[0] + args[1]
    if pattern.op == "fsub":
        return args[0] - args[1]
    if pattern.op == "fmul":
        return args[0] * args[1]
    if pattern.op == "fdiv":
        return args[0] / args[1]
    if pattern.op == "fneg":
        return -args[0]
    if pattern.op == "fma":
        return args[0] * args[1] + args[2]
    raise ValueError(f"Z3 rule validator does not support {pattern.op!r}")


def validate_rule(rule: Rule, timeout_ms: int = 2_000) -> dict:
    if rule.kind != "algebraic":
        value = rule.to_json()
        value["status"] = "rejected"
        value["validation"] = {
            "result": "wrong_validator",
            "detail": "only algebraic rules may enter the Z3 Real admission gate",
        }
        return value
    variables: Dict[str, Any] = {}
    try:
        lhs = _term(rule.lhs, variables)
        rhs = _term(rule.rhs, variables)
    except ValueError as exc:
        return {
            **rule.to_json(),
            "status": "rejected",
            "validation": {"result": "unsupported", "detail": str(exc)},
        }
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(lhs != rhs)
    formula = solver.sexpr()
    result = solver.check()
    validation: Dict[str, Any] = {
        "solver": "z3",
        "solver_version": z3.get_version_string(),
        "logic": "quantifier-free nonlinear real arithmetic",
        "query": "lhs != rhs",
        "query_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest(),
        "result": str(result),
    }
    if result == z3.sat:
        validation["counterexample"] = {
            declaration.name(): str(result_value)
            for declaration, result_value in (
                (declaration, solver.model()[declaration])
                for declaration in solver.model().decls()
            )
        }
    value = rule.to_json()
    value["status"] = "proved" if result == z3.unsat else "rejected"
    value["validation"] = validation
    return value


@lru_cache(maxsize=1)
def validated_builtin_rules() -> Tuple[Tuple[Rule, ...], Tuple[dict, ...]]:
    accepted = []
    results = []
    for rule in builtin_rules():
        result = validate_rule(rule)
        results.append(result)
        if result["status"] == "proved":
            accepted.append(rule)
    return tuple(accepted), tuple(results)


def admitted_rules(
    spec: PairSpec,
    additional_rules: Iterable[Rule] = (),
) -> Tuple[Tuple[Rule, ...], Tuple[dict, ...]]:
    builtin_accepted, builtin_results = validated_builtin_rules()
    accepted = list(builtin_accepted)
    results = list(builtin_results)
    seen_ids = {rule.rule_id for rule in builtin_accepted}
    for declaration in (*spec.rewrite_rules, *tuple(additional_rules)):
        if declaration.rule_id in seen_ids:
            result = declaration.to_json()
            result["status"] = "rejected"
            result["validation"] = {
                "result": "duplicate_rule_id",
                "detail": "rule id conflicts with an already admitted declaration",
            }
            results.append(result)
            continue
        seen_ids.add(declaration.rule_id)
        checks = [
            {
                "requirement": requirement.to_json(),
                "satisfied": requirement.evaluate(spec.facts),
            }
            for requirement in declaration.predicate_requirements
        ]
        if checks and not all(check["satisfied"] for check in checks):
            result = declaration.to_json()
            result["status"] = "skipped"
            result["validation"] = {
                "result": "predicate_requirements_not_met",
                "predicate_checks": checks,
            }
            results.append(result)
            continue

        if declaration.kind == "algebraic":
            policy = spec.rule_policy.algebraic_validation
            if policy == "trusted":
                result = declaration.to_json()
                result["status"] = "admitted_unverified"
                result["validation"] = {
                    "result": "skipped_by_policy",
                    "policy": policy,
                    "predicate_checks": checks,
                    "warning": UNVERIFIED_ALGEBRAIC_WARNING,
                }
                accepted.append(declaration)
                results.append(result)
                continue

            result = validate_rule(declaration)
            result["validation"]["policy"] = policy
            result["validation"]["predicate_checks"] = checks
            if result["status"] == "proved":
                accepted.append(declaration)
            elif policy == "best_effort":
                result["status"] = "admitted_unverified"
                result["validation"]["warning"] = UNVERIFIED_ALGEBRAIC_WARNING
                accepted.append(declaration)
            results.append(result)
            continue

        result = declaration.to_json()
        if checks and all(check["satisfied"] for check in checks):
            result["status"] = "admitted_unverified"
            result["validation"] = {
                "result": "trusted",
                "policy": spec.rule_policy.non_algebraic_validation,
                "predicate_checks": checks,
                "warning": TRUSTED_RULE_WARNING,
            }
            accepted.append(declaration)
        else:
            result["status"] = "skipped"
            result["validation"] = {
                "result": "predicate_requirements_not_met",
                "predicate_checks": checks,
            }
        results.append(result)
    return tuple(accepted), tuple(results)
