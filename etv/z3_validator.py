"""Z3 admission gate for equality rules used by the e-graph."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Tuple

import z3

from .casts import INTEGER_CAST_OPS, integer_width, normalize_cast_data
from .ir import Sort
from .model import PairSpec
from .observability import get_logger, log_event
from .rules import Pattern, Rule, builtin_rules


LOGGER = get_logger(__name__)

TRUSTED_RULE_WARNING = (
    "This predicate-gated rewrite is assumed equivalent without SMT or formal validation; "
    "an incorrect PairSpec predicate or rewrite can make PROVED unsound."
)
UNVERIFIED_ALGEBRAIC_WARNING = (
    "This algebraic rewrite was admitted by PairSpec policy without a successful proof; "
    "an incorrect rewrite can make PROVED unsound."
)


def _pattern_domain(pattern: Pattern, fallback: str = "real") -> str:
    if pattern.variable is not None:
        return fallback
    if pattern.op in INTEGER_CAST_OPS:
        _, result = normalize_cast_data(pattern.data)
        return "bool" if result == "i1" else "int"
    if pattern.op == "const_int" or pattern.sort == Sort.INT:
        return "int"
    if pattern.op == "const_bool" or pattern.sort == Sort.BOOL:
        return "bool"
    return "real"


def _unsigned_wrap(value: Any, width: int) -> Any:
    return value % (1 << width)


def _signed_wrap(value: Any, width: int) -> Any:
    unsigned = _unsigned_wrap(value, width)
    return z3.If(unsigned >= (1 << (width - 1)), unsigned - (1 << width), unsigned)


def _term(
    pattern: Pattern,
    variables: Dict[str, Any],
    variable_domains: Dict[str, str],
    expected_domain: str = "real",
) -> Any:
    if pattern.variable is not None:
        previous = variable_domains.get(pattern.variable)
        if previous is not None and previous != expected_domain:
            raise ValueError(
                f"pattern variable {pattern.variable!r} is used as both {previous} and "
                f"{expected_domain}"
            )
        variable_domains[pattern.variable] = expected_domain
        if pattern.variable not in variables:
            constructor = {
                "real": z3.Real,
                "int": z3.Int,
                "bool": z3.Bool,
            }[expected_domain]
            variables[pattern.variable] = constructor(pattern.variable)
        return variables[pattern.variable]
    if pattern.op == "const_float":
        numerator, denominator = pattern.data
        return z3.RealVal(numerator) / z3.RealVal(denominator)
    if pattern.op == "const_int":
        return z3.IntVal(int(pattern.data))
    if pattern.op == "const_bool":
        return z3.BoolVal(bool(pattern.data))
    if pattern.op in INTEGER_CAST_OPS:
        source_type, result_type = normalize_cast_data(pattern.data)
        source_width = integer_width(source_type)
        result_width = integer_width(result_type)
        source_domain = "bool" if source_type == "i1" else "int"
        raw = _term(pattern.args[0], variables, variable_domains, source_domain)
        raw_int = z3.If(raw, 1, 0) if source_domain == "bool" else raw
        if pattern.op == "sext" and result_width > source_width:
            result = _signed_wrap(raw_int, source_width)
        elif pattern.op == "zext" and result_width > source_width:
            result = _unsigned_wrap(raw_int, source_width)
        elif pattern.op == "trunc" and result_width < source_width:
            result = _signed_wrap(raw_int, result_width)
        else:
            raise ValueError(
                f"invalid or target-dependent integer cast {pattern.op}{pattern.data!r}"
            )
        return result != 0 if result_type == "i1" else result
    args = [_term(arg, variables, variable_domains, "real") for arg in pattern.args]
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
    log_event(
        LOGGER,
        logging.DEBUG,
        "rewrite_validation_started",
        "validating algebraic rewrite with Z3",
        rule_id=rule.rule_id,
        timeout_ms=timeout_ms,
    )
    if rule.kind != "algebraic":
        value = rule.to_json()
        value["status"] = "rejected"
        value["validation"] = {
            "result": "wrong_validator",
            "detail": "only algebraic rules may enter the Z3 Real admission gate",
        }
        return value
    variables: Dict[str, Any] = {}
    variable_domains: Dict[str, str] = {}
    try:
        domain = _pattern_domain(rule.lhs)
        rhs_domain = _pattern_domain(rule.rhs, domain)
        if rhs_domain != domain:
            raise ValueError(
                f"rule result domains differ: lhs is {domain}, rhs is {rhs_domain}"
            )
        lhs = _term(rule.lhs, variables, variable_domains, domain)
        rhs = _term(rule.rhs, variables, variable_domains, domain)
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
        "logic": (
            "quantifier-free integer arithmetic with exact finite-width cast formulas"
            if domain in {"int", "bool"}
            else "quantifier-free nonlinear real arithmetic"
        ),
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
    log_event(
        LOGGER,
        logging.INFO if value["status"] == "proved" else logging.WARNING,
        "rewrite_validation_finished",
        "completed Z3 rewrite validation",
        rule_id=rule.rule_id,
        status=value["status"],
        result=validation.get("result"),
    )
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
    additional_rules = tuple(additional_rules)
    log_event(
        LOGGER,
        logging.DEBUG,
        "rewrite_admission_started",
        "admitting built-in and PairSpec rewrite rules",
        additional_rules=len(additional_rules),
        pair_rules=len(spec.rewrite_rules),
    )
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
    log_event(
        LOGGER,
        logging.INFO,
        "rewrite_admission_finished",
        "rewrite admission completed",
        accepted=len(accepted),
        records=len(results),
        unverified=sum(
            1 for result in results if result.get("status") == "admitted_unverified"
        ),
    )
    return tuple(accepted), tuple(results)
