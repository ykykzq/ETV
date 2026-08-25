"""End-to-end symbolic translation validation for ETV program pairs."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .egraph import EGraph
from .evaluator import SideRoles, eval_expr, evaluate_program
from .llm import LLMError, propose_rules
from .model import (
    Evaluation,
    Expr,
    InputError,
    PairSpec,
    PartitionConfig,
    ProofLevel,
    Program,
    Sort,
    Status,
    UnsupportedSemantics,
    pairwise,
)
from .partition import PartitionError, PartitionPlan, propose_partition_plan
from .schema import load_pair_spec
from .parametric import ParametricFailure, verify_parametric_pair
from .ttir import load_program_artifact
from .z3_validator import admitted_rules


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block(
    kind: str,
    status: Status,
    summary: str,
    levels: Sequence[ProofLevel] = (),
    reason: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    required_for_final: bool = True,
) -> dict:
    value = {
        "kind": kind,
        "status": status.value,
        "summary": summary,
        "proof_levels": [level.value for level in levels],
        "required_for_final": required_for_final,
    }
    if reason is not None:
        value["reason"] = reason
    if details:
        value["details"] = dict(details)
    return value


def _base_report(spec: PairSpec) -> dict:
    def binding_text(value: int | Expr) -> str:
        return value.render() if isinstance(value, Expr) else str(value)

    assumptions = [
        {
            "text": f"{name} = {binding_text(value)}",
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.bindings",
        }
        for name, value in sorted(spec.facts.bindings.items())
    ]
    assumptions.extend(
        {
            "text": f"{side}.{name} = {binding_text(value)}",
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": f"PairSpec.facts.side_bindings.{side}",
        }
        for side in ("lhs", "rhs")
        for name, value in sorted(spec.facts.side_bindings.get(side, {}).items())
    )
    assumptions.extend(
        {
            "text": f"{name} in [{parameter.minimum}, {parameter.maximum}]",
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.parameters",
        }
        for name, parameter in sorted(spec.facts.parameters.items())
    )
    assumptions.extend(
        {
            "text": constraint.render(),
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.constraints",
        }
        for constraint in spec.facts.constraints
    )
    assumptions.extend(
        {
            "text": text,
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.assumptions",
        }
        for text in spec.facts.assumptions
    )
    assumptions.extend(
        {
            "text": "disjoint(" + ", ".join(sorted(group)) + ")",
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.disjoint",
        }
        for group in spec.facts.disjoint_groups
    )
    return {
        "schema_version": 4,
        "tool": {"name": "ETV", "version": "0.5.0"},
        "pair_id": spec.pair_id,
        "status": Status.UNKNOWN.value,
        "reason": "NOT_RUN",
        "semantic_mode": spec.semantic_mode,
        "scope": (
            "fixed-rank symbolic-shape single-store parametric translation validation"
            if spec.facts.parameters
            else "fixed-specialization single-store bounded translation validation"
        ),
        "inputs": {
            "spec": str(spec.source),
            "lhs": str(spec.lhs_path),
            "rhs": str(spec.rhs_path),
            "sha256": {
                "spec": _hash(spec.source),
                "lhs": _hash(spec.lhs_path),
                "rhs": _hash(spec.rhs_path),
            },
        },
        "assumptions": assumptions,
        "trusted_axioms": [
            "PairSpec role correspondence",
            (
                "PairSpec symbolic parameter domains and relational shape constraints"
                if spec.facts.parameters
                else "PairSpec fixed shape/stride/launch bindings"
            ),
            "PairSpec no-alias declarations",
            "ABSTRACT_FLOAT interprets floating operations over exact mathematical values",
            "Semantic TTIR evaluator and memory-token model",
        ],
        "blocks": [],
        "unsupported": [],
        "counterexample": None,
        "proof": {},
        "guarantees": {
            "proves": (
                "equal final Output memory for all abstract input values and all declared shape parameters"
                if spec.facts.parameters
                else "equal final Output memory for all abstract input values in the enumerated launch domain"
            ),
            "does_not_prove": [
                "IEEE-754 or bitwise GPU equality",
                "formal correctness of the raw-TTIR-to-Semantic-TTIR lifting implementation",
                "formal correctness of the Torch-Prims-to-ETV lifting implementation",
                (
                    "dynamic rank, unbounded loops, or shapes outside the declared parameter domain"
                    if spec.facts.parameters
                    else "parametric shapes or unbounded loops"
                ),
                "concurrent/shared-memory semantics",
                "allocated-buffer bounds or GPU memory safety",
            ],
        },
    }


def _finish(
    report: dict, status: Status, reason: str, counterexample: Optional[dict] = None
) -> dict:
    report["status"] = status.value
    report["reason"] = reason
    report["counterexample"] = counterexample
    if status == Status.PROVED:
        report["guarantees"]["establishes"] = report["guarantees"].pop("proves")
    elif status == Status.DISPROVED:
        report["guarantees"].pop("proves", None)
        report["guarantees"][
            "establishes"
        ] = "the reported concrete witness violates observable equivalence in ABSTRACT_FLOAT semantics"
    else:
        report["guarantees"].pop("proves", None)
        report["guarantees"][
            "establishes"
        ] = "no equivalence or inequivalence conclusion; the reason identifies the first unmet obligation"
    return report


def _invalid_report(path: Path, exc: Exception) -> dict:
    return {
        "schema_version": 4,
        "tool": {"name": "ETV", "version": "0.5.0"},
        "pair_id": path.stem,
        "status": Status.UNKNOWN.value,
        "reason": getattr(exc, "code", "INVALID_INPUT"),
        "semantic_mode": None,
        "scope": "input validation",
        "inputs": {"spec": str(path.resolve())},
        "assumptions": [],
        "trusted_axioms": [],
        "blocks": [],
        "unsupported": [str(exc)],
        "counterexample": None,
        "proof": {},
        "guarantees": {},
    }


def _active_map(evaluation: Evaluation) -> Tuple[Dict[int, Any], Optional[dict]]:
    result: Dict[int, Any] = {}
    for record in evaluation.active:
        previous = result.get(record.logical_index)
        if previous is not None:
            return {}, {
                "kind": "DUPLICATE_LOGICAL_WRITE",
                "logical_index": record.logical_index,
                "first": {
                    "pid": previous.pid,
                    "lane": previous.lane,
                    "offset": previous.offset,
                },
                "second": {
                    "pid": record.pid,
                    "lane": record.lane,
                    "offset": record.offset,
                },
            }
        result[record.logical_index] = record
    return result, None


def _race(evaluation: Evaluation) -> Optional[dict]:
    addresses: Dict[Tuple[str, int], Any] = {}
    for record in evaluation.active:
        key = (record.output_role, int(record.offset))
        previous = addresses.get(key)
        if previous is not None:
            return {
                "kind": "POTENTIAL_WRITE_RACE",
                "role": record.output_role,
                "offset": record.offset,
                "first": {"pid": previous.pid, "lane": previous.lane},
                "second": {"pid": record.pid, "lane": record.lane},
            }
        addresses[key] = record
    return None


def _leaf_signature(expr: Expr) -> Tuple[str, ...]:
    leaves: List[str] = []

    def visit(current: Expr) -> None:
        if current.op == "read":
            leaves.append(f"read({current.data},{current.args[0].render()})")
            return
        if current.op == "input":
            leaves.append(f"input({current.data})")
            return
        for arg in current.args:
            visit(arg)

    visit(expr)
    return tuple(sorted(leaves))


def _leaf_key(expr: Expr) -> Optional[str]:
    if expr.op == "read":
        return f"read({expr.data},{expr.args[0].render()})"
    if expr.op == "input":
        return f"input({expr.data})"
    return None


def _collect_leaves(expr: Expr, result: Set[str]) -> None:
    key = _leaf_key(expr)
    if key is not None:
        result.add(key)
        return
    for arg in expr.args:
        _collect_leaves(arg, result)


def _constant_abstract(expr: Expr) -> Optional[Fraction]:
    """Evaluate the exact-rational subset used by structural domain checks."""
    if expr.op == "const_float":
        return Fraction(expr.data[0], expr.data[1])
    values = [_constant_abstract(arg) for arg in expr.args]
    if any(value is None for value in values):
        return None
    concrete = [value for value in values if value is not None]
    if expr.op == "fadd":
        return concrete[0] + concrete[1]
    if expr.op == "fsub":
        return concrete[0] - concrete[1]
    if expr.op == "fmul":
        return concrete[0] * concrete[1]
    if expr.op == "fdiv" and concrete[1] != 0:
        return concrete[0] / concrete[1]
    if expr.op == "fneg":
        return -concrete[0]
    if expr.op == "fma":
        return concrete[0] * concrete[1] + concrete[2]
    return None


def _definedness_issue(expr: Expr) -> Optional[dict]:
    """Return the first floating-domain obligation not proved structurally."""
    for arg in expr.args:
        issue = _definedness_issue(arg)
        if issue is not None:
            return issue
    if expr.op == "fdiv":
        denominator = _constant_abstract(expr.args[1])
        if denominator is None or denominator == 0:
            return {
                "operation": "fdiv",
                "expression": expr.render(),
                "required": "denominator is a provably nonzero exact constant",
            }
    if expr.op in {"fsqrt", "frsqrt"}:
        radicand = _constant_abstract(expr.args[0])
        valid = radicand is not None and (
            radicand >= 0 if expr.op == "fsqrt" else radicand > 0
        )
        if not valid:
            return {
                "operation": expr.op,
                "expression": expr.render(),
                "required": (
                    "radicand is a provably nonnegative exact constant"
                    if expr.op == "fsqrt"
                    else "radicand is a provably positive exact constant"
                ),
            }
    return None


def _eval_abstract(expr: Expr, assignment: Mapping[str, Fraction]) -> Fraction:
    key = _leaf_key(expr)
    if key is not None:
        return assignment[key]
    if expr.op == "const_float":
        return Fraction(expr.data[0], expr.data[1])
    values = [_eval_abstract(arg, assignment) for arg in expr.args]
    if expr.op == "fadd":
        return values[0] + values[1]
    if expr.op == "fsub":
        return values[0] - values[1]
    if expr.op == "fmul":
        return values[0] * values[1]
    if expr.op == "fdiv":
        if values[1] == 0:
            raise ZeroDivisionError
        return values[0] / values[1]
    if expr.op == "fneg":
        return -values[0]
    if expr.op == "fma":
        return values[0] * values[1] + values[2]
    raise UnsupportedSemantics(
        f"concrete counterexample evaluator does not support {expr.op}",
        "COUNTEREXAMPLE_SEARCH_UNSUPPORTED",
    )


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _candidate_assignments(leaves: Sequence[str]) -> Iterable[Dict[str, Fraction]]:
    zero = {leaf: Fraction(0) for leaf in leaves}
    yield zero
    for leaf in leaves:
        for value in (Fraction(1), Fraction(-1), Fraction(2)):
            assignment = dict(zero)
            assignment[leaf] = value
            yield assignment
    yield {leaf: Fraction(1) for leaf in leaves}
    yield {leaf: Fraction(-1 if index % 2 else 2) for index, leaf in enumerate(leaves)}
    if len(leaves) <= 4:
        for values in itertools.product(
            (Fraction(-1), Fraction(0), Fraction(1)), repeat=len(leaves)
        ):
            yield dict(zip(leaves, values))


def _counterexample(lhs: Expr, rhs: Expr, logical_index: Any) -> Optional[dict]:
    leaves: Set[str] = set()
    _collect_leaves(lhs, leaves)
    _collect_leaves(rhs, leaves)
    ordered = sorted(leaves)
    try:
        for assignment in _candidate_assignments(ordered):
            try:
                lhs_value = _eval_abstract(lhs, assignment)
                rhs_value = _eval_abstract(rhs, assignment)
            except ZeroDivisionError:
                continue
            if lhs_value != rhs_value:
                return {
                    "kind": "ABSTRACT_VALUE_MODEL",
                    "logical_index": logical_index,
                    "assignment": {
                        key: _fraction_text(value) for key, value in assignment.items()
                    },
                    "lhs_value": _fraction_text(lhs_value),
                    "rhs_value": _fraction_text(rhs_value),
                    "lhs_expression": lhs.render(),
                    "rhs_expression": rhs.render(),
                }
    except UnsupportedSemantics:
        return None
    return None


def _expected_numel(spec: PairSpec) -> int:
    value = eval_expr(spec.contract.output_numel, {}, spec.facts, SideRoles({}, {}))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError("contract.output_numel must evaluate to a positive integer")
    return value


def _combine_saturation(first: dict, second: dict) -> dict:
    matches: Dict[str, int] = {}
    for stats in (first, second):
        for rule_id, count in stats.get("rule_matches", {}).items():
            matches[rule_id] = matches.get(rule_id, 0) + int(count)
    return {
        "backend": second.get("backend", first.get("backend", {})),
        "iterations": int(first.get("iterations", 0))
        + int(second.get("iterations", 0)),
        "enodes": second.get("enodes", first.get("enodes", 0)),
        "eclasses": second.get("eclasses", first.get("eclasses", 0)),
        "rule_matches": {key: matches[key] for key in sorted(matches) if matches[key]},
        "rule_applications": {
            key: matches[key] for key in sorted(matches) if matches[key]
        },
        "stop_reason": second.get("stop_reason", first.get("stop_reason")),
        "phase": "MULTI_PHASE",
        "iteration_trace": [
            *first.get("iteration_trace", []),
            *second.get("iteration_trace", []),
        ],
        "phases": [first, second],
    }


def _label_saturation(stats: dict, phase: str) -> dict:
    stats["phase"] = phase
    for item in stats.get("iteration_trace", []):
        item["phase"] = phase
    return stats


def _complete_whole_compute_proof(
    report: dict,
    spec: PairSpec,
    root_expressions: Sequence[Tuple[Any, Expr, Expr]],
    store_levels: Sequence[ProofLevel],
    relational_rules: Sequence[Any] = (),
    relational_admission: Sequence[Mapping[str, Any]] = (),
) -> dict:
    definedness_issues = []
    for logical_index, lhs_expression, rhs_expression in root_expressions:
        for side, expression in (("lhs", lhs_expression), ("rhs", rhs_expression)):
            issue = _definedness_issue(expression)
            if issue is not None:
                definedness_issues.append(
                    {"logical_index": logical_index, "side": side, **issue}
                )
                if len(definedness_issues) == 4:
                    break
        if len(definedness_issues) == 4:
            break
    if definedness_issues:
        report["blocks"].append(
            _block(
                "DEFINEDNESS",
                Status.UNKNOWN,
                "a partial floating operation lacks a machine-checked domain fact",
                reason="FLOAT_DEFINEDNESS_NOT_PROVED",
                details={"examples": definedness_issues},
            )
        )
        return _finish(report, Status.UNKNOWN, "FLOAT_DEFINEDNESS_NOT_PROVED")
    report["blocks"].append(
        _block(
            "DEFINEDNESS",
            Status.PROVED,
            "all partial floating operations have structurally proved constant domains",
            (ProofLevel.STRUCTURAL,),
        )
    )

    ordinary_rules, ordinary_admission = admitted_rules(spec)
    accepted_rules = tuple(relational_rules) + tuple(ordinary_rules)
    rule_admission = tuple(relational_admission) + tuple(ordinary_admission)
    egraph = EGraph()
    root_pairs = [
        (
            label,
            lhs_expr,
            rhs_expr,
            egraph.add_expr(lhs_expr),
            egraph.add_expr(rhs_expr),
        )
        for label, lhs_expr, rhs_expr in root_expressions
    ]
    egraph.rebuild()
    initially_unmatched = [
        item for item in root_pairs if not egraph.equivalent(item[3], item[4])
    ]
    initial_state = {
        "enodes": egraph.enode_count,
        "eclasses": egraph.eclass_count,
        "unmatched_root_pairs": len(initially_unmatched),
        "roots": [
            {"label": str(label), "lhs": lhs.render(), "rhs": rhs.render()}
            for label, lhs, rhs, _, _ in root_pairs
        ],
    }
    unmatched_after_facts = initially_unmatched
    if initially_unmatched and relational_rules:
        saturation = _label_saturation(
            egraph.saturate(relational_rules, spec.limits),
            "FACT_DERIVED_RELATIONAL_REWRITES",
        )
        unmatched_after_facts = [
            item for item in root_pairs if not egraph.equivalent(item[3], item[4])
        ]
        if unmatched_after_facts and ordinary_rules:
            algebraic_saturation = _label_saturation(
                egraph.saturate(ordinary_rules, spec.limits),
                "ALGEBRAIC_REWRITES",
            )
            saturation = _combine_saturation(saturation, algebraic_saturation)
    elif initially_unmatched:
        saturation = _label_saturation(
            egraph.saturate(ordinary_rules, spec.limits), "ALGEBRAIC_REWRITES"
        )
    else:
        saturation = {
            "backend": egraph.backend,
            "iterations": 0,
            "enodes": egraph.enode_count,
            "eclasses": egraph.eclass_count,
            "rule_matches": {},
            "rule_applications": {},
            "stop_reason": "ROOTS_ALREADY_CONGRUENT",
            "phase": "NO_REWRITE_REQUIRED",
        }

    unmatched = [item for item in root_pairs if not egraph.equivalent(item[3], item[4])]
    if unmatched and spec.llm.enabled:
        candidates = [(str(label), lhs, rhs) for label, lhs, rhs, _, _ in unmatched]
        try:
            assistance = propose_rules(spec, candidates)
            report["proof"]["llm_assistance"] = dict(assistance.audit)
            if assistance.rules:
                ordinary_rules, ordinary_admission = admitted_rules(
                    spec, assistance.rules
                )
                accepted_rules = tuple(relational_rules) + tuple(ordinary_rules)
                rule_admission = tuple(relational_admission) + tuple(
                    ordinary_admission
                )
                llm_saturation = _label_saturation(
                    egraph.saturate(ordinary_rules, spec.limits),
                    "LLM_ASSISTED_REWRITES",
                )
                saturation = _combine_saturation(saturation, llm_saturation)
                unmatched = [
                    item
                    for item in root_pairs
                    if not egraph.equivalent(item[3], item[4])
                ]
        except LLMError as exc:
            report["proof"]["llm_assistance"] = {
                "enabled": True,
                "provider": spec.llm.provider,
                "model": spec.llm.model,
                "status": "failed",
                "error": str(exc),
            }

    report["proof"]["rule_admission"] = list(rule_admission)
    report["proof"]["rule_validation"] = list(rule_admission)
    rule_matches = saturation["rule_matches"]
    rule_log = []
    for admission in rule_admission:
        entry = dict(admission)
        entry["matches"] = rule_matches.get(admission["id"], 0)
        entry["used"] = entry["matches"] > 0
        rule_log.append(entry)
    unverified_rule_uses = [
        entry
        for entry in rule_log
        if entry["status"] == "admitted_unverified" and entry["used"]
    ]
    trusted_rule_uses = [
        entry for entry in unverified_rule_uses if entry["kind"] == "trusted_fact"
    ]
    if unverified_rule_uses:
        trusted_ids = ", ".join(entry["id"] for entry in unverified_rule_uses)
        report["trusted_axioms"].append(
            f"unverified applied rewrite rule(s): {trusted_ids}"
        )
        report["guarantees"]["does_not_prove"].append(
            "soundness of rewrite rules admitted without a successful validation proof"
        )

    report["proof"]["egraph"] = {
        "stats": saturation,
        "initial_state": initial_state,
        "after_fact_rewrites": {
            "unmatched_root_pairs": len(unmatched_after_facts),
            "rules": [rule.rule_id for rule in relational_rules],
        },
        "root_pairs": len(root_pairs),
        "equivalent_root_pairs": len(root_pairs) - len(unmatched),
        "admitted_rule_ids": [rule.rule_id for rule in accepted_rules],
        "rule_log": rule_log,
        "rule_application": [
            {
                "id": entry["id"],
                "matches": entry["matches"],
                "used": entry["used"],
                "admission_status": entry["status"],
            }
            for entry in rule_log
        ],
        "accepted_rules": [
            result
            for result in rule_admission
            if result["status"] in {"proved", "admitted_unverified"}
        ],
        "trusted_rule_uses": trusted_rule_uses,
        "unverified_rule_uses": unverified_rule_uses,
    }

    if unmatched:
        logical_index, lhs_expr, rhs_expr, _, _ = unmatched[0]
        witness = _counterexample(lhs_expr, rhs_expr, logical_index)
        if witness is not None:
            report["blocks"].append(
                _block(
                    "COMPUTE",
                    Status.DISPROVED,
                    "a concrete abstract-value assignment makes the computations differ",
                    reason="COMPUTE_MISMATCH",
                    details=witness,
                )
            )
            return _finish(report, Status.DISPROVED, "COMPUTE_MISMATCH", witness)
        details = {
            "logical_index": logical_index,
            "lhs": lhs_expr.render(),
            "rhs": rhs_expr.render(),
            "egraph_stop_reason": saturation["stop_reason"],
        }
        report["blocks"].append(
            _block(
                "COMPUTE",
                Status.UNKNOWN,
                "the admitted rewrite rules do not connect the compute roots",
                reason="EGRAPH_NOT_EQUIVALENT",
                details=details,
            )
        )
        return _finish(report, Status.UNKNOWN, "EGRAPH_NOT_EQUIVALENT")

    used_levels = [ProofLevel.CONGRUENCE]
    matched_ids = set(saturation["rule_matches"])
    admission_by_id = {entry["id"]: entry for entry in rule_admission}
    if any(
        admission_by_id.get(rule_id, {}).get("kind") == "algebraic"
        and admission_by_id[rule_id]["status"] == "proved"
        for rule_id in matched_ids
    ):
        used_levels.append(ProofLevel.ALGEBRAIC)
    if any(
        admission_by_id.get(rule_id, {}).get("kind") == "relational"
        for rule_id in matched_ids
    ):
        used_levels.append(ProofLevel.PARAMETRIC_SMT)
    if unverified_rule_uses:
        used_levels.append(ProofLevel.TRUSTED_AXIOM)
    report["blocks"].append(
        _block(
            "COMPUTE",
            Status.PROVED,
            "every required pair of symbolic output values belongs to the same e-class",
            used_levels,
            details={
                "root_pairs": len(root_pairs),
                "rule_matches": saturation["rule_matches"],
                "stop_reason": saturation["stop_reason"],
            },
        )
    )
    report["blocks"].append(
        _block(
            "STORE",
            Status.PROVED,
            "same domain, address, value, and frame condition imply equal final Output memory",
            tuple(store_levels),
            details={
                "memory_model": "store(M, Output, offset, value, mask) -> M'",
                "frame_condition": "all non-Output logical blocks are unchanged",
            },
        )
    )
    return _finish(report, Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT")


def _merge_rule_logs(reports: Sequence[Mapping[str, Any]]) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in reports:
        for entry in item["proof"]["egraph"]["rule_log"]:
            if entry["id"] not in merged:
                merged[entry["id"]] = {**dict(entry), "matches": 0, "used": False}
            current = merged[entry["id"]]
            current["matches"] = current.get("matches", 0) + entry.get("matches", 0)
            current["used"] = current["matches"] > 0
    return list(merged.values())


def _partitioned_egraph_report(
    plan: PartitionPlan, subreports: Sequence[Mapping[str, Any]]
) -> dict:
    egraphs = [item["proof"]["egraph"] for item in subreports]
    rule_log = _merge_rule_logs(subreports)
    rule_matches = {
        entry["id"]: entry["matches"] for entry in rule_log if entry["matches"]
    }
    admitted_rule_ids = list(
        dict.fromkeys(
            rule_id
            for egraph in egraphs
            for rule_id in egraph["admitted_rule_ids"]
        )
    )
    root_pairs = sum(item["root_pairs"] for item in egraphs)
    iteration_trace = [
        {
            **trace,
            "subgraph": batch.partition_id,
            "phase": f"SUBGRAPH[{batch.partition_id}]/{trace.get('phase', 'UNSPECIFIED')}",
        }
        for batch, egraph in zip(plan.batches, egraphs)
        for trace in egraph["stats"].get("iteration_trace", [])
    ]
    return {
        "stats": {
            "backend": egraphs[0]["stats"]["backend"],
            "phase": "PAIRED_SUBGRAPH_DECOMPOSITION",
            "iterations": sum(item["stats"]["iterations"] for item in egraphs),
            "enodes": sum(item["stats"]["enodes"] for item in egraphs),
            "eclasses": sum(item["stats"]["eclasses"] for item in egraphs),
            "rule_matches": rule_matches,
            "rule_applications": rule_matches,
            "stop_reason": "ALL_SUBGRAPHS_EQUIVALENT",
            "subgraph_count": len(egraphs),
            "iteration_trace": iteration_trace,
            "phases": [
                {
                    "subgraph": batch.partition_id,
                    **egraph["stats"],
                }
                for batch, egraph in zip(plan.batches, egraphs)
            ],
        },
        "initial_state": {
            "unmatched_root_pairs": sum(
                item["initial_state"]["unmatched_root_pairs"] for item in egraphs
            ),
            "roots": [],
        },
        "after_fact_rewrites": {
            "unmatched_root_pairs": sum(
                item["after_fact_rewrites"]["unmatched_root_pairs"]
                for item in egraphs
            ),
            "rules": list(
                dict.fromkeys(
                    rule_id
                    for item in egraphs
                    for rule_id in item["after_fact_rewrites"]["rules"]
                )
            ),
        },
        "root_pairs": root_pairs,
        "equivalent_root_pairs": root_pairs,
        "admitted_rule_ids": admitted_rule_ids,
        "rule_log": rule_log,
        "rule_application": [
            {
                "id": entry["id"],
                "matches": entry["matches"],
                "used": entry["used"],
                "admission_status": entry["status"],
            }
            for entry in rule_log
        ],
        "accepted_rules": list(
            {
                entry["id"]: entry
                for item in egraphs
                for entry in item["accepted_rules"]
            }.values()
        ),
        "trusted_rule_uses": [
            entry
            for entry in rule_log
            if entry["status"] == "admitted_unverified"
            and entry["kind"] == "trusted_fact"
            and entry["used"]
        ],
        "unverified_rule_uses": [
            entry
            for entry in rule_log
            if entry["status"] == "admitted_unverified" and entry["used"]
        ],
        "subgraphs": [
            {
                "id": batch.partition_id,
                "family": batch.family_id,
                "semantic": batch.semantic,
                "dependencies": list(batch.dependencies),
                "root_pairs": len(batch.root_pairs),
                "egraph": subreport["proof"]["egraph"],
            }
            for batch, subreport in zip(plan.batches, subreports)
        ],
    }


def _prove_partition_plan(
    report: dict,
    spec: PairSpec,
    plan: PartitionPlan,
    store_levels: Sequence[ProofLevel],
    relational_rules: Sequence[Any],
    relational_admission: Sequence[Mapping[str, Any]],
) -> Optional[dict]:
    sub_spec = replace(
        spec,
        partition=PartitionConfig(),
        llm=replace(spec.llm, enabled=False),
    )
    subreports = []
    for batch in plan.batches:
        subreport = _complete_whole_compute_proof(
            _base_report(sub_spec),
            sub_spec,
            batch.root_pairs,
            (ProofLevel.DECOMPOSITION, ProofLevel.CONGRUENCE),
            relational_rules,
            relational_admission,
        )
        subreports.append(subreport)
        if subreport["status"] != Status.PROVED.value:
            report["proof"]["partitioning"] = {
                **dict(plan.audit),
                "status": "verification_failed",
                "failed_partition": batch.partition_id,
                "subgraph_status": subreport["status"],
                "subgraph_reason": subreport["reason"],
                "fallback": "whole_program",
            }
            report["blocks"].append(
                _block(
                    "SUBGRAPH_PARTITION",
                    Status.UNKNOWN,
                    "a proposed subgraph obligation was not proved; using whole-program verification",
                    reason="SUBGRAPH_PROOF_FAILED",
                    details={
                        "failed_partition": batch.partition_id,
                        "subgraph_status": subreport["status"],
                        "subgraph_reason": subreport["reason"],
                        "fallback": "whole_program",
                    },
                    required_for_final=False,
                )
            )
            return None

    egraph_report = _partitioned_egraph_report(plan, subreports)
    report["proof"]["partitioning"] = {
        **dict(plan.audit),
        "status": "proved",
        "proof_order": [batch.partition_id for batch in plan.batches],
        "composition": (
            "proved child pairs are replaced by identical typed boundary inputs "
            "before proving each parent pair"
        ),
    }
    report["proof"]["rule_admission"] = subreports[0]["proof"]["rule_admission"]
    report["proof"]["rule_validation"] = subreports[0]["proof"]["rule_validation"]
    report["proof"]["egraph"] = egraph_report
    unverified = egraph_report["unverified_rule_uses"]
    if unverified:
        trusted_ids = ", ".join(item["id"] for item in unverified)
        report["trusted_axioms"].append(
            f"unverified applied rewrite rule(s): {trusted_ids}"
        )
        report["guarantees"]["does_not_prove"].append(
            "soundness of rewrite rules admitted without a successful validation proof"
        )

    report["blocks"].append(
        _block(
            "SUBGRAPH_PARTITION",
            Status.PROVED,
            "the LLM proposal passed structural checks and every paired subgraph was proved",
            (ProofLevel.STRUCTURAL, ProofLevel.DECOMPOSITION),
            details={
                "partitions": len(plan.batches),
                "proof_order": [batch.partition_id for batch in plan.batches],
                "whole_program_llm_calls": 1,
            },
        )
    )
    report["blocks"].append(
        _block(
            "DEFINEDNESS",
            Status.PROVED,
            "all partial floating operations in every exclusive subgraph have proved domains",
            (ProofLevel.STRUCTURAL, ProofLevel.DECOMPOSITION),
        )
    )
    compute_levels = [ProofLevel.CONGRUENCE, ProofLevel.DECOMPOSITION]
    for subreport in subreports:
        compute = next(
            block for block in subreport["blocks"] if block["kind"] == "COMPUTE"
        )
        for level in compute["proof_levels"]:
            parsed = ProofLevel(level)
            if parsed not in compute_levels:
                compute_levels.append(parsed)
    report["blocks"].append(
        _block(
            "COMPUTE",
            Status.PROVED,
            "all dependency-ordered paired subgraphs are equivalent and compose to equal roots",
            compute_levels,
            details={
                "subgraphs": len(plan.batches),
                "root_pairs": egraph_report["root_pairs"],
                "rule_matches": egraph_report["stats"]["rule_matches"],
            },
        )
    )
    report["blocks"].append(
        _block(
            "STORE",
            Status.PROVED,
            "same domain, address, composed value, and frame condition imply equal final Output memory",
            tuple(store_levels) + (ProofLevel.DECOMPOSITION,),
            details={
                "memory_model": "store(M, Output, offset, value, mask) -> M'",
                "frame_condition": "all non-Output logical blocks are unchanged",
            },
        )
    )
    return _finish(report, Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT")


def _complete_compute_proof(
    report: dict,
    spec: PairSpec,
    lhs_program: Program,
    rhs_program: Program,
    root_expressions: Sequence[Tuple[Any, Expr, Expr]],
    store_levels: Sequence[ProofLevel],
    relational_rules: Sequence[Any] = (),
    relational_admission: Sequence[Mapping[str, Any]] = (),
) -> dict:
    if spec.partition.enabled:
        try:
            plan = propose_partition_plan(
                spec, lhs_program, rhs_program, root_expressions
            )
            partitioned = _prove_partition_plan(
                report,
                spec,
                plan,
                store_levels,
                relational_rules,
                relational_admission,
            )
            if partitioned is not None:
                return partitioned
        except LLMError as exc:
            failure_audit = exc.audit if isinstance(exc, PartitionError) else {}
            report["proof"]["partitioning"] = {
                "enabled": True,
                "status": "proposal_rejected",
                "provider": spec.llm.provider,
                "configured_model": spec.llm.model,
                **failure_audit,
                "error": str(exc),
                "fallback": "whole_program",
            }
            report["blocks"].append(
                _block(
                    "SUBGRAPH_PARTITION",
                    Status.UNKNOWN,
                    "the LLM partition proposal failed machine checks; using whole-program verification",
                    reason="SUBGRAPH_PARTITION_REJECTED",
                    details={"error": str(exc), "fallback": "whole_program"},
                    required_for_final=False,
                )
            )
    return _complete_whole_compute_proof(
        report,
        spec,
        root_expressions,
        store_levels,
        relational_rules,
        relational_admission,
    )


def verify_pair(spec: PairSpec) -> dict:
    report = _base_report(spec)
    try:
        lhs_program = load_program_artifact(spec.lhs_path, spec.lhs_frontend)
        rhs_program = load_program_artifact(spec.rhs_path, spec.rhs_frontend)
    except UnsupportedSemantics as exc:
        report["unsupported"].append(str(exc))
        report["blocks"].append(
            _block("FRONTEND", Status.UNKNOWN, str(exc), reason=exc.code)
        )
        return _finish(report, Status.UNKNOWN, exc.code)
    report["inputs"]["frontends"] = {
        "lhs": {"name": lhs_program.frontend, "version": lhs_program.frontend_version},
        "rhs": {"name": rhs_program.frontend, "version": rhs_program.frontend_version},
    }
    if lhs_program.frontend == "libtriton" or rhs_program.frontend == "libtriton":
        report["trusted_axioms"].append(
            "ETV TTIR-to-Semantic-TTIR lifting implementation"
        )
        report["blocks"].append(
            _block(
                "FRONTEND",
                Status.PROVED,
                "raw modules were parsed and verified by pinned libtriton before semantic lifting",
                (ProofLevel.STRUCTURAL,),
                details=report["inputs"]["frontends"],
            )
        )
    if (
        lhs_program.frontend == "torch_prims_json"
        or rhs_program.frontend == "torch_prims_json"
    ):
        report["trusted_axioms"].append(
            "ETV Torch-Prims-to-internal-IR lifting implementation"
        )
        report["blocks"].append(
            _block(
                "FRONTEND",
                Status.PROVED,
                "the strict fixed-rank Torch Prims graph was type-checked and "
                "lifted directly without TorchInductor",
                (ProofLevel.STRUCTURAL,),
                details=report["inputs"]["frontends"],
            )
        )
    heterogeneous_symbolic_pair = {
        lhs_program.frontend,
        rhs_program.frontend,
    } == {"libtriton", "torch_prims_json"}
    if heterogeneous_symbolic_pair and not spec.facts.parameters:
        report["blocks"].append(
            _block(
                "PARAMETER_DOMAIN",
                Status.UNKNOWN,
                "TTIR-versus-Prims verification requires fixed-rank symbolic "
                "parameters, including singleton domains for fixed specializations",
                reason="SYMBOLIC_FACTS_REQUIRED",
            )
        )
        return _finish(report, Status.UNKNOWN, "SYMBOLIC_FACTS_REQUIRED")

    if spec.semantic_mode != "abstract_float":
        report["unsupported"].append(
            f"semantic mode {spec.semantic_mode!r}; MVP supports only abstract_float"
        )
        report["blocks"].append(
            _block(
                "SEMANTICS",
                Status.UNKNOWN,
                "requested numeric semantics are not implemented",
                reason="UNSUPPORTED_SEMANTIC_MODE",
            )
        )
        return _finish(report, Status.UNKNOWN, "UNSUPPORTED_SEMANTIC_MODE")

    missing_alias = [
        (lhs, rhs)
        for lhs, rhs in pairwise(spec.contract.require_disjoint)
        if not spec.facts.disjoint(lhs, rhs)
    ]
    if missing_alias:
        report["blocks"].append(
            _block(
                "ABI",
                Status.UNKNOWN,
                "required no-alias facts are missing",
                reason="MISSING_ALIAS_FACT",
                details={"missing_pairs": [list(pair) for pair in missing_alias]},
            )
        )
        return _finish(report, Status.UNKNOWN, "MISSING_ALIAS_FACT")

    role_details = [
        {
            "logical": role.logical,
            "lhs": {
                "kind": role.lhs.kind,
                "name": role.lhs.name,
                "index": role.lhs.index,
            },
            "rhs": {
                "kind": role.rhs.kind,
                "name": role.rhs.name,
                "index": role.rhs.index,
            },
        }
        for role in spec.roles
    ]
    report["blocks"].append(
        _block(
            "ABI",
            Status.PROVED,
            "physical parameters are aligned to explicit logical roles",
            (ProofLevel.TRUSTED_AXIOM,),
            details={"roles": role_details},
        )
    )

    if spec.facts.parameters:
        try:
            symbolic = verify_parametric_pair(lhs_program, rhs_program, spec)
        except ParametricFailure as exc:
            report["blocks"].append(
                _block(
                    exc.block,
                    exc.status,
                    exc.summary,
                    reason=exc.reason,
                    details=exc.details,
                )
            )
            if exc.status == Status.UNKNOWN:
                report["unsupported"].append(exc.summary)
            return _finish(
                report,
                exc.status,
                exc.reason,
                exc.details if exc.status == Status.DISPROVED else None,
            )
        report["blocks"].extend(symbolic.blocks)
        report["proof"]["parametric_domain"] = dict(symbolic.proof)
        return _complete_compute_proof(
            report,
            spec,
            lhs_program,
            rhs_program,
            symbolic.root_pairs,
            (ProofLevel.PARAMETRIC_SMT, ProofLevel.CONGRUENCE),
            symbolic.rewrite_rules,
            symbolic.rewrite_admission,
        )

    try:
        expected_numel = _expected_numel(spec)
        lhs = evaluate_program(lhs_program, spec, "lhs")
        rhs = evaluate_program(rhs_program, spec, "rhs")
    except UnsupportedSemantics as exc:
        report["unsupported"].append(str(exc))
        report["blocks"].append(
            _block("LIFT_EVAL", Status.UNKNOWN, str(exc), reason=exc.code)
        )
        return _finish(report, Status.UNKNOWN, exc.code)

    lhs_map, lhs_duplicate = _active_map(lhs)
    rhs_map, rhs_duplicate = _active_map(rhs)
    if lhs_duplicate or rhs_duplicate:
        witness = lhs_duplicate or rhs_duplicate
        report["blocks"].append(
            _block(
                "INDEX",
                Status.UNKNOWN,
                "a logical output element is assigned by multiple lanes",
                reason="DUPLICATE_LOGICAL_WRITE",
                details=witness,
            )
        )
        return _finish(report, Status.UNKNOWN, "DUPLICATE_LOGICAL_WRITE", witness)

    report["blocks"].append(
        _block(
            "INDEX",
            Status.PROVED,
            "all launch programs and lanes were exhaustively enumerated",
            (ProofLevel.BOUNDED_EXHAUSTIVE,),
            details={
                "lhs": {
                    "programs": lhs.program_count,
                    "lanes": lhs.program.lanes,
                    "enumerated": len(lhs.records),
                },
                "rhs": {
                    "programs": rhs.program_count,
                    "lanes": rhs.program.lanes,
                    "enumerated": len(rhs.records),
                },
            },
        )
    )

    lhs_domain = set(lhs_map)
    rhs_domain = set(rhs_map)
    if lhs_domain != rhs_domain:
        differing = sorted(lhs_domain.symmetric_difference(rhs_domain))[0]
        witness = {
            "kind": "MASK_DOMAIN_MISMATCH",
            "logical_index": differing,
            "lhs_active": differing in lhs_domain,
            "rhs_active": differing in rhs_domain,
        }
        report["blocks"].append(
            _block(
                "MASK",
                Status.DISPROVED,
                "the active logical output domains differ",
                (ProofLevel.BOUNDED_EXHAUSTIVE,),
                reason="MASK_MISMATCH",
                details=witness,
            )
        )
        return _finish(report, Status.DISPROVED, "MASK_MISMATCH", witness)
    report["blocks"].append(
        _block(
            "MASK",
            Status.PROVED,
            "both store masks select the same logical output domain",
            (ProofLevel.BOUNDED_EXHAUSTIVE,),
            details={"active_elements": len(lhs_domain)},
        )
    )

    expected_domain = set(range(expected_numel))
    if spec.contract.require_full_coverage and lhs_domain != expected_domain:
        missing = sorted(expected_domain - lhs_domain)
        extra = sorted(lhs_domain - expected_domain)
        details = {
            "missing": missing[:16],
            "extra": extra[:16],
            "expected_numel": expected_numel,
        }
        report["blocks"].append(
            _block(
                "COVERAGE",
                Status.UNKNOWN,
                "both kernels agree but do not establish the requested full-output contract",
                (ProofLevel.BOUNDED_EXHAUSTIVE,),
                reason="INCOMPLETE_COVERAGE",
                details=details,
            )
        )
        return _finish(report, Status.UNKNOWN, "INCOMPLETE_COVERAGE")
    report["blocks"].append(
        _block(
            "COVERAGE",
            Status.PROVED,
            "the active lanes cover every contracted output element exactly once",
            (ProofLevel.BOUNDED_EXHAUSTIVE,),
            details={"output_numel": expected_numel},
        )
    )

    race = _race(lhs) or _race(rhs)
    if race:
        report["blocks"].append(
            _block(
                "RACE_FREEDOM",
                Status.UNKNOWN,
                "two active lanes target the same output address",
                reason="RACE_FREEDOM_NOT_PROVED",
                details=race,
            )
        )
        return _finish(report, Status.UNKNOWN, "RACE_FREEDOM_NOT_PROVED", race)
    report["blocks"].append(
        _block(
            "RACE_FREEDOM",
            Status.PROVED,
            "active store addresses are injective on both sides",
            (ProofLevel.BOUNDED_EXHAUSTIVE,),
        )
    )

    for logical_index in sorted(lhs_domain):
        lhs_record = lhs_map[logical_index]
        rhs_record = rhs_map[logical_index]
        if (
            lhs_record.output_role != spec.contract.output_role
            or rhs_record.output_role != spec.contract.output_role
            or lhs_record.offset != rhs_record.offset
        ):
            witness = {
                "kind": "OUTPUT_ADDRESS_MISMATCH",
                "logical_index": logical_index,
                "lhs": {"role": lhs_record.output_role, "offset": lhs_record.offset},
                "rhs": {"role": rhs_record.output_role, "offset": rhs_record.offset},
            }
            report["blocks"].append(
                _block(
                    "ADDRESS",
                    Status.DISPROVED,
                    "corresponding logical elements write different observable addresses",
                    (ProofLevel.BOUNDED_EXHAUSTIVE,),
                    reason="OUTPUT_ADDRESS_MISMATCH",
                    details=witness,
                )
            )
            return _finish(report, Status.DISPROVED, "OUTPUT_ADDRESS_MISMATCH", witness)
    report["blocks"].append(
        _block(
            "ADDRESS",
            Status.PROVED,
            "all corresponding logical elements write the same Output offsets",
            (ProofLevel.BOUNDED_EXHAUSTIVE,),
        )
    )

    load_mismatches = []
    for logical_index in sorted(lhs_domain):
        lhs_signature = _leaf_signature(lhs_map[logical_index].value)
        rhs_signature = _leaf_signature(rhs_map[logical_index].value)
        if lhs_signature != rhs_signature:
            load_mismatches.append(
                {
                    "logical_index": logical_index,
                    "lhs": list(lhs_signature),
                    "rhs": list(rhs_signature),
                }
            )
            if len(load_mismatches) == 4:
                break
    if load_mismatches:
        report["blocks"].append(
            _block(
                "LOAD",
                Status.UNKNOWN,
                "input dependency frontiers differ; compute proof will decide observability",
                reason="LOAD_FRONTIER_MISMATCH",
                details={"examples": load_mismatches},
                required_for_final=False,
            )
        )
    else:
        report["blocks"].append(
            _block(
                "LOAD",
                Status.PROVED,
                "all output values depend on the same logical input/scalar leaves",
                (ProofLevel.BOUNDED_EXHAUSTIVE, ProofLevel.TRUSTED_AXIOM),
            )
        )

    report["proof"]["finite_domain"] = {
        "lhs_lanes": len(lhs.records),
        "rhs_lanes": len(rhs.records),
        "active_output_elements": len(lhs_domain),
        "output_numel": expected_numel,
        "complete_for_specialization": True,
    }
    root_expressions = [
        (logical_index, lhs_map[logical_index].value, rhs_map[logical_index].value)
        for logical_index in sorted(lhs_domain)
    ]
    return _complete_compute_proof(
        report,
        spec,
        lhs_program,
        rhs_program,
        root_expressions,
        (ProofLevel.BOUNDED_EXHAUSTIVE, ProofLevel.CONGRUENCE),
    )


def verify_spec(path: Path) -> dict:
    path = Path(path)
    try:
        spec = load_pair_spec(path)
        return verify_pair(spec)
    except InputError as exc:
        return _invalid_report(path, exc)
