"""End-to-end bounded translation validation for Semantic TTIR pairs."""

from __future__ import annotations

import hashlib
import itertools
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .egraph import EGraph
from .evaluator import SideRoles, eval_expr, evaluate_program
from .model import (
    Evaluation,
    Expr,
    InputError,
    PairSpec,
    ProofLevel,
    Sort,
    Status,
    UnsupportedSemantics,
    pairwise,
)
from .schema import load_pair_spec, load_program
from .z3_validator import validated_builtin_rules


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
    assumptions = [
        {
            "text": f"{name} = {value}",
            "evidence": ProofLevel.TRUSTED_AXIOM.value,
            "source": "PairSpec.facts.bindings",
        }
        for name, value in sorted(spec.facts.bindings.items())
    ]
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
        "schema_version": 1,
        "tool": {"name": "ETV", "version": "0.1.0"},
        "pair_id": spec.pair_id,
        "status": Status.UNKNOWN.value,
        "reason": "NOT_RUN",
        "semantic_mode": spec.semantic_mode,
        "scope": "fixed-specialization single-store bounded translation validation",
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
            "PairSpec fixed shape/stride/launch bindings",
            "PairSpec no-alias declarations",
            "ABSTRACT_FLOAT interprets floating operations over exact mathematical values",
            "Semantic TTIR evaluator and memory-token model",
        ],
        "blocks": [],
        "unsupported": [],
        "counterexample": None,
        "proof": {},
        "guarantees": {
            "proves": "equal final Output memory for all abstract input values in the enumerated launch domain",
            "does_not_prove": [
                "IEEE-754 or bitwise GPU equality",
                "correct lifting from raw TTIR to Semantic TTIR",
                "parametric shapes or unbounded loops",
                "concurrent/shared-memory semantics",
                "allocated-buffer bounds or GPU memory safety",
            ],
        },
    }


def _finish(report: dict, status: Status, reason: str, counterexample: Optional[dict] = None) -> dict:
    report["status"] = status.value
    report["reason"] = reason
    report["counterexample"] = counterexample
    if status == Status.PROVED:
        report["guarantees"]["establishes"] = report["guarantees"].pop("proves")
    elif status == Status.DISPROVED:
        report["guarantees"].pop("proves", None)
        report["guarantees"]["establishes"] = (
            "the reported concrete witness violates observable equivalence in ABSTRACT_FLOAT semantics"
        )
    else:
        report["guarantees"].pop("proves", None)
        report["guarantees"]["establishes"] = (
            "no equivalence or inequivalence conclusion; the reason identifies the first unmet obligation"
        )
    return report


def _invalid_report(path: Path, exc: Exception) -> dict:
    return {
        "schema_version": 1,
        "tool": {"name": "ETV", "version": "0.1.0"},
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
                "first": {"pid": previous.pid, "lane": previous.lane, "offset": previous.offset},
                "second": {"pid": record.pid, "lane": record.lane, "offset": record.offset},
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


def _collect_ops(expr: Expr, result: Set[str]) -> None:
    result.add(expr.op)
    for arg in expr.args:
        _collect_ops(arg, result)


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
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


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
        for values in itertools.product((Fraction(-1), Fraction(0), Fraction(1)), repeat=len(leaves)):
            yield dict(zip(leaves, values))


def _counterexample(lhs: Expr, rhs: Expr, logical_index: int) -> Optional[dict]:
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
                    "assignment": {key: _fraction_text(value) for key, value in assignment.items()},
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


def verify_pair(spec: PairSpec) -> dict:
    lhs_program = load_program(spec.lhs_path)
    rhs_program = load_program(spec.rhs_path)
    report = _base_report(spec)

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
            "lhs": {"kind": role.lhs.kind, "name": role.lhs.name, "index": role.lhs.index},
            "rhs": {"kind": role.rhs.kind, "name": role.rhs.name, "index": role.rhs.index},
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
                "lhs": {"programs": lhs.program_count, "lanes": lhs.program.lanes, "enumerated": len(lhs.records)},
                "rhs": {"programs": rhs.program_count, "lanes": rhs.program.lanes, "enumerated": len(rhs.records)},
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
        details = {"missing": missing[:16], "extra": extra[:16], "expected_numel": expected_numel}
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

    definedness_issues = []
    for logical_index in sorted(lhs_domain):
        for side, expression in (
            ("lhs", lhs_map[logical_index].value),
            ("rhs", rhs_map[logical_index].value),
        ):
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

    accepted_rules, rule_validation = validated_builtin_rules()
    report["proof"]["rule_validation"] = list(rule_validation)
    egraph = EGraph()
    root_pairs = []
    for logical_index in sorted(lhs_domain):
        root_pairs.append(
            (
                logical_index,
                egraph.add_expr(lhs_map[logical_index].value),
                egraph.add_expr(rhs_map[logical_index].value),
            )
        )
    egraph.rebuild()
    initially_unmatched = [
        (logical_index, lhs_root, rhs_root)
        for logical_index, lhs_root, rhs_root in root_pairs
        if not egraph.equivalent(lhs_root, rhs_root)
    ]
    if initially_unmatched:
        relevant_ops: Set[str] = set()
        for logical_index, _, _ in initially_unmatched:
            _collect_ops(lhs_map[logical_index].value, relevant_ops)
            _collect_ops(rhs_map[logical_index].value, relevant_ops)
        rule_families = {
            "fadd": {"fadd_comm", "fadd_assoc", "fadd_zero"},
            "fmul": {"fmul_comm", "fmul_assoc", "fmul_one", "fmul_zero"},
            "fdiv": {"fdiv_one"},
            "fsub": {"fsub_def"},
            "fneg": {"fneg_involution"},
            "fma": {"fma_def"},
        }
        relevant_rule_ids = set()
        for op in relevant_ops:
            relevant_rule_ids.update(rule_families.get(op, set()))
        candidate_rules = tuple(rule for rule in accepted_rules if rule.rule_id in relevant_rule_ids)
        saturation = egraph.saturate(candidate_rules, spec.limits)
    else:
        candidate_rules = ()
        saturation = {
            "iterations": 0,
            "enodes": egraph.enode_count,
            "eclasses": len(egraph.roots()),
            "merges": len(egraph.merge_log),
            "rule_applications": {},
            "stop_reason": "ROOTS_ALREADY_CONGRUENT",
        }
    unmatched = [
        (logical_index, lhs_root, rhs_root)
        for logical_index, lhs_root, rhs_root in root_pairs
        if not egraph.equivalent(lhs_root, rhs_root)
    ]
    report["proof"]["egraph"] = {
        "stats": saturation,
        "root_pairs": len(root_pairs),
        "equivalent_root_pairs": len(root_pairs) - len(unmatched),
        "candidate_rule_ids": [rule.rule_id for rule in candidate_rules],
        "merge_ledger": egraph.merge_log,
        "accepted_rules": [result for result in rule_validation if result["status"] == "proved"],
    }

    if unmatched:
        logical_index, _, _ = unmatched[0]
        lhs_expr = lhs_map[logical_index].value
        rhs_expr = rhs_map[logical_index].value
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
                "the accepted algebraic rules do not connect the compute roots",
                reason="EGRAPH_NOT_EQUIVALENT",
                details=details,
            )
        )
        return _finish(report, Status.UNKNOWN, "EGRAPH_NOT_EQUIVALENT")

    used_levels = [ProofLevel.CONGRUENCE]
    if saturation["rule_applications"]:
        used_levels.append(ProofLevel.ALGEBRAIC)
    report["blocks"].append(
        _block(
            "COMPUTE",
            Status.PROVED,
            "every pair of symbolic output values belongs to the same e-class",
            used_levels,
            details={
                "root_pairs": len(root_pairs),
                "rule_applications": saturation["rule_applications"],
                "stop_reason": saturation["stop_reason"],
            },
        )
    )
    report["blocks"].append(
        _block(
            "STORE",
            Status.PROVED,
            "same mask, address, value, and initial memory imply the same final Output memory",
            (ProofLevel.BOUNDED_EXHAUSTIVE, ProofLevel.CONGRUENCE),
            details={
                "memory_model": "store(M, Output, offset, value, mask) -> M'",
                "frame_condition": "all non-Output logical blocks are unchanged",
            },
        )
    )
    report["proof"]["finite_domain"] = {
        "lhs_lanes": len(lhs.records),
        "rhs_lanes": len(rhs.records),
        "active_output_elements": len(lhs_domain),
        "output_numel": expected_numel,
        "complete_for_specialization": True,
    }
    return _finish(report, Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT")


def verify_spec(path: Path) -> dict:
    path = Path(path)
    try:
        spec = load_pair_spec(path)
        return verify_pair(spec)
    except InputError as exc:
        return _invalid_report(path, exc)
