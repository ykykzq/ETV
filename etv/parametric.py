"""Fixed-rank symbolic obligations and predicate-derived relational rewrites."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import z3

from .casts import INTEGER_CAST_OPS, integer_width, normalize_cast_data
from .evaluator import SideRoles, build_side_roles
from .ir import Expr, Program, Sort, bool_const, int_const
from .model import (
    PairSpec,
    ProofLevel,
    Status,
)
from .rules import Rule, node, pattern_from_expr, render_pattern, var

I32_MIN = -(2**31)
I32_MAX = 2**31 - 1


@dataclass(frozen=True)
class SMTTerm:
    value: Any
    conditions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SideState:
    side: str
    program: Program
    context: "SMTContext"
    roles: SideRoles
    programs: Any
    candidate_env: Mapping[str, Any]
    candidate_offset: Any
    candidate_value: Expr


@dataclass(frozen=True)
class ParametricResult:
    blocks: tuple[dict, ...]
    proof: Mapping[str, Any]
    root_pairs: tuple[tuple[str, Expr, Expr], ...]
    rewrite_rules: tuple[Rule, ...] = ()
    rewrite_admission: tuple[dict, ...] = ()


class ParametricFailure(RuntimeError):
    def __init__(
        self,
        block: str,
        reason: str,
        summary: str,
        details: Mapping[str, Any],
        status: Status = Status.UNKNOWN,
    ) -> None:
        super().__init__(summary)
        self.block = block
        self.reason = reason
        self.summary = summary
        self.details = dict(details)
        self.status = status


def _and(values: Sequence[Any]) -> Any:
    return z3.And(*values) if values else z3.BoolVal(True)


def _i32(value: Any) -> Any:
    return z3.And(value >= I32_MIN, value <= I32_MAX)


def _abs(value: Any) -> Any:
    return z3.If(value >= 0, value, -value)


def _trunc_div(lhs: Any, rhs: Any) -> Any:
    quotient = _abs(lhs) / _abs(rhs)
    return z3.If(z3.Xor(lhs < 0, rhs < 0), -quotient, quotient)


def _unsigned_wrap(value: Any, width: int) -> Any:
    return value % (1 << width)


def _signed_wrap(value: Any, width: int) -> Any:
    unsigned = _unsigned_wrap(value, width)
    return z3.If(unsigned >= (1 << (width - 1)), unsigned - (1 << width), unsigned)


class SMTContext:
    def __init__(
        self,
        spec: PairSpec,
        parameters: Mapping[str, Any],
        side: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.parameters = parameters
        self.side = side
        self.facts = spec.facts if side is None else spec.facts.for_side(side)

    def term(
        self,
        expression: Expr,
        env: Mapping[str, Any],
        resolving: tuple[str, ...] = (),
    ) -> SMTTerm:
        op = expression.op
        if op == "const_int":
            return SMTTerm(z3.IntVal(int(expression.data)))
        if op == "const_bool":
            return SMTTerm(z3.BoolVal(bool(expression.data)))
        if op == "var":
            name = str(expression.data)
            if name in env:
                return SMTTerm(env[name])
            if name in self.facts.bindings:
                if name in resolving:
                    raise ParametricFailure(
                        "PARAMETER_DOMAIN",
                        "CYCLIC_SYMBOLIC_BINDING",
                        f"symbolic binding cycle contains {name!r}",
                        {"cycle": list((*resolving, name))},
                    )
                value = self.facts.bindings[name]
                if isinstance(value, Expr):
                    return self.term(value, env, (*resolving, name))
                return SMTTerm(z3.IntVal(int(value)))
            if name in self.parameters:
                return SMTTerm(self.parameters[name])
            raise ParametricFailure(
                "PARAMETER_DOMAIN",
                "UNBOUND_VARIABLE",
                f"unbound symbolic integer variable {name!r}",
                {"variable": name, "side": self.side},
            )

        children = tuple(
            self.term(argument, env, resolving) for argument in expression.args
        )
        conditions = tuple(
            condition for child in children for condition in child.conditions
        )
        values = tuple(child.value for child in children)
        if op in INTEGER_CAST_OPS:
            try:
                source_type, result_type = normalize_cast_data(expression.data)
                source_width = integer_width(source_type)
                result_width = integer_width(result_type)
            except ValueError as exc:
                raise ParametricFailure(
                    "PARAMETER_DOMAIN",
                    "SYMBOLIC_INTEGER_CAST_UNSUPPORTED",
                    f"cannot encode integer cast {expression.render()}: {exc}",
                    {"operation": op, "side": self.side},
                ) from exc
            raw = z3.If(values[0], 1, 0) if z3.is_bool(values[0]) else values[0]
            if op == "sext" and result_width > source_width:
                result = _signed_wrap(raw, source_width)
            elif op == "zext" and result_width > source_width:
                result = _unsigned_wrap(raw, source_width)
            elif op == "trunc" and result_width < source_width:
                truncated = _signed_wrap(raw, result_width)
                result = truncated != 0 if result_type == "i1" else truncated
            else:
                raise ParametricFailure(
                    "PARAMETER_DOMAIN",
                    "SYMBOLIC_INTEGER_CAST_UNSUPPORTED",
                    f"invalid or target-dependent integer cast {expression.render()}",
                    {"operation": op, "side": self.side},
                )
            result_conditions = (
                conditions if result_type == "i1" else (*conditions, _i32(result))
            )
            return SMTTerm(result, result_conditions)
        if op == "iadd":
            result = values[0] + values[1]
            return SMTTerm(result, (*conditions, _i32(result)))
        if op == "isub":
            result = values[0] - values[1]
            return SMTTerm(result, (*conditions, _i32(result)))
        if op == "imul":
            result = values[0] * values[1]
            return SMTTerm(result, (*conditions, _i32(result)))
        if op in {"idiv", "irem"}:
            quotient = _trunc_div(values[0], values[1])
            result = quotient if op == "idiv" else values[0] - quotient * values[1]
            defined = z3.And(
                values[1] != 0,
                z3.Not(z3.And(values[0] == I32_MIN, values[1] == -1)),
                _i32(result),
            )
            return SMTTerm(result, (*conditions, defined))
        if op == "ceildiv":
            result = (values[0] + values[1] - 1) / values[1]
            return SMTTerm(
                result,
                (*conditions, values[0] >= 0, values[1] > 0, _i32(result)),
            )
        if op == "lt":
            return SMTTerm(values[0] < values[1], conditions)
        if op == "le":
            return SMTTerm(values[0] <= values[1], conditions)
        if op == "gt":
            return SMTTerm(values[0] > values[1], conditions)
        if op == "ge":
            return SMTTerm(values[0] >= values[1], conditions)
        if op == "eq":
            return SMTTerm(values[0] == values[1], conditions)
        if op == "ne":
            return SMTTerm(values[0] != values[1], conditions)
        if op == "and":
            return SMTTerm(z3.And(values[0], values[1]), conditions)
        if op == "or":
            return SMTTerm(z3.Or(values[0], values[1]), conditions)
        if op == "not":
            return SMTTerm(z3.Not(values[0]), conditions)
        if op == "select":
            result = z3.If(values[0], values[1], values[2])
            result_conditions = (
                (*conditions, _i32(result))
                if expression.sort == Sort.INT
                else conditions
            )
            return SMTTerm(result, result_conditions)
        raise ParametricFailure(
            "PARAMETER_DOMAIN",
            "SYMBOLIC_OPERATION_UNSUPPORTED",
            f"operation {op!r} is not supported by the symbolic integer encoder",
            {"operation": op, "side": self.side},
        )


class Prover:
    def __init__(
        self, base: Sequence[Any], parameters: Mapping[str, Any], timeout_ms: int
    ) -> None:
        self.base = tuple(base)
        self.parameters = dict(parameters)
        self.timeout_ms = timeout_ms
        self.checks: list[dict] = []

    def check_bad(self, name: str, bad: Any, witnesses: Sequence[Any] = ()) -> dict:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(*self.base, bad)
        formula = solver.sexpr()
        outcome = solver.check()
        result: dict[str, Any] = {
            "name": name,
            "solver": "z3",
            "solver_version": z3.get_version_string(),
            "query": "domain and counterexample",
            "query_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest(),
            "result": str(outcome),
        }
        if outcome == z3.sat:
            model = solver.model()
            values = {
                name: str(model.eval(symbol, model_completion=True))
                for name, symbol in sorted(self.parameters.items())
            }
            for symbol in witnesses:
                values[str(symbol)] = str(model.eval(symbol, model_completion=True))
            result["counterexample"] = values
        self.checks.append(result)
        return result

    def require_unsat(
        self,
        block: str,
        reason: str,
        name: str,
        bad: Any,
        summary: str,
        witnesses: Sequence[Any] = (),
        status: Status = Status.UNKNOWN,
    ) -> dict:
        result = self.check_bad(name, bad, witnesses)
        if result["result"] != "unsat":
            raise ParametricFailure(block, reason, summary, result, status=status)
        return result


def _substitute(expression: Expr, env: Mapping[str, Expr]) -> Expr:
    if expression.op == "var" and str(expression.data) in env:
        return env[str(expression.data)]
    if not expression.args:
        return expression
    return Expr(
        expression.op,
        args=tuple(_substitute(argument, env) for argument in expression.args),
        data=expression.data,
        sort=expression.sort,
    )


class SymbolicValueRewriter:
    def __init__(
        self,
        context: SMTContext,
        roles: SideRoles,
        prover: Prover,
        antecedent: Any,
        smt_env: Mapping[str, Any],
        expr_env: Mapping[str, Expr],
        canonical_index: Expr,
        side: str,
        rules: list[Rule],
        admissions: list[dict],
    ) -> None:
        self.context = context
        self.roles = roles
        self.prover = prover
        self.antecedent = antecedent
        self.smt_env = smt_env
        self.expr_env = expr_env
        self.canonical_index = canonical_index
        self.side = side
        self.rules = rules
        self.admissions = admissions
        self._rule_keys: set[tuple[str, str]] = set()
        self._next_rule = 0

    def _defined(self, term: SMTTerm, label: str) -> None:
        self.prover.require_unsat(
            "INDEX",
            "SYMBOLIC_INTEGER_DEFINEDNESS_NOT_PROVED",
            f"{self.side}:{label}:defined",
            z3.And(self.antecedent, z3.Not(_and(term.conditions))),
            f"integer definedness was not proved for {label}",
        )

    def _add_rule(
        self,
        category: str,
        lhs: Expr,
        rhs: Expr,
        validation: Mapping[str, Any],
        evidence: ProofLevel = ProofLevel.PARAMETRIC_SMT,
    ) -> None:
        lhs_pattern = pattern_from_expr(lhs)
        rhs_pattern = pattern_from_expr(rhs)
        key = (render_pattern(lhs_pattern), render_pattern(rhs_pattern))
        if key in self._rule_keys:
            return
        self._rule_keys.add(key)
        rule_id = f"parametric_{category}_{self.side}_{self._next_rule}"
        self._next_rule += 1
        rule = Rule(
            rule_id=rule_id,
            lhs=lhs_pattern,
            rhs=rhs_pattern,
            evidence=evidence,
            validator=(
                "pair_spec_role_mapping"
                if evidence == ProofLevel.TRUSTED_AXIOM
                else "pair_predicates_and_z3_unsat"
            ),
            statement=f"{lhs.render()} == {rhs.render()}",
            requires=("PairSpec role mapping", "symbolic output domain"),
            kind="relational",
            source="etv.parametric",
        )
        admission = rule.to_json()
        admission["status"] = "proved"
        admission["validation"] = dict(validation)
        self.rules.append(rule)
        self.admissions.append(admission)

    def value(self, expression: Expr) -> Expr:
        op = expression.op
        if op == "const_float":
            return expression
        if op == "scalar":
            logical = self.roles.scalars.get(str(expression.data))
            if logical is None:
                raise ParametricFailure(
                    "LOAD", "MISSING_ROLE", f"unmapped scalar {expression.data!r}", {}
                )
            physical = Expr(
                "input", data=f"{self.side}:{expression.data}", sort=Sort.FLOAT
            )
            canonical = Expr("input", data=logical, sort=Sort.FLOAT)
            self._add_rule(
                "scalar_role",
                physical,
                canonical,
                {
                    "result": "pair_fact",
                    "fact": {
                        "kind": "role_mapping",
                        "side": self.side,
                        "physical": str(expression.data),
                        "logical": logical,
                    },
                },
                evidence=ProofLevel.TRUSTED_AXIOM,
            )
            return physical
        if op == "load":
            offset = self.context.term(expression.args[0], self.smt_env)
            mask = self.context.term(expression.args[1], self.smt_env)
            self._defined(offset, "load-offset")
            self._defined(mask, "load-mask")
            mask_proof = self.prover.require_unsat(
                "LOAD",
                "SYMBOLIC_LOAD_MASK_UNSUPPORTED",
                f"{self.side}:load-mask-true",
                z3.And(self.antecedent, z3.Not(mask.value)),
                "a load mask is not true over the observed symbolic output domain",
            )
            mapping = self.roles.blocks.get(str(expression.data))
            if mapping is None:
                raise ParametricFailure(
                    "LOAD",
                    "MISSING_ROLE",
                    f"unmapped block {expression.data!r}",
                    {},
                )
            logical, endpoint = mapping
            physical = Expr(
                "load",
                args=(
                    _substitute(expression.args[0], self.expr_env),
                    _substitute(expression.args[1], self.expr_env),
                    self.value(expression.args[2]),
                ),
                data=f"{self.side}:{expression.data}",
                sort=Sort.FLOAT,
            )
            expected_offset = (
                z3.IntVal(endpoint.index)
                if endpoint.kind == "scalar_block"
                else self.context.term(self.canonical_index, self.smt_env).value
                - endpoint.offset
            )
            offset_proof = self.prover.require_unsat(
                "LOAD",
                (
                    "SCALAR_ABI_MISMATCH"
                    if endpoint.kind == "scalar_block"
                    else "LOAD_ADDRESS_RELATION_NOT_PROVED"
                ),
                f"{self.side}:load-address:{logical}",
                z3.And(self.antecedent, offset.value != expected_offset),
                f"{self.side} load from {logical} was not proved to use the "
                "canonical logical address",
            )
            canonical = (
                Expr("input", data=logical, sort=Sort.FLOAT)
                if endpoint.kind == "scalar_block"
                else Expr(
                    "read",
                    args=(self.canonical_index,),
                    data=logical,
                    sort=Sort.FLOAT,
                )
            )
            self._add_rule(
                "load",
                physical,
                canonical,
                {
                    "result": "unsat",
                    "logic": "symbolic integer arithmetic under PairSpec predicates",
                    "premises": {
                        "role_mapping": {
                            "side": self.side,
                            "physical": str(expression.data),
                            "logical": logical,
                            "endpoint_kind": endpoint.kind,
                        },
                        "mask_true": mask_proof,
                        "address_equal": offset_proof,
                    },
                },
            )
            return physical
        if op == "select":
            condition = self.context.term(expression.args[0], self.smt_env)
            self._defined(condition, "select-condition")
            true_branch = self.value(expression.args[1])
            false_branch = self.value(expression.args[2])
            physical = Expr(
                "select",
                args=(
                    _substitute(expression.args[0], self.expr_env),
                    true_branch,
                    false_branch,
                ),
                sort=expression.sort,
            )
            for expected, selected, bad in (
                (True, true_branch, z3.Not(condition.value)),
                (False, false_branch, condition.value),
            ):
                proof = self.prover.check_bad(
                    f"{self.side}:select-constant:{str(expected).lower()}",
                    z3.And(self.antecedent, bad),
                )
                if proof["result"] == "unsat":
                    self._add_rule(
                        "select",
                        physical,
                        selected,
                        {
                            "result": "unsat",
                            "condition_value": expected,
                            "proof": proof,
                        },
                    )
                    return physical
            raise ParametricFailure(
                "COMPUTE",
                "SYMBOLIC_SELECT_UNSUPPORTED",
                "a floating select condition varies over the symbolic output domain",
                {"expression": expression.render(), "side": self.side},
            )
        if op in {"fadd", "fsub", "fmul", "fdiv", "fneg", "fsqrt", "frsqrt", "fma"}:
            return Expr(
                op,
                args=tuple(self.value(arg) for arg in expression.args),
                sort=Sort.FLOAT,
            )
        raise ParametricFailure(
            "COMPUTE",
            "SYMBOLIC_VALUE_OPERATION_UNSUPPORTED",
            f"symbolic value operation {op!r} is unsupported",
            {"expression": expression.render(), "side": self.side},
        )


def _store_rule(
    side: str,
    physical_block: str,
    logical_role: str,
    offset: Expr,
    mask: Expr,
    canonical_index: Expr,
    offset_proof: Mapping[str, Any],
    mask_proof: Mapping[str, Any],
) -> tuple[Rule, dict]:
    value = var("stored_value")
    lhs = node(
        "observe_store",
        pattern_from_expr(offset),
        pattern_from_expr(mask),
        value,
        data=f"{side}:{physical_block}",
        match_data=True,
        sort=Sort.FLOAT,
    )
    rhs = node(
        "observe_store",
        pattern_from_expr(canonical_index),
        pattern_from_expr(bool_const(True)),
        value,
        data=logical_role,
        match_data=True,
        sort=Sort.FLOAT,
    )
    rule = Rule(
        rule_id=f"parametric_store_{side}",
        lhs=lhs,
        rhs=rhs,
        evidence=ProofLevel.PARAMETRIC_SMT,
        validator="pair_predicates_and_z3_unsat",
        statement=f"{render_pattern(lhs)} == {render_pattern(rhs)}",
        requires=("PairSpec output role mapping", "symbolic output domain"),
        kind="relational",
        source="etv.parametric",
    )
    admission = rule.to_json()
    admission["status"] = "proved"
    admission["validation"] = {
        "result": "unsat",
        "logic": "symbolic integer arithmetic under PairSpec predicates",
        "premises": {
            "role_mapping": {
                "side": side,
                "physical": physical_block,
                "logical": logical_role,
            },
            "mask_true": dict(mask_proof),
            "address_equal": dict(offset_proof),
        },
    }
    return rule, admission


def _proof_block(
    kind: str, summary: str, details: Optional[Mapping[str, Any]] = None
) -> dict:
    value = {
        "kind": kind,
        "status": Status.PROVED.value,
        "summary": summary,
        "proof_levels": [ProofLevel.PARAMETRIC_SMT.value],
        "required_for_final": True,
    }
    if details:
        value["details"] = dict(details)
    return value


def _leaf_signature(expression: Expr) -> tuple[str, ...]:
    leaves: list[str] = []

    def visit(current: Expr) -> None:
        if current.op in {"read", "input"}:
            leaves.append(current.render())
            return
        for argument in current.args:
            visit(argument)

    visit(expression)
    return tuple(sorted(leaves))


def verify_parametric_pair(
    lhs_program: Program, rhs_program: Program, spec: PairSpec
) -> ParametricResult:
    if len(lhs_program.stores) != 1 or len(rhs_program.stores) != 1:
        raise ParametricFailure(
            "INDEX",
            "MULTIPLE_STORES_UNSUPPORTED",
            "parametric verification requires exactly one store per side",
            {"lhs": len(lhs_program.stores), "rhs": len(rhs_program.stores)},
        )

    parameters = {name: z3.Int(name) for name in sorted(spec.facts.parameters)}
    shared = SMTContext(spec, parameters)
    base: list[Any] = []
    for name, declaration in sorted(spec.facts.parameters.items()):
        symbol = parameters[name]
        base.append(
            z3.And(symbol >= declaration.minimum, symbol <= declaration.maximum)
        )
    for constraint in spec.facts.constraints:
        encoded = shared.term(constraint, {})
        base.extend(encoded.conditions)
        base.append(encoded.value)

    prover = Prover(base, parameters, spec.limits.timeout_ms)
    consistency = prover.check_bad("parameter-domain-satisfiable", z3.BoolVal(True))
    if consistency["result"] != "sat":
        raise ParametricFailure(
            "PARAMETER_DOMAIN",
            "PARAMETER_DOMAIN_NOT_SATISFIABLE",
            "the declared symbolic parameter domain is not known to be satisfiable",
            consistency,
        )
    sample_model = consistency.get("counterexample", {})

    output_numel = shared.term(spec.contract.output_numel, {})
    prover.require_unsat(
        "PARAMETER_DOMAIN",
        "OUTPUT_NUMEL_NOT_POSITIVE",
        "output-numel-positive-and-defined",
        z3.Not(z3.And(_and(output_numel.conditions), output_numel.value > 0)),
        "contract.output_numel is not positive for every allowed parameter assignment",
    )

    k = z3.Int("logical_output_k")
    k_domain = z3.And(k >= 0, k < output_numel.value)
    side_data: dict[str, dict[str, Any]] = {}
    blocks: list[dict] = [
        _proof_block(
            "PARAMETER_DOMAIN",
            "the machine-readable shape constraints define a nonempty signed-i32 domain",
            {
                "parameters": {
                    name: {"min": item.minimum, "max": item.maximum}
                    for name, item in sorted(spec.facts.parameters.items())
                },
                "constraints": [
                    constraint.to_json() for constraint in spec.facts.constraints
                ],
                "sample_model": sample_model,
            },
        )
    ]

    for side, program in (("lhs", lhs_program), ("rhs", rhs_program)):
        context = SMTContext(spec, parameters, side)
        roles = build_side_roles(spec, side)
        store = program.stores[0]
        output_mapping = roles.blocks.get(store.block)
        if output_mapping is None or output_mapping[1].kind != "block":
            raise ParametricFailure(
                "ABI",
                "MISSING_ROLE",
                f"{side} store block is not mapped to a block role",
                {},
            )
        if output_mapping[0] != spec.contract.output_role:
            raise ParametricFailure(
                "ADDRESS",
                "OUTPUT_ROLE_MISMATCH",
                f"{side} writes logical role {output_mapping[0]!r}, not the contracted output",
                {"side": side, "role": output_mapping[0]},
                status=Status.DISPROVED,
            )

        programs = context.term(program.programs, {})
        prover.require_unsat(
            "INDEX",
            "SYMBOLIC_LAUNCH_NOT_PROVED",
            f"{side}:program-count-positive-and-defined",
            z3.Not(z3.And(_and(programs.conditions), programs.value > 0)),
            f"{side} launch count is not positive and defined for every parameter assignment",
        )

        pid = z3.Int(f"{side}_pid")
        lane = z3.Int(f"{side}_lane")
        env = {"pid": pid, "lane": lane}
        valid = z3.And(pid >= 0, pid < programs.value, lane >= 0, lane < program.lanes)
        logical = context.term(store.logical_index, env)
        mask = context.term(store.mask, env)
        offset = context.term(store.offset, env)
        active = z3.And(valid, mask.value)
        prover.require_unsat(
            "INDEX",
            "SYMBOLIC_INTEGER_DEFINEDNESS_NOT_PROVED",
            f"{side}:lane-index-defined",
            z3.And(valid, z3.Not(_and((*logical.conditions, *mask.conditions)))),
            f"{side} index or mask can be undefined in the symbolic launch domain",
            (pid, lane),
        )
        prover.require_unsat(
            "MASK",
            "PARAMETRIC_MASK_NOT_EXACT",
            f"{side}:active-index-in-contract",
            z3.And(
                active,
                z3.Not(z3.And(logical.value >= 0, logical.value < output_numel.value)),
            ),
            f"{side} can activate a logical index outside the contracted output domain",
            (pid, lane),
            status=Status.DISPROVED,
        )

        candidate_env = {
            "pid": k / program.lanes,
            "lane": k % program.lanes,
            "logical_output_k": k,
        }
        candidate_logical = context.term(store.logical_index, candidate_env)
        candidate_mask = context.term(store.mask, candidate_env)
        candidate_program_valid = z3.And(
            candidate_env["pid"] >= 0,
            candidate_env["pid"] < programs.value,
            candidate_env["lane"] >= 0,
            candidate_env["lane"] < program.lanes,
        )
        candidate_ok = z3.And(
            _and((*candidate_logical.conditions, *candidate_mask.conditions)),
            candidate_program_valid,
            candidate_mask.value,
            candidate_logical.value == k,
        )
        prover.require_unsat(
            "COVERAGE",
            "PARAMETRIC_COVERAGE_NOT_PROVED",
            f"{side}:canonical-writer-covers-every-k",
            z3.And(k_domain, z3.Not(candidate_ok)),
            f"{side} does not prove a canonical active writer for every output index",
            (k,),
        )

        pid2 = z3.Int(f"{side}_pid_2")
        lane2 = z3.Int(f"{side}_lane_2")
        env2 = {"pid": pid2, "lane": lane2}
        valid2 = z3.And(
            pid2 >= 0, pid2 < programs.value, lane2 >= 0, lane2 < program.lanes
        )
        logical2 = context.term(store.logical_index, env2)
        mask2 = context.term(store.mask, env2)
        offset2 = context.term(store.offset, env2)
        active2 = z3.And(valid2, mask2.value)
        distinct = z3.Or(pid != pid2, lane != lane2)
        prover.require_unsat(
            "ADDRESS",
            "SYMBOLIC_INTEGER_DEFINEDNESS_NOT_PROVED",
            f"{side}:active-output-address-defined",
            z3.And(active, z3.Not(_and(offset.conditions))),
            f"{side} has an undefined physical output address on an active lane",
            (pid, lane),
        )
        prover.require_unsat(
            "ADDRESS",
            "SYMBOLIC_INTEGER_DEFINEDNESS_NOT_PROVED",
            f"{side}:second-active-output-address-defined",
            z3.And(active2, z3.Not(_and(offset2.conditions))),
            f"{side} has an undefined physical output address on an active lane",
            (pid2, lane2),
        )
        prover.require_unsat(
            "INDEX",
            "DUPLICATE_LOGICAL_WRITE",
            f"{side}:logical-writer-unique",
            z3.And(
                active,
                active2,
                _and((*logical2.conditions, *mask2.conditions)),
                distinct,
                logical.value == logical2.value,
            ),
            f"{side} can assign one logical output from multiple lanes",
            (pid, lane, pid2, lane2),
        )
        prover.require_unsat(
            "RACE_FREEDOM",
            "RACE_FREEDOM_NOT_PROVED",
            f"{side}:physical-output-address-injective",
            z3.And(
                active,
                active2,
                _and((*offset.conditions, *offset2.conditions)),
                distinct,
                offset.value == offset2.value,
            ),
            f"{side} can write one physical output address from multiple lanes",
            (pid, lane, pid2, lane2),
        )

        candidate_offset = context.term(store.offset, candidate_env)
        prover.require_unsat(
            "ADDRESS",
            "SYMBOLIC_INTEGER_DEFINEDNESS_NOT_PROVED",
            f"{side}:candidate-output-address-defined",
            z3.And(k_domain, z3.Not(_and(candidate_offset.conditions))),
            f"{side} output address is not defined over the full symbolic output domain",
            (k,),
        )
        side_data[side] = {
            "program": program,
            "context": context,
            "roles": roles,
            "programs": programs.value,
            "candidate_env": candidate_env,
            "candidate_offset": candidate_offset.value + output_mapping[1].offset,
            "output_endpoint": output_mapping[1],
            "candidate_expr_env": {
                "pid": Expr(
                    "idiv",
                    args=(
                        Expr("var", data="logical_output_k", sort=Sort.INT),
                        int_const(program.lanes),
                    ),
                    sort=Sort.INT,
                ),
                "lane": Expr(
                    "irem",
                    args=(
                        Expr("var", data="logical_output_k", sort=Sort.INT),
                        int_const(program.lanes),
                    ),
                    sort=Sort.INT,
                ),
            },
        }

    prover.require_unsat(
        "ADDRESS",
        "OUTPUT_ADDRESS_MISMATCH",
        "lhs-rhs-output-address-equality",
        z3.And(
            k_domain,
            side_data["lhs"]["candidate_offset"]
            != side_data["rhs"]["candidate_offset"],
        ),
        "corresponding symbolic output elements can write different physical offsets",
        (k,),
        status=Status.DISPROVED,
    )

    canonical_index = Expr("var", data="logical_output_k", sort=Sort.INT)
    rewrite_rules: list[Rule] = []
    rewrite_admission: list[dict] = []
    roots: dict[str, Expr] = {}
    for side in ("lhs", "rhs"):
        item = side_data[side]
        store = item["program"].stores[0]
        value_rewriter = SymbolicValueRewriter(
            item["context"],
            item["roles"],
            prover,
            k_domain,
            item["candidate_env"],
            item["candidate_expr_env"],
            canonical_index,
            side,
            rewrite_rules,
            rewrite_admission,
        )
        physical_value = value_rewriter.value(store.value)
        physical_offset = _substitute(store.offset, item["candidate_expr_env"])
        if item["output_endpoint"].offset:
            physical_offset = Expr(
                "iadd",
                args=(physical_offset, int_const(item["output_endpoint"].offset)),
                sort=Sort.INT,
            )
        physical_mask = _substitute(store.mask, item["candidate_expr_env"])
        mask_term = item["context"].term(store.mask, item["candidate_env"])
        store_mask_proof = prover.require_unsat(
            "MASK",
            "PARAMETRIC_MASK_NOT_EXACT",
            f"{side}:canonical-store-mask-true",
            z3.And(k_domain, z3.Not(mask_term.value)),
            f"{side} canonical store is not active over the full output domain",
            (k,),
            status=Status.DISPROVED,
        )
        canonical_term = item["context"].term(canonical_index, item["candidate_env"])
        store_offset_proof = prover.require_unsat(
            "ADDRESS",
            "OUTPUT_ADDRESS_MISMATCH",
            f"{side}:canonical-store-address",
            z3.And(k_domain, item["candidate_offset"] != canonical_term.value),
            f"{side} output address does not rewrite to the canonical logical address",
            (k,),
            status=Status.DISPROVED,
        )
        physical_root = Expr(
            "observe_store",
            args=(physical_offset, physical_mask, physical_value),
            data=f"{side}:{store.block}",
            sort=Sort.FLOAT,
        )
        roots[side] = physical_root
        rule, admission = _store_rule(
            side,
            store.block,
            spec.contract.output_role,
            physical_offset,
            physical_mask,
            canonical_index,
            store_offset_proof,
            store_mask_proof,
        )
        rewrite_rules.append(rule)
        rewrite_admission.append(admission)

    blocks.extend(
        (
            _proof_block(
                "INDEX",
                "symbolic launch indices have a unique canonical writer for every output element",
            ),
            _proof_block(
                "MASK",
                "both active symbolic domains are exactly the contracted output interval",
            ),
            _proof_block(
                "COVERAGE",
                "every symbolic output index is covered exactly once on both sides",
                {"output_numel": spec.contract.output_numel.to_json()},
            ),
            _proof_block(
                "RACE_FREEDOM",
                "symbolic active store addresses are injective on both sides",
            ),
            _proof_block(
                "ADDRESS",
                "corresponding symbolic output elements write equal physical offsets",
            ),
        )
    )
    load_rewrites = [
        item for item in rewrite_admission if item["id"].startswith("parametric_load_")
    ]
    if load_rewrites:
        blocks.append(
            _proof_block(
                "LOAD",
                "physical loads are retained until predicate-derived, SMT-conditioned "
                "rewrites map them to logical reads",
                {
                    "rewrite_rule_ids": [item["id"] for item in load_rewrites],
                    "decision_boundary": (
                        "load equivalence is accepted only after e-graph rule "
                        "application"
                    ),
                },
            )
        )
    else:
        blocks.append(
            {
                "kind": "LOAD",
                "status": Status.UNKNOWN.value,
                "summary": (
                    "symbolic input dependency frontiers differ; compute proof will "
                    "decide observability"
                ),
                "reason": "LOAD_FRONTIER_MISMATCH",
                "proof_levels": [],
                "required_for_final": False,
                "details": {
                    "lhs": list(_leaf_signature(roots["lhs"])),
                    "rhs": list(_leaf_signature(roots["rhs"])),
                },
            }
        )

    return ParametricResult(
        blocks=tuple(blocks),
        proof={
            "parameters": {
                name: {"min": item.minimum, "max": item.maximum}
                for name, item in sorted(spec.facts.parameters.items())
            },
            "constraints": [
                constraint.to_json() for constraint in spec.facts.constraints
            ],
            "sample_model": sample_model,
            "output_numel": spec.contract.output_numel.to_json(),
            "checks": prover.checks,
            "complete_for_parameter_domain": True,
            "rewrite_targets": {
                "lhs": roots["lhs"].render(),
                "rhs": roots["rhs"].render(),
            },
            "fact_derived_rewrites": [
                {
                    "id": item["id"],
                    "statement": item["statement"],
                    "evidence": item["evidence"],
                    "validation": item["validation"],
                }
                for item in rewrite_admission
            ],
        },
        root_pairs=(("logical_output_k", roots["lhs"], roots["rhs"]),),
        rewrite_rules=tuple(rewrite_rules),
        rewrite_admission=tuple(rewrite_admission),
    )
