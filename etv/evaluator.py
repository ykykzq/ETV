"""Finite-domain symbolic evaluator for ETV IR."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Mapping, Tuple, Union

from .casts import INTEGER_CAST_OPS, apply_integer_cast
from .ir import Expr, Program, Sort, bool_const, float_const, int_const
from .model import (
    Evaluation,
    FactContext,
    InputError,
    LaneRecord,
    PairSpec,
    RoleEndpoint,
    UnsupportedSemantics,
)

Concrete = Union[int, bool, Expr]
I32_MIN = -(2**31)
I32_MAX = 2**31 - 1


@dataclass(frozen=True)
class SideRoles:
    blocks: Mapping[str, Tuple[str, RoleEndpoint]]
    scalars: Mapping[str, str]


def build_side_roles(spec: PairSpec, side: str) -> SideRoles:
    blocks: Dict[str, Tuple[str, RoleEndpoint]] = {}
    scalars: Dict[str, str] = {}
    for role in spec.roles:
        endpoint = role.lhs if side == "lhs" else role.rhs
        if endpoint.kind in {"block", "scalar_block"}:
            blocks[endpoint.name] = (role.logical, endpoint)
        else:
            scalars[endpoint.name] = role.logical
    return SideRoles(blocks=blocks, scalars=scalars)


def _as_int(value: Concrete, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{where} requires an integer operand", "TYPE_ERROR")
    return value


def _as_bool(value: Concrete, where: str) -> bool:
    if not isinstance(value, bool):
        raise InputError(f"{where} requires a boolean operand", "TYPE_ERROR")
    return value


def _as_float(value: Concrete, where: str) -> Expr:
    if not isinstance(value, Expr) or value.sort != Sort.FLOAT:
        raise InputError(f"{where} requires an abstract-float operand", "TYPE_ERROR")
    return value


def _i32(value: int, where: str) -> int:
    if value < I32_MIN or value > I32_MAX:
        raise UnsupportedSemantics(
            f"i32 no-overflow obligation failed in {where}: {value}",
            "INTEGER_OVERFLOW",
        )
    return value


def _trunc_div(lhs: int, rhs: int, where: str) -> int:
    if rhs == 0:
        raise UnsupportedSemantics(
            f"integer division by zero in {where}", "UNDEFINED_DIVISION"
        )
    if lhs == I32_MIN and rhs == -1:
        raise UnsupportedSemantics(
            f"i32 division overflow in {where}", "INTEGER_OVERFLOW"
        )
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) != (rhs < 0) else quotient


def eval_expr(
    expr: Expr,
    env: Mapping[str, int],
    facts: FactContext,
    roles: SideRoles,
) -> Concrete:
    op = expr.op
    if op == "const_int":
        return int(expr.data)
    if op == "const_bool":
        return bool(expr.data)
    if op == "const_float":
        return expr
    if op == "var":
        if expr.data in env:
            return _i32(int(env[expr.data]), f"variable {expr.data}")
        if expr.data in facts.bindings:
            binding = facts.bindings[expr.data]
            if isinstance(binding, Expr):
                return eval_expr(binding, env, facts, roles)
            return _i32(int(binding), f"fact {expr.data}")
        raise InputError(f"unbound integer variable {expr.data!r}", "UNBOUND_VARIABLE")
    if op == "scalar":
        logical = roles.scalars.get(expr.data)
        if logical is None:
            raise InputError(f"unmapped scalar {expr.data!r}", "MISSING_ROLE")
        return Expr("input", data=logical, sort=Sort.FLOAT)

    if op == "select":
        condition = _as_bool(eval_expr(expr.args[0], env, facts, roles), "select")
        branch = expr.args[1] if condition else expr.args[2]
        return eval_expr(branch, env, facts, roles)

    if op == "load":
        offset = _as_int(eval_expr(expr.args[0], env, facts, roles), "load offset")
        mask = _as_bool(eval_expr(expr.args[1], env, facts, roles), "load mask")
        if not mask:
            return _as_float(eval_expr(expr.args[2], env, facts, roles), "load default")
        mapping = roles.blocks.get(expr.data)
        if mapping is None:
            raise InputError(f"unmapped memory block {expr.data!r}", "MISSING_ROLE")
        logical, endpoint = mapping
        if endpoint.kind == "scalar_block":
            if offset != endpoint.index:
                raise UnsupportedSemantics(
                    f"scalar block {expr.data!r} read at {offset}, expected {endpoint.index}",
                    "SCALAR_ABI_MISMATCH",
                )
            return Expr("input", data=logical, sort=Sort.FLOAT)
        return Expr("read", args=(int_const(offset),), data=logical, sort=Sort.FLOAT)

    if op in INTEGER_CAST_OPS:
        value = eval_expr(expr.args[0], env, facts, roles)
        if isinstance(value, Expr):
            raise UnsupportedSemantics(
                f"symbolic operand for integer cast {op!r} is unsupported",
                "SYMBOLIC_INTEGER_CAST_UNSUPPORTED",
            )
        try:
            result = apply_integer_cast(op, value, expr.data)
        except ValueError as exc:
            raise UnsupportedSemantics(
                f"cannot evaluate integer cast {expr.render()}: {exc}",
                "INTEGER_CAST_UNSUPPORTED",
            ) from exc
        return result if isinstance(result, bool) else _i32(result, op)

    if op in {"and", "or"}:
        lhs = _as_bool(eval_expr(expr.args[0], env, facts, roles), op)
        rhs = _as_bool(eval_expr(expr.args[1], env, facts, roles), op)
        return lhs and rhs if op == "and" else lhs or rhs
    if op == "not":
        return not _as_bool(eval_expr(expr.args[0], env, facts, roles), op)
    if op in {"lt", "le", "gt", "ge", "eq", "ne"}:
        lhs = eval_expr(expr.args[0], env, facts, roles)
        rhs = eval_expr(expr.args[1], env, facts, roles)
        if isinstance(lhs, Expr) or isinstance(rhs, Expr):
            raise UnsupportedSemantics(
                f"symbolic comparison {op} is outside the MVP",
                "SYMBOLIC_MASK_UNSUPPORTED",
            )
        if op == "lt":
            return lhs < rhs
        if op == "le":
            return lhs <= rhs
        if op == "gt":
            return lhs > rhs
        if op == "ge":
            return lhs >= rhs
        if op == "eq":
            return lhs == rhs
        return lhs != rhs

    if op in {"iadd", "isub", "imul", "idiv", "irem", "ceildiv"}:
        lhs = _as_int(eval_expr(expr.args[0], env, facts, roles), op)
        rhs = _as_int(eval_expr(expr.args[1], env, facts, roles), op)
        if op == "iadd":
            result = lhs + rhs
        elif op == "isub":
            result = lhs - rhs
        elif op == "imul":
            result = lhs * rhs
        elif op == "idiv":
            result = _trunc_div(lhs, rhs, op)
        elif op == "irem":
            result = lhs - _trunc_div(lhs, rhs, op) * rhs
        else:
            if lhs < 0 or rhs <= 0:
                raise UnsupportedSemantics(
                    "ceildiv currently requires lhs >= 0 and rhs > 0",
                    "CEILDIV_DOMAIN_UNSUPPORTED",
                )
            result = (lhs + rhs - 1) // rhs
        return _i32(result, op)

    if op in {"fadd", "fsub", "fmul", "fdiv", "fneg", "fsqrt", "frsqrt", "fma"}:
        args = tuple(
            _as_float(eval_expr(arg, env, facts, roles), op) for arg in expr.args
        )
        return Expr(op, args=args, sort=Sort.FLOAT)

    raise UnsupportedSemantics(f"cannot evaluate operation {op!r}")


def evaluate_program(program: Program, spec: PairSpec, side: str) -> Evaluation:
    if len(program.stores) != 1:
        raise UnsupportedSemantics(
            f"{program.name} has {len(program.stores)} stores; MVP requires exactly one",
            "MULTIPLE_STORES_UNSUPPORTED",
        )
    roles = build_side_roles(spec, side)
    facts = spec.facts.for_side(side)
    count_value = eval_expr(program.programs, {}, facts, roles)
    program_count = _as_int(count_value, "launch.programs")
    if program_count <= 0:
        raise InputError("launch.programs must evaluate to a positive integer")
    if program_count * program.lanes > 1_000_000:
        raise UnsupportedSemantics(
            "finite launch domain exceeds the MVP limit of 1,000,000 lanes",
            "FINITE_DOMAIN_TOO_LARGE",
        )

    store = program.stores[0]
    output_mapping = roles.blocks.get(store.block)
    if output_mapping is None:
        raise InputError(f"unmapped store block {store.block!r}", "MISSING_ROLE")
    output_role, endpoint = output_mapping
    if endpoint.kind != "block":
        raise InputError("store target must map to a block role", "ROLE_KIND_MISMATCH")

    records = []
    for pid in range(program_count):
        for lane in range(program.lanes):
            env = {"pid": pid, "lane": lane}
            logical_index = _as_int(
                eval_expr(store.logical_index, env, facts, roles),
                "store.logical_index",
            )
            active = _as_bool(eval_expr(store.mask, env, facts, roles), "store.mask")
            if active:
                offset = _as_int(
                    eval_expr(store.offset, env, facts, roles), "store.offset"
                )
                value = _as_float(
                    eval_expr(store.value, env, facts, roles), "store.value"
                )
            else:
                offset = None
                value = None
            records.append(
                LaneRecord(
                    pid=pid,
                    lane=lane,
                    logical_index=logical_index,
                    active=active,
                    output_role=output_role,
                    offset=offset,
                    value=value,
                )
            )
    return Evaluation(
        program=program, program_count=program_count, records=tuple(records)
    )
