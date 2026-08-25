"""Strict JSON readers for Semantic TTIR programs and pair specifications."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .model import (
    Contract,
    Expr,
    FactContext,
    InputError,
    Limits,
    PairSpec,
    Program,
    RoleEndpoint,
    RolePair,
    Sort,
    StoreTemplate,
    bool_const,
    float_const,
    int_const,
)


FORMAT_PROGRAM = "etv-semantic-program-v1"
FORMAT_PAIR = "etv-pair-v1"

_ARITY = {
    "iadd": 2,
    "isub": 2,
    "imul": 2,
    "idiv": 2,
    "irem": 2,
    "ceildiv": 2,
    "lt": 2,
    "le": 2,
    "gt": 2,
    "ge": 2,
    "eq": 2,
    "ne": 2,
    "and": 2,
    "or": 2,
    "not": 1,
    "fadd": 2,
    "fsub": 2,
    "fmul": 2,
    "fdiv": 2,
    "fneg": 1,
    "fsqrt": 1,
    "frsqrt": 1,
    "fma": 3,
    "select": 3,
}

_SORT = {
    **{name: Sort.INT for name in ("iadd", "isub", "imul", "idiv", "irem", "ceildiv")},
    **{name: Sort.BOOL for name in ("lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not")},
    **{name: Sort.FLOAT for name in ("fadd", "fsub", "fmul", "fdiv", "fneg", "fsqrt", "frsqrt", "fma")},
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}", "READ_ERROR") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}", "PARSE_ERROR") from exc
    if not isinstance(value, dict):
        raise InputError(f"top-level value in {path} must be an object")
    return value


def _only_keys(value: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise InputError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def parse_expr(value: Any, where: str = "expression") -> Expr:
    if isinstance(value, bool):
        return bool_const(value)
    if isinstance(value, int):
        return int_const(value)
    if isinstance(value, float):
        return float_const(Fraction(str(value)))
    if not isinstance(value, dict):
        raise InputError(f"{where} must be a literal or expression object")

    if set(value) == {"var"}:
        name = value["var"]
        if not isinstance(name, str) or not name:
            raise InputError(f"{where}.var must be a non-empty string")
        return Expr("var", data=name, sort=Sort.INT)
    if set(value) == {"scalar"}:
        name = value["scalar"]
        if not isinstance(name, str) or not name:
            raise InputError(f"{where}.scalar must be a non-empty string")
        return Expr("scalar", data=name, sort=Sort.FLOAT)
    if set(value) == {"float"}:
        raw = value["float"]
        try:
            parsed = Fraction(str(raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise InputError(f"{where}.float is not an exact rational literal") from exc
        return float_const(parsed)

    op = value.get("op")
    if op == "load":
        _only_keys(value, {"op", "block", "offset", "mask", "default"}, where)
        block = value.get("block")
        if not isinstance(block, str) or not block:
            raise InputError(f"{where}.block must be a non-empty string")
        missing = [key for key in ("offset", "mask", "default") if key not in value]
        if missing:
            raise InputError(f"missing key(s) in {where}: {', '.join(missing)}")
        return Expr(
            "load",
            args=(
                parse_expr(value["offset"], f"{where}.offset"),
                parse_expr(value["mask"], f"{where}.mask"),
                parse_expr(value["default"], f"{where}.default"),
            ),
            data=block,
            sort=Sort.FLOAT,
        )

    _only_keys(value, {"op", "args"}, where)
    if op not in _ARITY:
        raise InputError(f"unsupported expression op {op!r} in {where}")
    args = value.get("args")
    if not isinstance(args, list) or len(args) != _ARITY[op]:
        raise InputError(f"{where}.{op} expects {_ARITY[op]} argument(s)")
    parsed_args = tuple(parse_expr(arg, f"{where}.{op}[{index}]") for index, arg in enumerate(args))
    if op == "select":
        result_sort = parsed_args[1].sort
        if parsed_args[2].sort != result_sort:
            raise InputError(f"{where}.select branches have different sorts")
    else:
        result_sort = _SORT[op]
    return Expr(op, args=parsed_args, sort=result_sort)


def load_program(path: Path) -> Program:
    path = path.resolve()
    raw = _read_json(path)
    _only_keys(raw, {"format", "name", "launch", "stores"}, str(path))
    if raw.get("format") != FORMAT_PROGRAM:
        raise InputError(f"{path} must declare format {FORMAT_PROGRAM!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise InputError(f"{path}.name must be a non-empty string")
    launch = raw.get("launch")
    if not isinstance(launch, dict):
        raise InputError(f"{path}.launch must be an object")
    _only_keys(launch, {"programs", "lanes"}, f"{path}.launch")
    if "programs" not in launch:
        raise InputError(f"{path}.launch.programs is required")
    lanes = launch.get("lanes")
    if not isinstance(lanes, int) or isinstance(lanes, bool) or lanes <= 0:
        raise InputError(f"{path}.launch.lanes must be a positive integer")

    stores_raw = raw.get("stores")
    if not isinstance(stores_raw, list) or not stores_raw:
        raise InputError(f"{path}.stores must be a non-empty list")
    stores = []
    for index, item in enumerate(stores_raw):
        where = f"{path}.stores[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(item, {"block", "logical_index", "offset", "mask", "value"}, where)
        missing = [key for key in ("block", "logical_index", "offset", "mask", "value") if key not in item]
        if missing:
            raise InputError(f"missing key(s) in {where}: {', '.join(missing)}")
        block = item["block"]
        if not isinstance(block, str) or not block:
            raise InputError(f"{where}.block must be a non-empty string")
        stores.append(
            StoreTemplate(
                block=block,
                logical_index=parse_expr(item["logical_index"], f"{where}.logical_index"),
                offset=parse_expr(item["offset"], f"{where}.offset"),
                mask=parse_expr(item["mask"], f"{where}.mask"),
                value=parse_expr(item["value"], f"{where}.value"),
            )
        )
    return Program(
        name=name,
        source=path,
        programs=parse_expr(launch["programs"], f"{path}.launch.programs"),
        lanes=lanes,
        stores=tuple(stores),
    )


def _endpoint(value: Any, where: str) -> RoleEndpoint:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    _only_keys(value, {"kind", "name", "index"}, where)
    kind = value.get("kind")
    name = value.get("name")
    if kind not in {"block", "scalar", "scalar_block"}:
        raise InputError(f"{where}.kind must be block, scalar, or scalar_block")
    if not isinstance(name, str) or not name:
        raise InputError(f"{where}.name must be a non-empty string")
    index = value.get("index", 0)
    if not isinstance(index, int) or isinstance(index, bool):
        raise InputError(f"{where}.index must be an integer")
    if kind != "scalar_block" and "index" in value:
        raise InputError(f"{where}.index is only valid for scalar_block")
    return RoleEndpoint(kind=kind, name=name, index=index)


def _strings(value: Any, where: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise InputError(f"{where} must be a list of non-empty strings")
    return tuple(value)


def load_pair_spec(path: Path) -> PairSpec:
    path = path.resolve()
    raw = _read_json(path)
    _only_keys(
        raw,
        {"format", "pair_id", "lhs", "rhs", "semantic_mode", "roles", "facts", "contract", "limits"},
        str(path),
    )
    if raw.get("format") != FORMAT_PAIR:
        raise InputError(f"{path} must declare format {FORMAT_PAIR!r}")
    pair_id = raw.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise InputError(f"{path}.pair_id must be a non-empty string")
    lhs = raw.get("lhs")
    rhs = raw.get("rhs")
    if not isinstance(lhs, str) or not isinstance(rhs, str):
        raise InputError(f"{path}.lhs and .rhs must be relative path strings")
    semantic_mode = raw.get("semantic_mode")
    if not isinstance(semantic_mode, str):
        raise InputError(f"{path}.semantic_mode must be a string")

    roles_raw = raw.get("roles")
    if not isinstance(roles_raw, dict) or not roles_raw:
        raise InputError(f"{path}.roles must be a non-empty object")
    roles = []
    seen_lhs = set()
    seen_rhs = set()
    for logical in sorted(roles_raw):
        item = roles_raw[logical]
        where = f"{path}.roles.{logical}"
        if not isinstance(item, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(item, {"lhs", "rhs"}, where)
        if "lhs" not in item or "rhs" not in item:
            raise InputError(f"{where} requires lhs and rhs endpoints")
        lhs_endpoint = _endpoint(item["lhs"], f"{where}.lhs")
        rhs_endpoint = _endpoint(item["rhs"], f"{where}.rhs")
        lhs_key = (lhs_endpoint.kind, lhs_endpoint.name)
        rhs_key = (rhs_endpoint.kind, rhs_endpoint.name)
        if lhs_key in seen_lhs or rhs_key in seen_rhs:
            raise InputError(f"physical role endpoint is mapped more than once in {where}")
        seen_lhs.add(lhs_key)
        seen_rhs.add(rhs_key)
        roles.append(RolePair(logical=logical, lhs=lhs_endpoint, rhs=rhs_endpoint))

    facts_raw = raw.get("facts")
    if not isinstance(facts_raw, dict):
        raise InputError(f"{path}.facts must be an object")
    _only_keys(facts_raw, {"bindings", "assumptions", "disjoint"}, f"{path}.facts")
    bindings_raw = facts_raw.get("bindings", {})
    if not isinstance(bindings_raw, dict):
        raise InputError(f"{path}.facts.bindings must be an object")
    bindings: Dict[str, int] = {}
    for key, value in bindings_raw.items():
        if not isinstance(key, str) or not key:
            raise InputError(f"fact binding names must be non-empty strings")
        if not isinstance(value, int) or isinstance(value, bool):
            raise InputError(f"fact binding {key!r} must be an integer")
        bindings[key] = value
    assumptions = _strings(facts_raw.get("assumptions", []), f"{path}.facts.assumptions")
    disjoint_raw = facts_raw.get("disjoint", [])
    if not isinstance(disjoint_raw, list):
        raise InputError(f"{path}.facts.disjoint must be a list of role lists")
    disjoint_groups = []
    for index, group in enumerate(disjoint_raw):
        names = _strings(group, f"{path}.facts.disjoint[{index}]")
        if len(names) < 2:
            raise InputError(f"{path}.facts.disjoint[{index}] needs at least two roles")
        disjoint_groups.append(frozenset(names))

    contract_raw = raw.get("contract")
    if not isinstance(contract_raw, dict):
        raise InputError(f"{path}.contract must be an object")
    _only_keys(
        contract_raw,
        {"output_role", "output_numel", "require_full_coverage", "require_disjoint"},
        f"{path}.contract",
    )
    output_role = contract_raw.get("output_role")
    if not isinstance(output_role, str) or output_role not in roles_raw:
        raise InputError(f"{path}.contract.output_role must name a mapped role")
    require_full_coverage = contract_raw.get("require_full_coverage", True)
    if not isinstance(require_full_coverage, bool):
        raise InputError(f"{path}.contract.require_full_coverage must be boolean")
    require_disjoint = _strings(
        contract_raw.get("require_disjoint", []),
        f"{path}.contract.require_disjoint",
    )
    unknown_roles = sorted(set(require_disjoint) - set(roles_raw))
    if unknown_roles:
        raise InputError(f"unknown role(s) in require_disjoint: {', '.join(unknown_roles)}")
    if "output_numel" not in contract_raw:
        raise InputError(f"{path}.contract.output_numel is required")

    limits_raw = raw.get("limits", {})
    if not isinstance(limits_raw, dict):
        raise InputError(f"{path}.limits must be an object")
    _only_keys(limits_raw, {"max_iterations", "max_enodes", "timeout_ms"}, f"{path}.limits")
    defaults = Limits()
    limit_values = {}
    for key in ("max_iterations", "max_enodes", "timeout_ms"):
        value = limits_raw.get(key, getattr(defaults, key))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(f"{path}.limits.{key} must be a positive integer")
        limit_values[key] = value

    base = path.parent
    return PairSpec(
        pair_id=pair_id,
        source=path,
        lhs_path=(base / lhs).resolve(),
        rhs_path=(base / rhs).resolve(),
        semantic_mode=semantic_mode,
        roles=tuple(roles),
        facts=FactContext(
            bindings=bindings,
            assumptions=assumptions,
            disjoint_groups=tuple(disjoint_groups),
        ),
        contract=Contract(
            output_role=output_role,
            output_numel=parse_expr(contract_raw["output_numel"], f"{path}.contract.output_numel"),
            require_full_coverage=require_full_coverage,
            require_disjoint=require_disjoint,
        ),
        limits=Limits(**limit_values),
    )
