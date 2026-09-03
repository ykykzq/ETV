"""Strict readers for TTIR PairSpecs and proof-core test fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .casts import INTEGER_CAST_OPS, normalize_cast_data
from .ir import (
    Expr,
    Program,
    Sort,
    StoreTemplate,
    bool_const,
    float_const,
    int_const,
)
from .model import (
    Contract,
    FrontendSpec,
    InputError,
    LaunchSpec,
    LLMConfig,
    Limits,
    Parameter,
    PairSpec,
    PartitionConfig,
    PredicateDeclaration,
    PredicateSet,
    ProofLevel,
    RewriteSource,
    RulePolicy,
    RoleEndpoint,
    RolePair,
)
from .rules import (
    Pattern,
    PredicateRequirement,
    Rule,
    builtin_rules,
    node,
    number,
    pattern_variables,
    render_pattern,
    var,
)

FORMAT_PROGRAM = "etv-semantic-program-v1"
FORMAT_PAIR = "etv-pair-v1"
FORMAT_PAIR_V2 = "etv-pair-v2"
FORMAT_REWRITE = "etv-rewrite-v1"

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
    **{name: 1 for name in INTEGER_CAST_OPS},
}

_SORT = {
    **{
        name: Sort.INT
        for name in (
            "iadd",
            "isub",
            "imul",
            "idiv",
            "irem",
            "ceildiv",
            *INTEGER_CAST_OPS,
        )
    },
    **{
        name: Sort.BOOL
        for name in ("lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not")
    },
    **{
        name: Sort.FLOAT
        for name in ("fadd", "fsub", "fmul", "fdiv", "fneg", "fsqrt", "frsqrt", "fma")
    },
}

_RULE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SYMBOLIC_PREDICATE_OPS = {
    "const_int",
    "const_bool",
    "var",
    "iadd",
    "isub",
    "imul",
    "idiv",
    "irem",
    "ceildiv",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "and",
    "or",
    "not",
    "select",
    *INTEGER_CAST_OPS,
}


def _read_json_document(path: Path) -> Tuple[Mapping[str, Any], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}", "READ_ERROR") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"invalid UTF-8 in {path}: {exc}", "PARSE_ERROR") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}", "PARSE_ERROR") from exc
    if not isinstance(value, dict):
        raise InputError(f"top-level value in {path} must be an object")
    return value, content


def _read_json(path: Path) -> Mapping[str, Any]:
    return _read_json_document(path)[0]


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
        offset = parse_expr(value["offset"], f"{where}.offset")
        mask = parse_expr(value["mask"], f"{where}.mask")
        default = parse_expr(value["default"], f"{where}.default")
        if offset.sort != Sort.INT:
            raise InputError(f"{where}.offset must be an integer expression")
        if mask.sort != Sort.BOOL:
            raise InputError(f"{where}.mask must be a boolean expression")
        if default.sort != Sort.FLOAT:
            raise InputError(f"{where}.default must be abstract_float")
        return Expr("load", args=(offset, mask, default), data=block, sort=Sort.FLOAT)

    _only_keys(
        value,
        {"op", "args", "data"} if op in INTEGER_CAST_OPS else {"op", "args"},
        where,
    )
    if op not in _ARITY:
        raise InputError(f"unsupported expression op {op!r} in {where}")
    args = value.get("args")
    if not isinstance(args, list) or len(args) != _ARITY[op]:
        raise InputError(f"{where}.{op} expects {_ARITY[op]} argument(s)")
    parsed_args = tuple(
        parse_expr(arg, f"{where}.{op}[{index}]") for index, arg in enumerate(args)
    )
    cast_data = None
    if op in INTEGER_CAST_OPS:
        try:
            cast_data = normalize_cast_data(value.get("data"))
        except ValueError as exc:
            raise InputError(f"{where}.{op} has invalid cast data: {exc}") from exc
        source_type, result_type = cast_data
        operand_sort = Sort.BOOL if source_type == "i1" else Sort.INT
        if parsed_args[0].sort != operand_sort:
            raise InputError(
                f"{where}.{op} operand sort does not match source type {source_type}"
            )
        result_sort = Sort.BOOL if result_type == "i1" else Sort.INT
    elif op == "select":
        if parsed_args[0].sort != Sort.BOOL:
            raise InputError(f"{where}.select condition must be boolean")
        result_sort = parsed_args[1].sort
        if parsed_args[2].sort != result_sort:
            raise InputError(f"{where}.select branches have different sorts")
    elif op in {"iadd", "isub", "imul", "idiv", "irem", "ceildiv"}:
        if any(argument.sort != Sort.INT for argument in parsed_args):
            raise InputError(f"{where}.{op} operands must be integers")
        result_sort = Sort.INT
    elif op in {"and", "or", "not"}:
        if any(argument.sort != Sort.BOOL for argument in parsed_args):
            raise InputError(f"{where}.{op} operands must be boolean")
        result_sort = Sort.BOOL
    elif op in {"lt", "le", "gt", "ge"}:
        if (
            parsed_args[0].sort != parsed_args[1].sort
            or parsed_args[0].sort == Sort.BOOL
        ):
            raise InputError(f"{where}.{op} operands must have the same numeric sort")
        result_sort = Sort.BOOL
    elif op in {"eq", "ne"}:
        if parsed_args[0].sort != parsed_args[1].sort:
            raise InputError(f"{where}.{op} operands must have the same sort")
        result_sort = Sort.BOOL
    elif op in {"fadd", "fsub", "fmul", "fdiv", "fneg", "fsqrt", "frsqrt", "fma"}:
        if any(argument.sort != Sort.FLOAT for argument in parsed_args):
            raise InputError(f"{where}.{op} operands must be abstract_float")
        result_sort = Sort.FLOAT
    else:
        result_sort = _SORT[op]
    return Expr(op, args=parsed_args, data=cast_data, sort=result_sort)


def _validate_symbolic_predicate(expression: Expr, where: str) -> None:
    def operations(node: Expr) -> set[str]:
        result = {node.op}
        for argument in node.args:
            result.update(operations(argument))
        return result

    unsupported = sorted(operations(expression) - _SYMBOLIC_PREDICATE_OPS)
    if unsupported:
        raise InputError(
            f"{where} uses unsupported symbolic predicate operation(s): {', '.join(unsupported)}"
        )


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
        missing = [
            key
            for key in ("block", "logical_index", "offset", "mask", "value")
            if key not in item
        ]
        if missing:
            raise InputError(f"missing key(s) in {where}: {', '.join(missing)}")
        block = item["block"]
        if not isinstance(block, str) or not block:
            raise InputError(f"{where}.block must be a non-empty string")
        stores.append(
            StoreTemplate(
                block=block,
                logical_index=parse_expr(
                    item["logical_index"], f"{where}.logical_index"
                ),
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
        frontend="semantic_fixture",
        frontend_version="1",
    )


def _endpoint(value: Any, where: str) -> RoleEndpoint:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    _only_keys(value, {"kind", "name", "index", "offset"}, where)
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
    offset = value.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise InputError(f"{where}.offset must be an integer")
    if kind != "block" and "offset" in value:
        raise InputError(f"{where}.offset is only valid for block")
    return RoleEndpoint(kind=kind, name=name, index=index, offset=offset)


def _strings(value: Any, where: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise InputError(f"{where} must be a list of non-empty strings")
    return tuple(value)


def _rewrite_pattern(value: Any, where: str) -> Pattern:
    if isinstance(value, bool):
        return node("const_bool", data=value, match_data=True, sort=Sort.BOOL)
    if isinstance(value, int):
        return node("const_int", data=value, match_data=True, sort=Sort.INT)
    if isinstance(value, float):
        fraction = Fraction(str(value))
        return number(fraction.numerator, fraction.denominator)
    if not isinstance(value, dict):
        raise InputError(f"{where} must be a pattern object or literal")

    if set(value) == {"match"}:
        name = value["match"]
        if not isinstance(name, str) or not _RULE_ID.fullmatch(name):
            raise InputError(f"{where}.match must be an identifier")
        return var(name)
    if set(value) == {"float"}:
        try:
            fraction = Fraction(str(value["float"]))
        except (ValueError, ZeroDivisionError) as exc:
            raise InputError(f"{where}.float is not an exact rational literal") from exc
        return number(fraction.numerator, fraction.denominator)
    if set(value) == {"input"}:
        role = value["input"]
        if not isinstance(role, str) or not role:
            raise InputError(f"{where}.input must be a non-empty logical role")
        return node("input", data=role, match_data=True)
    if set(value) == {"read"}:
        read = value["read"]
        if not isinstance(read, dict):
            raise InputError(f"{where}.read must be an object")
        _only_keys(read, {"role", "offset"}, f"{where}.read")
        role = read.get("role")
        if not isinstance(role, str) or not role or "offset" not in read:
            raise InputError(f"{where}.read requires a role and offset")
        return node(
            "read",
            _rewrite_pattern(read["offset"], f"{where}.read.offset"),
            data=role,
            match_data=True,
        )

    _only_keys(value, {"op", "args", "data", "sort"}, where)
    op = value.get("op")
    if op not in _ARITY:
        raise InputError(f"unsupported rewrite operation {op!r} in {where}")
    args = value.get("args")
    if not isinstance(args, list) or len(args) != _ARITY[op]:
        raise InputError(f"{where}.{op} expects {_ARITY[op]} argument(s)")
    sort_raw = value.get("sort")
    cast_data = None
    if op in INTEGER_CAST_OPS:
        try:
            cast_data = normalize_cast_data(value.get("data"))
        except ValueError as exc:
            raise InputError(f"{where}.{op} has invalid cast data: {exc}") from exc
        inferred_sort = Sort.BOOL if cast_data[1] == "i1" else Sort.INT
    else:
        inferred_sort = Sort.FLOAT if op == "select" else _SORT[op]
    if sort_raw is None:
        sort = inferred_sort
    else:
        try:
            sort = Sort(sort_raw)
        except ValueError as exc:
            raise InputError(
                f"{where}.sort must be int, bool, or abstract_float"
            ) from exc
        if op in INTEGER_CAST_OPS and sort != inferred_sort:
            raise InputError(
                f"{where}.sort does not match integer cast result type {cast_data[1]}"
            )
    return node(
        op,
        *(
            _rewrite_pattern(arg, f"{where}.{op}[{index}]")
            for index, arg in enumerate(args)
        ),
        data=cast_data if op in INTEGER_CAST_OPS else value.get("data"),
        match_data="data" in value,
        sort=sort,
    )


def _rule_requirement(value: Any, where: str) -> PredicateRequirement:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    kind = value.get("kind")
    if kind == "binding_equals":
        _only_keys(value, {"kind", "name", "value"}, where)
        name = value.get("name")
        expected = value.get("value")
        if not isinstance(name, str) or not name:
            raise InputError(f"{where}.name must be a non-empty binding name")
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise InputError(f"{where}.value must be an integer")
        return PredicateRequirement(kind=kind, name=name, value=expected)
    if kind == "assumption":
        _only_keys(value, {"kind", "text"}, where)
        text = value.get("text")
        if not isinstance(text, str) or not text:
            raise InputError(f"{where}.text must be a non-empty assumption")
        return PredicateRequirement(kind=kind, value=text)
    if kind == "predicate":
        _only_keys(value, {"kind", "id"}, where)
        predicate_id = value.get("id")
        if not isinstance(predicate_id, str) or not _RULE_ID.fullmatch(predicate_id):
            raise InputError(f"{where}.id must be a stable predicate identifier")
        return PredicateRequirement(kind=kind, value=predicate_id)
    if kind == "disjoint":
        _only_keys(value, {"kind", "roles"}, where)
        roles = _strings(value.get("roles"), f"{where}.roles")
        if len(roles) < 2:
            raise InputError(f"{where}.roles needs at least two roles")
        return PredicateRequirement(kind=kind, roles=roles)
    if kind == "constraint":
        _only_keys(value, {"kind", "expression"}, where)
        if "expression" not in value:
            raise InputError(f"{where}.expression is required")
        expression = parse_expr(value["expression"], f"{where}.expression")
        if expression.sort != Sort.BOOL:
            raise InputError(f"{where}.expression must be boolean")
        return PredicateRequirement(kind=kind, expression=expression)
    raise InputError(
        f"{where}.kind must be binding_equals, assumption, predicate, disjoint, or constraint"
    )


def _rewrite_rule(value: Any, where: str) -> Rule:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    _only_keys(
        value,
        {"id", "kind", "lhs", "rhs", "statement", "requires", "provenance"},
        where,
    )
    rule_id = value.get("id")
    if not isinstance(rule_id, str) or not _RULE_ID.fullmatch(rule_id):
        raise InputError(f"{where}.id must be a stable rule identifier")
    if rule_id.startswith("parametric_"):
        raise InputError(f"{where}.id uses the reserved 'parametric_' prefix")
    kind = value.get("kind")
    if kind not in {"algebraic", "trusted_fact", "trusted_predicate"}:
        raise InputError(
            f"{where}.kind must be algebraic, trusted_predicate, or legacy trusted_fact"
        )
    if "lhs" not in value or "rhs" not in value:
        raise InputError(f"{where} requires lhs and rhs patterns")
    lhs = _rewrite_pattern(value["lhs"], f"{where}.lhs")
    rhs = _rewrite_pattern(value["rhs"], f"{where}.rhs")
    if lhs.variable is not None:
        raise InputError(f"{where}.lhs cannot be a bare match variable")
    unbound = sorted(pattern_variables(rhs) - pattern_variables(lhs))
    if unbound:
        raise InputError(
            f"{where}.rhs has unbound match variable(s): {', '.join(unbound)}"
        )

    requirements_raw = value.get("requires", [])
    if not isinstance(requirements_raw, list):
        raise InputError(f"{where}.requires must be a list")
    requirements = tuple(
        _rule_requirement(item, f"{where}.requires[{index}]")
        for index, item in enumerate(requirements_raw)
    )
    if kind in {"trusted_fact", "trusted_predicate"} and not requirements:
        raise InputError(
            f"{where} trusted predicate rules require at least one formal predicate gate"
        )
    provenance_raw = value.get("provenance", {"generated_by": "human"})
    if not isinstance(provenance_raw, dict):
        raise InputError(f"{where}.provenance must be an object")
    _only_keys(
        provenance_raw,
        {"generated_by", "generator", "prompt_sha256"},
        f"{where}.provenance",
    )
    generated_by = provenance_raw.get("generated_by", "human")
    if generated_by not in {"human", "llm"}:
        raise InputError(f"{where}.provenance.generated_by must be human or llm")
    generator = provenance_raw.get("generator")
    prompt_sha256 = provenance_raw.get("prompt_sha256")
    if generated_by == "llm":
        if not isinstance(generator, str) or not generator:
            raise InputError(f"{where}.provenance.generator is required for LLM rules")
        if not isinstance(prompt_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", prompt_sha256
        ):
            raise InputError(
                f"{where}.provenance.prompt_sha256 must be a lowercase SHA-256"
            )
    elif generator is not None or prompt_sha256 is not None:
        raise InputError(
            f"{where}.provenance generator metadata is only valid for LLM rules"
        )

    statement = value.get(
        "statement", f"{render_pattern(lhs)} == {render_pattern(rhs)}"
    )
    if not isinstance(statement, str) or not statement:
        raise InputError(f"{where}.statement must be a non-empty string")
    return Rule(
        rule_id=rule_id,
        lhs=lhs,
        rhs=rhs,
        evidence=(
            ProofLevel.ALGEBRAIC if kind == "algebraic" else ProofLevel.TRUSTED_AXIOM
        ),
        validator=(
            "z3_real_unsat" if kind == "algebraic" else "trusted_pair_predicate"
        ),
        statement=statement,
        requires=("semantic_mode == abstract_float",),
        kind=kind,
        source=where,
        predicate_requirements=requirements,
        generated_by=generated_by,
        generator=generator,
        prompt_sha256=prompt_sha256,
    )


def parse_rewrite_rule(value: Any, where: str = "rewrite_rule") -> Rule:
    """Parse one rule declaration, including declarations returned by an LLM."""

    return _rewrite_rule(value, where)


def _validate_rewrite_rule_ids(rules: Sequence[Rule]) -> None:
    rule_ids = [rule.rule_id for rule in rules]
    duplicate_rule_ids = sorted(
        {rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1}
    )
    builtin_rule_ids = {rule.rule_id for rule in builtin_rules()}
    conflicting_rule_ids = sorted(set(rule_ids) & builtin_rule_ids)
    if duplicate_rule_ids:
        raise InputError(
            f"duplicate rewrite rule id(s): {', '.join(duplicate_rule_ids)}"
        )
    if conflicting_rule_ids:
        raise InputError(
            f"rewrite rule id conflicts with builtin rule(s): {', '.join(conflicting_rule_ids)}"
        )


def _parse_custom_predicates(raw: Any, where: str) -> Tuple[PredicateDeclaration, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InputError(f"{where} must be a list")
    declarations = []
    seen = set()
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{item_where} must be an object")
        _only_keys(item, {"id", "formula", "kind", "encoder", "status"}, item_where)
        predicate_id = item.get("id")
        if not isinstance(predicate_id, str) or not _RULE_ID.fullmatch(predicate_id):
            raise InputError(f"{item_where}.id must be a stable predicate identifier")
        if predicate_id in seen:
            raise InputError(f"duplicate predicate id {predicate_id!r}")
        seen.add(predicate_id)
        kind = item.get("kind", "custom")
        if kind not in {"custom", "shape", "relation", "abi"}:
            raise InputError(f"{item_where}.kind is not supported")
        encoder = item.get("encoder", "trusted")
        if encoder not in {"trusted", "z3_expr"}:
            raise InputError(f"{item_where}.encoder must be trusted or z3_expr")
        status = item.get("status", "assumed")
        if status != "assumed":
            raise InputError(
                f"{item_where}.status must be assumed; PairSpec declarations cannot self-assert a proof"
            )
        formula = None
        if "formula" in item:
            formula = parse_expr(item["formula"], f"{item_where}.formula")
            if formula.sort != Sort.BOOL:
                raise InputError(f"{item_where}.formula must be boolean")
        if encoder == "z3_expr" and formula is None:
            raise InputError(f"{item_where}.formula is required for z3_expr predicates")
        if encoder == "z3_expr" and formula is not None:
            _validate_symbolic_predicate(formula, f"{item_where}.formula")
        declarations.append(
            PredicateDeclaration(
                predicate_id=predicate_id,
                formula=formula,
                kind=kind,
                encoder=encoder,
                status=status,
            )
        )
    return tuple(declarations)


def _load_rewrite_registry(
    path: Path, raw: Any
) -> Tuple[Tuple[Rule, ...], Tuple[RewriteSource, ...]]:
    if not isinstance(raw, list):
        raise InputError(f"{path}.rewrites must be a list")
    rules = []
    sources = []
    for index, item in enumerate(raw):
        where = f"{path}.rewrites[{index}]"
        if isinstance(item, str):
            relative = item
        elif isinstance(item, dict):
            _only_keys(item, {"file"}, where)
            relative = item.get("file")
        else:
            raise InputError(f"{where} must be a file path or object with file")
        if not isinstance(relative, str) or not relative:
            raise InputError(f"{where}.file must be a non-empty relative path")
        if Path(relative).is_absolute():
            raise InputError(f"{where}.file must be relative to the PairSpec")
        source_path = (path.parent / relative).resolve()
        if source_path == path or source_path.is_dir():
            raise InputError(f"{where}.file must reference a rewrite file")
        payload, content = _read_json_document(source_path)
        _only_keys(payload, {"format", "rules"}, str(source_path))
        if payload.get("format") != FORMAT_REWRITE:
            raise InputError(f"{source_path} must declare format {FORMAT_REWRITE!r}")
        rules_raw = payload.get("rules")
        if not isinstance(rules_raw, list):
            raise InputError(f"{source_path}.rules must be a list")
        rules.extend(
            replace(
                _rewrite_rule(item, f"{source_path}.rules[{rule_index}]"),
                source=f"user:{source_path}",
            )
            for rule_index, item in enumerate(rules_raw)
        )
        digest = hashlib.sha256(content).hexdigest()
        sources.append(RewriteSource(kind="user", path=source_path, sha256=digest))
    _validate_rewrite_rule_ids(rules)
    return tuple(rules), tuple(sources)


def _load_pair_spec_v2(
    path: Path, raw: Mapping[str, Any], require_ttir: bool
) -> PairSpec:
    _only_keys(
        raw,
        {"format", "metadata", "assumptions", "predicates", "observation", "rewrites"},
        str(path),
    )
    if raw.get("format") != FORMAT_PAIR_V2:
        raise InputError(f"{path} must declare format {FORMAT_PAIR_V2!r}")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise InputError(f"{path}.metadata must be an object")
    _only_keys(
        metadata,
        {
            "pair_id",
            "lhs",
            "rhs",
            "frontends",
            "semantic_mode",
            "limits",
            "rule_policy",
            "llm",
            "partition",
            "launches",
        },
        f"{path}.metadata",
    )
    assumptions_raw = raw.get("assumptions", {"for_llm": []})
    if isinstance(assumptions_raw, list):
        assumptions = _strings(assumptions_raw, f"{path}.assumptions")
    elif isinstance(assumptions_raw, dict):
        _only_keys(assumptions_raw, {"for_llm"}, f"{path}.assumptions")
        assumptions = _strings(
            assumptions_raw.get("for_llm", []), f"{path}.assumptions.for_llm"
        )
    else:
        raise InputError(f"{path}.assumptions must be an object with for_llm")

    predicates_raw = raw.get("predicates")
    if not isinstance(predicates_raw, dict):
        raise InputError(f"{path}.predicates must be an object")
    _only_keys(
        predicates_raw,
        {
            "abi",
            "bindings",
            "side_bindings",
            "parameters",
            "constraints",
            "disjoint",
            "custom",
        },
        f"{path}.predicates",
    )
    abi = predicates_raw.get("abi")
    if not isinstance(abi, dict) or not abi:
        raise InputError(f"{path}.predicates.abi must be a non-empty object")
    custom = _parse_custom_predicates(
        predicates_raw.get("custom", []), f"{path}.predicates.custom"
    )
    facts_raw = {
        "bindings": predicates_raw.get("bindings", {}),
        "side_bindings": predicates_raw.get("side_bindings", {}),
        "parameters": predicates_raw.get("parameters", {}),
        "constraints": predicates_raw.get("constraints", []),
        "assumptions": [],
        "disjoint": predicates_raw.get("disjoint", []),
    }
    observation = raw.get("observation")
    if not isinstance(observation, dict):
        raise InputError(f"{path}.observation must be an object")
    # The proof core calls this section Contract; the public v2 spelling is
    # observation to make the distinction from assumptions explicit.
    launches_raw = metadata.get("launches", {})
    if not isinstance(launches_raw, dict):
        raise InputError(f"{path}.metadata.launches must be an object")
    _only_keys(launches_raw, {"lhs", "rhs"}, f"{path}.metadata.launches")

    def launch_fallback(side: str, key: str) -> Any:
        values = launches_raw.get(side)
        if not isinstance(values, list) or not values:
            return None
        first = values[0]
        return first.get(key) if isinstance(first, dict) else None

    frontends = metadata.get("frontends", {})
    if not isinstance(frontends, dict):
        raise InputError(f"{path}.metadata.frontends must be an object")
    normalized_frontends = dict(frontends)
    for side in ("lhs", "rhs"):
        if side not in normalized_frontends:
            fallback = launch_fallback(side, "frontend")
            if fallback is not None:
                normalized_frontends[side] = fallback

    normalized = {
        "format": FORMAT_PAIR,
        "pair_id": metadata.get("pair_id"),
        "lhs": metadata.get("lhs", launch_fallback("lhs", "file")),
        "rhs": metadata.get("rhs", launch_fallback("rhs", "file")),
        "frontends": normalized_frontends,
        "semantic_mode": metadata.get("semantic_mode"),
        "roles": abi,
        "facts": facts_raw,
        "contract": observation,
        "limits": metadata.get("limits", {}),
        "rule_policy": metadata.get("rule_policy", {}),
        "llm": metadata.get("llm", {}),
        "partition": metadata.get("partition", {}),
    }
    rewrite_rules, rewrite_sources = _load_rewrite_registry(
        path, raw.get("rewrites", [])
    )
    normalized["rewrite_rules"] = []
    spec = _load_pair_spec(path, require_ttir, _raw=normalized)
    launch_sequences = _parse_launch_sequences(
        path, launches_raw, require_ttir=require_ttir
    )
    predicate_set = PredicateSet(
        abi=spec.roles,
        bindings=spec.predicates.bindings,
        side_bindings=spec.predicates.side_bindings,
        parameters=spec.predicates.parameters,
        constraints=spec.predicates.constraints,
        disjoint_groups=spec.predicates.disjoint_groups,
        formal_assumptions=(),
        custom=custom,
    )
    custom_ids = {declaration.predicate_id for declaration in custom}
    generated_ids = set(replace(predicate_set, custom=()).predicate_ids)
    conflicts = sorted(custom_ids & generated_ids)
    if conflicts:
        raise InputError(
            f"custom predicate id conflicts with generated predicate id(s): {', '.join(conflicts)}"
        )
    return replace(
        spec,
        predicates=predicate_set,
        assumptions=assumptions,
        rewrite_rules=rewrite_rules,
        rewrite_sources=rewrite_sources,
        launch_sequences=launch_sequences,
    )


def _parse_launch_sequences(
    path: Path,
    raw: Mapping[str, Any],
    *,
    require_ttir: bool,
) -> Mapping[str, Tuple[LaunchSpec, ...]]:
    sequences: Dict[str, Tuple[LaunchSpec, ...]] = {}
    for side in ("lhs", "rhs"):
        values = raw.get(side)
        if values is None:
            continue
        where = f"{path}.metadata.launches.{side}"
        if not isinstance(values, list) or len(values) < 2:
            raise InputError(f"{where} must contain at least two ordered launches")
        launches: list[LaunchSpec] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(values):
            item_where = f"{where}[{index}]"
            if not isinstance(item, dict):
                raise InputError(f"{item_where} must be an object")
            _only_keys(
                item,
                {
                    "id",
                    "file",
                    "frontend",
                    "semantic",
                    "abi",
                    "bindings",
                    "step",
                },
                item_where,
            )
            launch_id = item.get("id")
            if not isinstance(launch_id, str) or not _RULE_ID.fullmatch(launch_id):
                raise InputError(f"{item_where}.id must be a stable identifier")
            if launch_id in seen_ids:
                raise InputError(f"duplicate launch id {launch_id!r} in {where}")
            seen_ids.add(launch_id)
            relative = item.get("file")
            if not isinstance(relative, str) or not relative:
                raise InputError(f"{item_where}.file must be a relative path string")
            artifact = (path.parent / relative).resolve()
            expected_suffixes = {".ttir", ".mlir"} if require_ttir else {".json"}
            if artifact.suffix not in expected_suffixes:
                expected = ".ttir/.mlir" if require_ttir else ".json"
                raise InputError(
                    f"{item_where}.file must reference {expected}",
                    "TTIR_PAIR_REQUIRED" if require_ttir else "INVALID_INPUT",
                )
            if not artifact.is_file():
                raise InputError(
                    f"{item_where}.file does not exist: {artifact}", "READ_ERROR"
                )

            frontend_raw = item.get("frontend")
            if not isinstance(frontend_raw, dict):
                raise InputError(f"{item_where}.frontend must be an object")
            _only_keys(
                frontend_raw,
                {"kind", "function", "programs", "store_index"},
                f"{item_where}.frontend",
            )
            expected_kind = "ttir" if require_ttir else "semantic_json"
            kind = frontend_raw.get("kind")
            if kind != expected_kind:
                raise InputError(
                    f"{item_where}.frontend.kind must be {expected_kind}",
                    "TTIR_PAIR_REQUIRED" if require_ttir else "INVALID_INPUT",
                )
            function = frontend_raw.get("function")
            programs_raw = frontend_raw.get("programs")
            store_index = frontend_raw.get("store_index")
            if require_ttir:
                if not isinstance(function, str) or not function:
                    raise InputError(
                        f"{item_where}.frontend.function must name the TTIR entry"
                    )
                if programs_raw is None:
                    raise InputError(
                        f"{item_where}.frontend.programs is required because TTIR "
                        "does not encode the host launch grid"
                    )
            elif any(
                value is not None for value in (function, programs_raw, store_index)
            ):
                raise InputError(
                    f"{item_where}.frontend semantic fixtures accept only kind"
                )
            if store_index is not None and (
                not isinstance(store_index, int)
                or isinstance(store_index, bool)
                or store_index < 0
            ):
                raise InputError(
                    f"{item_where}.frontend.store_index must be non-negative"
                )
            programs = (
                None
                if programs_raw is None
                else parse_expr(programs_raw, f"{item_where}.frontend.programs")
            )
            if programs is not None and programs.sort != Sort.INT:
                raise InputError(
                    f"{item_where}.frontend.programs must be an integer expression"
                )

            semantic = item.get("semantic", launch_id)
            if not isinstance(semantic, str) or not semantic.strip():
                raise InputError(f"{item_where}.semantic must be a non-empty string")
            step = item.get("step", index)
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise InputError(f"{item_where}.step must be a non-negative integer")
            if launches and step < launches[-1].step:
                raise InputError(f"{where} steps must be nondecreasing")
            abi_raw = item.get("abi")
            if not isinstance(abi_raw, dict) or not abi_raw:
                raise InputError(f"{item_where}.abi must be a non-empty object")
            roles: Dict[str, RoleEndpoint] = {}
            endpoints: set[tuple[str, str]] = set()
            for logical, endpoint_raw in sorted(abi_raw.items()):
                if not isinstance(logical, str) or not logical:
                    raise InputError(f"{item_where}.abi role names must be non-empty")
                endpoint = _endpoint(endpoint_raw, f"{item_where}.abi.{logical}")
                key = (endpoint.kind, endpoint.name)
                if key in endpoints:
                    raise InputError(
                        f"physical endpoint is mapped more than once in {item_where}.abi"
                    )
                endpoints.add(key)
                roles[logical] = endpoint

            bindings_raw = item.get("bindings", {})
            if not isinstance(bindings_raw, dict):
                raise InputError(f"{item_where}.bindings must be an object")
            bindings: Dict[str, int | Expr] = {}
            for name, value in bindings_raw.items():
                if not isinstance(name, str) or not name:
                    raise InputError(
                        f"{item_where}.bindings names must be non-empty strings"
                    )
                if isinstance(value, int) and not isinstance(value, bool):
                    bindings[name] = value
                else:
                    expression = parse_expr(value, f"{item_where}.bindings.{name}")
                    if expression.sort != Sort.INT:
                        raise InputError(
                            f"{item_where}.bindings.{name} must be an integer expression"
                        )
                    bindings[name] = expression

            launches.append(
                LaunchSpec(
                    launch_id=launch_id,
                    path=artifact,
                    frontend=FrontendSpec(
                        kind=kind,
                        function=function,
                        programs=programs,
                        store_index=store_index,
                    ),
                    semantic=semantic.strip(),
                    roles=roles,
                    bindings=bindings,
                    step=step,
                )
            )
        sequences[side] = tuple(launches)
    return sequences


def _load_pair_spec(
    path: Path, require_ttir: bool, _raw: Mapping[str, Any] | None = None
) -> PairSpec:
    path = path.resolve()
    raw = _read_json(path) if _raw is None else _raw
    if raw.get("format") == FORMAT_PAIR_V2:
        return _load_pair_spec_v2(path, raw, require_ttir)
    _only_keys(
        raw,
        {
            "format",
            "pair_id",
            "lhs",
            "rhs",
            "frontends",
            "semantic_mode",
            "roles",
            "facts",
            "contract",
            "limits",
            "rewrite_rules",
            "rule_policy",
            "llm",
            "partition",
        },
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
            raise InputError(
                f"physical role endpoint is mapped more than once in {where}"
            )
        seen_lhs.add(lhs_key)
        seen_rhs.add(rhs_key)
        roles.append(RolePair(logical=logical, lhs=lhs_endpoint, rhs=rhs_endpoint))

    facts_raw = raw.get("facts")
    if not isinstance(facts_raw, dict):
        raise InputError(f"{path}.facts must be an object")
    _only_keys(
        facts_raw,
        {
            "bindings",
            "side_bindings",
            "parameters",
            "constraints",
            "assumptions",
            "disjoint",
        },
        f"{path}.facts",
    )
    bindings_raw = facts_raw.get("bindings", {})
    if not isinstance(bindings_raw, dict):
        raise InputError(f"{path}.facts.bindings must be an object")
    bindings: Dict[str, int | Expr] = {}
    for key, value in bindings_raw.items():
        if not isinstance(key, str) or not key:
            raise InputError(f"fact binding names must be non-empty strings")
        if isinstance(value, int) and not isinstance(value, bool):
            bindings[key] = value
        else:
            expression = parse_expr(value, f"{path}.facts.bindings.{key}")
            if expression.sort != Sort.INT:
                raise InputError(f"fact binding {key!r} must be an integer expression")
            bindings[key] = expression

    parameters_raw = facts_raw.get("parameters", {})
    if not isinstance(parameters_raw, dict):
        raise InputError(f"{path}.facts.parameters must be an object")
    parameters: Dict[str, Parameter] = {}
    for key, value in parameters_raw.items():
        where = f"{path}.facts.parameters.{key}"
        if not isinstance(key, str) or not key:
            raise InputError(f"parameter names must be non-empty strings")
        if key in bindings:
            raise InputError(f"{where} duplicates a concrete fact binding")
        if not isinstance(value, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(value, {"min", "max"}, where)
        minimum = value.get("min", -(2**31))
        maximum = value.get("max", 2**31 - 1)
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in (minimum, maximum)
        ):
            raise InputError(f"{where}.min and .max must be integers")
        if minimum < -(2**31) or maximum > 2**31 - 1 or minimum > maximum:
            raise InputError(f"{where} must describe a nonempty signed-i32 interval")
        parameters[key] = Parameter(minimum=minimum, maximum=maximum)

    constraints_raw = facts_raw.get("constraints", [])
    if not isinstance(constraints_raw, list):
        raise InputError(f"{path}.facts.constraints must be a list")
    constraints = tuple(
        parse_expr(item, f"{path}.facts.constraints[{index}]")
        for index, item in enumerate(constraints_raw)
    )
    if any(expression.sort != Sort.BOOL for expression in constraints):
        raise InputError(f"{path}.facts.constraints must contain boolean expressions")
    for index, expression in enumerate(constraints):
        _validate_symbolic_predicate(expression, f"{path}.facts.constraints[{index}]")
    side_bindings_raw = facts_raw.get("side_bindings", {})
    if not isinstance(side_bindings_raw, dict):
        raise InputError(f"{path}.facts.side_bindings must be an object")
    _only_keys(side_bindings_raw, {"lhs", "rhs"}, f"{path}.facts.side_bindings")
    side_bindings: Dict[str, Dict[str, int | Expr]] = {}
    for side, values in side_bindings_raw.items():
        where = f"{path}.facts.side_bindings.{side}"
        if not isinstance(values, dict):
            raise InputError(f"{where} must be an object")
        parsed: Dict[str, int | Expr] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise InputError(f"{where} binding names must be non-empty strings")
            if key in bindings:
                raise InputError(f"{where}.{key} duplicates a shared fact binding")
            if isinstance(value, int) and not isinstance(value, bool):
                parsed[key] = value
            else:
                expression = parse_expr(value, f"{where}.{key}")
                if expression.sort != Sort.INT:
                    raise InputError(f"{where}.{key} must be an integer expression")
                parsed[key] = expression
        side_bindings[side] = parsed
    assumptions = _strings(
        facts_raw.get("assumptions", []), f"{path}.facts.assumptions"
    )
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
        raise InputError(
            f"unknown role(s) in require_disjoint: {', '.join(unknown_roles)}"
        )
    if "output_numel" not in contract_raw:
        raise InputError(f"{path}.contract.output_numel is required")

    limits_raw = raw.get("limits", {})
    if not isinstance(limits_raw, dict):
        raise InputError(f"{path}.limits must be an object")
    _only_keys(
        limits_raw, {"max_iterations", "max_enodes", "timeout_ms"}, f"{path}.limits"
    )
    defaults = Limits()
    limit_values = {}
    for key in ("max_iterations", "max_enodes", "timeout_ms"):
        value = limits_raw.get(key, getattr(defaults, key))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(f"{path}.limits.{key} must be a positive integer")
        limit_values[key] = value

    rule_policy_raw = raw.get("rule_policy", {})
    if not isinstance(rule_policy_raw, dict):
        raise InputError(f"{path}.rule_policy must be an object")
    _only_keys(
        rule_policy_raw,
        {"algebraic_validation", "non_algebraic_validation"},
        f"{path}.rule_policy",
    )
    algebraic_validation = rule_policy_raw.get("algebraic_validation", "required")
    if algebraic_validation not in {"required", "best_effort", "trusted"}:
        raise InputError(
            f"{path}.rule_policy.algebraic_validation must be required, best_effort, or trusted"
        )
    non_algebraic_validation = rule_policy_raw.get(
        "non_algebraic_validation", "trusted"
    )
    if non_algebraic_validation != "trusted":
        raise InputError(
            f"{path}.rule_policy.non_algebraic_validation currently supports only trusted"
        )

    llm_raw = raw.get("llm", {})
    if not isinstance(llm_raw, dict):
        raise InputError(f"{path}.llm must be an object")
    _only_keys(
        llm_raw,
        {
            "enabled",
            "provider",
            "model",
            "base_url",
            "select_nodes",
            "generate_rules",
            "max_candidates",
            "timeout_ms",
        },
        f"{path}.llm",
    )
    llm_defaults = LLMConfig()
    llm_values = {
        key: llm_raw.get(key, getattr(llm_defaults, key))
        for key in (
            "enabled",
            "provider",
            "model",
            "base_url",
            "select_nodes",
            "generate_rules",
            "max_candidates",
            "timeout_ms",
        )
    }
    for key in ("enabled", "select_nodes", "generate_rules"):
        if not isinstance(llm_values[key], bool):
            raise InputError(f"{path}.llm.{key} must be boolean")
    for key in ("provider", "model", "base_url"):
        if not isinstance(llm_values[key], str) or not llm_values[key]:
            raise InputError(f"{path}.llm.{key} must be a non-empty string")
    if llm_values["provider"] != "deepseek":
        raise InputError(f"{path}.llm.provider currently supports only deepseek")
    if llm_values["base_url"].rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise InputError(
            f"{path}.llm.base_url must use the official DeepSeek API endpoint"
        )
    for key in ("max_candidates", "timeout_ms"):
        if (
            not isinstance(llm_values[key], int)
            or isinstance(llm_values[key], bool)
            or llm_values[key] <= 0
        ):
            raise InputError(f"{path}.llm.{key} must be a positive integer")

    partition_raw = raw.get("partition", {})
    if not isinstance(partition_raw, dict):
        raise InputError(f"{path}.partition must be an object")
    _only_keys(
        partition_raw,
        {"enabled", "min_partitions", "max_partitions"},
        f"{path}.partition",
    )
    partition_defaults = PartitionConfig()
    partition_values = {
        key: partition_raw.get(key, getattr(partition_defaults, key))
        for key in ("enabled", "min_partitions", "max_partitions")
    }
    if not isinstance(partition_values["enabled"], bool):
        raise InputError(f"{path}.partition.enabled must be boolean")
    for key in ("min_partitions", "max_partitions"):
        value = partition_values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(f"{path}.partition.{key} must be a positive integer")
    if partition_values["min_partitions"] > partition_values["max_partitions"]:
        raise InputError(
            f"{path}.partition.min_partitions cannot exceed max_partitions"
        )
    if partition_values["enabled"] and not llm_values["enabled"]:
        raise InputError(
            f"{path}.partition.enabled requires llm.enabled so the whole-program scan is explicit"
        )

    base = path.parent
    frontends_raw = raw.get("frontends", {})
    if not isinstance(frontends_raw, dict):
        raise InputError(f"{path}.frontends must be an object")
    _only_keys(frontends_raw, {"lhs", "rhs"}, f"{path}.frontends")

    def parse_frontend(side: str) -> FrontendSpec:
        if require_ttir and side not in frontends_raw:
            raise InputError(
                f"{path}.frontends.{side} must explicitly declare a TTIR frontend",
                "TTIR_PAIR_REQUIRED",
            )
        value = frontends_raw.get(side, {})
        where = f"{path}.frontends.{side}"
        if not isinstance(value, dict):
            raise InputError(f"{where} must be an object")
        _only_keys(value, {"kind", "function", "programs", "store_index"}, where)
        if require_ttir and "kind" not in value:
            raise InputError(
                f"{where}.kind must explicitly be ttir", "TTIR_PAIR_REQUIRED"
            )
        kind = value.get("kind", "ttir" if require_ttir else "semantic_json")
        allowed = {"ttir"} if require_ttir else {"semantic_json"}
        if kind not in allowed:
            expected = "ttir" if require_ttir else "semantic_json"
            raise InputError(f"{where}.kind must be {expected}", "TTIR_PAIR_REQUIRED")
        function = value.get("function")
        programs = value.get("programs")
        store_index = value.get("store_index")
        if store_index is not None and (
            not isinstance(store_index, int)
            or isinstance(store_index, bool)
            or store_index < 0
        ):
            raise InputError(f"{where}.store_index must be a non-negative integer")
        if kind == "ttir":
            if not isinstance(function, str) or not function:
                raise InputError(f"{where}.function must name the TTIR entry function")
            if programs is None:
                raise InputError(
                    f"{where}.programs is required because TTIR does not encode the host launch grid"
                )
        elif function is not None or programs is not None or store_index is not None:
            raise InputError(
                f"{where} internal Semantic IR fixtures do not accept function or programs"
            )
        parsed_programs = (
            None if programs is None else parse_expr(programs, f"{where}.programs")
        )
        if parsed_programs is not None and parsed_programs.sort != Sort.INT:
            raise InputError(f"{where}.programs must be an integer expression")
        return FrontendSpec(
            kind=kind,
            function=function,
            programs=parsed_programs,
            store_index=store_index,
        )

    rewrite_rules_raw = raw.get("rewrite_rules", [])
    if not isinstance(rewrite_rules_raw, list):
        raise InputError(f"{path}.rewrite_rules must be a list")
    rewrite_rules = tuple(
        _rewrite_rule(item, f"{path}.rewrite_rules[{index}]")
        for index, item in enumerate(rewrite_rules_raw)
    )
    _validate_rewrite_rule_ids(rewrite_rules)

    lhs_path = (base / lhs).resolve()
    rhs_path = (base / rhs).resolve()
    if require_ttir:
        for side, artifact in (("lhs", lhs_path), ("rhs", rhs_path)):
            if artifact.suffix not in {".ttir", ".mlir"}:
                raise InputError(
                    f"{path}.{side} must reference a raw .ttir or .mlir file",
                    "TTIR_PAIR_REQUIRED",
                )
            if not artifact.is_file():
                raise InputError(
                    f"{path}.{side} does not exist: {artifact}", "READ_ERROR"
                )

    predicate_set = PredicateSet(
        abi=tuple(roles),
        bindings=bindings,
        formal_assumptions=assumptions,
        disjoint_groups=tuple(disjoint_groups),
        side_bindings=side_bindings,
        parameters=parameters,
        constraints=constraints,
    )
    output_numel = parse_expr(
        contract_raw["output_numel"], f"{path}.contract.output_numel"
    )
    if output_numel.sort != Sort.INT:
        raise InputError(f"{path}.contract.output_numel must be an integer expression")
    return PairSpec(
        pair_id=pair_id,
        source=path,
        lhs_path=lhs_path,
        rhs_path=rhs_path,
        semantic_mode=semantic_mode,
        predicates=predicate_set,
        assumptions=assumptions,
        contract=Contract(
            output_role=output_role,
            output_numel=output_numel,
            require_full_coverage=require_full_coverage,
            require_disjoint=require_disjoint,
        ),
        limits=Limits(**limit_values),
        lhs_frontend=parse_frontend("lhs"),
        rhs_frontend=parse_frontend("rhs"),
        rewrite_rules=rewrite_rules,
        rule_policy=RulePolicy(
            algebraic_validation=algebraic_validation,
            non_algebraic_validation=non_algebraic_validation,
        ),
        llm=LLMConfig(**llm_values),
        partition=PartitionConfig(**partition_values),
    )


def load_pair_spec(path: Path) -> PairSpec:
    """Load the production PairSpec contract: both artifacts must be raw TTIR."""

    return _load_pair_spec(path, require_ttir=True)


def load_internal_pair_spec(path: Path) -> PairSpec:
    """Load Semantic IR fixtures used only to test the proof core."""

    return _load_pair_spec(path, require_ttir=False)
