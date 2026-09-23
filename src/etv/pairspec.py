"""Strict v3 contracts. Natural-language assumptions never enter the proof."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .errors import InputError
from .ir import BOOL, INDEX, Expr, Type, const, parse_type, var
from .jsonio import (
    JSON,
    array,
    boolean,
    digest,
    integer,
    mapping,
    obj,
    read_json,
    resolve_file,
    string,
)


@dataclass(frozen=True)
class Limits:
    smt_timeout_ms: int = 2000
    egraph_timeout_ms: int = 10000
    max_iterations: int = 8
    max_enodes: int = 20000
    max_instances: int = 100000


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    base_url_env: str = ""
    api_key_env: str = ""
    timeout_ms: int = 10000


@dataclass(frozen=True)
class PartitionConfig:
    enabled: bool = False
    provider: str = "none"
    max_parts: int = 16


@dataclass(frozen=True)
class Role:
    name: str
    kind: Literal["buffer", "scalar"]
    element: Type


@dataclass(frozen=True)
class Endpoint:
    role: str
    kind: Literal["block", "scalar", "scalar_block"]
    name: str
    offset: int = 0
    index: int | None = None


@dataclass(frozen=True)
class SelectedStore:
    index: int
    role: str


@dataclass(frozen=True)
class LaunchSpec:
    id: str
    side: Literal["lhs", "rhs"]
    file: Path
    function: str
    step: int
    programs: Expr
    stores: tuple[SelectedStore, ...]
    abi: tuple[Endpoint, ...]
    bindings: tuple[tuple[str, Expr], ...]
    index_bits: int | None
    max_instances: int


@dataclass(frozen=True)
class Parameter:
    name: str
    type: Type
    minimum: int
    maximum: int


@dataclass(frozen=True)
class Predicate:
    id: str
    kind: str
    description: str
    expr: Expr | None = None
    trusted: bool = False


@dataclass(frozen=True)
class RewriteSource:
    file: Path
    sha256: str


@dataclass(frozen=True)
class Observation:
    role: str
    numel: Expr
    require_full_coverage: bool
    require_disjoint: tuple[str, ...]


@dataclass(frozen=True)
class PairSpec:
    path: Path
    pair_id: str
    semantic_mode: str
    index_bits: int | None
    limits: Limits
    lhs: tuple[LaunchSpec, ...]
    rhs: tuple[LaunchSpec, ...]
    roles: tuple[Role, ...]
    bindings: tuple[tuple[str, Expr], ...]
    parameters: tuple[Parameter, ...]
    predicates: tuple[Predicate, ...]
    disjoint: tuple[tuple[str, ...], ...]
    observation: Observation
    rewrites: tuple[RewriteSource, ...]
    assumptions_for_llm: tuple[str, ...]
    partition: PartitionConfig
    llm: LLMConfig


ARITIES = {
    **dict.fromkeys("add sub mul div rem ceildiv eq ne lt le gt ge and or".split(), 2),
    "not": 1,
    "select": 3,
}


def config_expr(value: JSON, symbols: dict[str, Expr]) -> Expr:
    if type(value) is bool:
        return const(value, BOOL)
    if type(value) is int:
        return const(value)
    node = mapping(value)
    if "var" in node:
        obj(node, "var")
        name = string(node["var"])
        if name not in symbols:
            raise InputError("INVALID_PAIRSPEC", f"undeclared variable: {name}")
        return symbols[name]
    obj(node, "op args")
    op = string(node["op"])
    args = tuple(config_expr(v, symbols) for v in array(node["args"]))
    if op not in ARITIES or len(args) != ARITIES[op]:
        raise InputError("INVALID_PAIRSPEC", f"invalid operator/arity: {op}")
    types = tuple(a.type for a in args)
    if op == "select":
        valid = types[0] == BOOL and types[1] == types[2]
        typ = types[1]
    elif op in ("and", "or", "not"):
        valid = all(t == BOOL for t in types)
        typ = BOOL
    elif op in ("eq", "ne"):
        valid = types[0] == types[1]
        typ = BOOL
    else:
        valid = all(t == INDEX for t in types)
        typ = BOOL if op in ("lt", "le", "gt", "ge") else INDEX
    if not valid:
        raise InputError("INVALID_PAIRSPEC", f"sort mismatch: {op}")
    return Expr(op, typ, args)


def _bindings(value: JSON, symbols: dict[str, Expr]) -> tuple[tuple[str, Expr], ...]:
    pending = dict(mapping(value))
    result: dict[str, Expr] = {}
    if pending.keys() & symbols.keys():
        raise InputError("INVALID_PAIRSPEC", "binding shadows declared parameter or binding")
    while pending:
        progressed = False
        for name, raw in tuple(sorted(pending.items())):
            try:
                expr = config_expr(raw, {**symbols, **result})
            except InputError as exc:
                if exc.detail.startswith("undeclared variable:"):
                    continue
                raise
            if expr.type != INDEX:
                raise InputError("INVALID_PAIRSPEC", f"noninteger binding: {name}")
            result[name] = expr
            del pending[name]
            progressed = True
        if not progressed:
            raise InputError(
                "INVALID_PAIRSPEC", f"cyclic or undeclared bindings: {sorted(pending)}"
            )
    return tuple(sorted(result.items()))


def _launches(
    value: JSON,
    side: Literal["lhs", "rhs"],
    root: Path,
    roles: dict[str, Role],
    symbols: dict[str, Expr],
    bits: int | None,
    limits: Limits,
    external: bool,
) -> tuple[LaunchSpec, ...]:
    result: list[LaunchSpec] = []
    for raw in array(value):
        item = obj(raw, "id file function step grid stores abi bindings")
        bindings = _bindings(item["bindings"], symbols)
        local = {**symbols, **dict(bindings)}
        programs = config_expr(obj(item["grid"], "programs")["programs"], local)
        if programs.type != INDEX:
            raise InputError("INVALID_PAIRSPEC", "grid must be integer")
        abi: list[Endpoint] = []
        for name, raw_endpoint in sorted(mapping(item["abi"]).items()):
            ep = obj(raw_endpoint, "kind name", "index offset")
            kind = string(ep["kind"])
            if name not in roles or kind not in ("block", "scalar", "scalar_block"):
                raise InputError("INVALID_PAIRSPEC", f"invalid ABI role/kind: {name}")
            if (kind == "block") != (roles[name].kind == "buffer"):
                raise InputError("INVALID_PAIRSPEC", f"endpoint kind disagrees with role: {name}")
            if ("index" in ep) != (kind == "scalar_block"):
                raise InputError("INVALID_PAIRSPEC", "scalar_block requires an index")
            if "offset" in ep and kind == "scalar":
                raise InputError("INVALID_PAIRSPEC", "scalar endpoint cannot have an offset")
            arg = string(ep["name"])
            if not arg.startswith("arg") or not arg[3:].isdigit():
                raise InputError("INVALID_PAIRSPEC", "ABI names are positional argN")
            abi.append(
                Endpoint(
                    name,
                    cast(Literal["block", "scalar", "scalar_block"], kind),
                    arg,
                    integer(ep.get("offset", 0)),
                    integer(ep["index"], 0) if "index" in ep else None,
                )
            )
        if len({ep.name for ep in abi}) != len(abi):
            raise InputError("INVALID_PAIRSPEC", "duplicate physical endpoint")
        if {ep.name for ep in abi} & dict(bindings).keys():
            raise InputError("INVALID_PAIRSPEC", "ABI endpoint also has a binding")
        stores = []
        for raw_store in array(item["stores"]):
            store = obj(raw_store, "index role")
            role = string(store["role"])
            if role not in roles or roles[role].kind != "buffer":
                raise InputError("INVALID_PAIRSPEC", "undeclared store role")
            stores.append(SelectedStore(integer(store["index"], 0), role))
        if not stores or len({s.index for s in stores}) != len(stores):
            raise InputError("INVALID_PAIRSPEC", "empty or duplicate selected stores")
        result.append(
            LaunchSpec(
                string(item["id"]),
                side,
                resolve_file(root, string(item["file"]), external),
                string(item["function"]),
                integer(item["step"], 0),
                programs,
                tuple(stores),
                tuple(abi),
                bindings,
                bits,
                limits.max_instances,
            )
        )
    if not result or [x.step for x in result] != sorted(x.step for x in result):
        raise InputError("INVALID_PAIRSPEC", "launches must be nonempty and ordered by step")
    return tuple(result)


def parse_pair_spec(path: Path, allow_external: bool = False) -> PairSpec:
    path = path.resolve()
    top = obj(
        read_json(path), "format metadata programs assumptions predicates observation rewrites"
    )
    if top["format"] != "etv-pair-v3":
        raise InputError("INVALID_PAIRSPEC", "expected etv-pair-v3")
    meta = obj(top["metadata"], "pair_id semantic_mode limits", "target partition llm")
    if meta["semantic_mode"] != "abstract_float":
        raise InputError("INVALID_PAIRSPEC", "only abstract_float is defined")
    target = obj(meta.get("target", {}), "", "index_bits")
    bits = integer(target["index_bits"]) if "index_bits" in target else None
    if bits is not None and bits not in (32, 64):
        raise InputError("INVALID_PAIRSPEC", "index_bits must be 32 or 64")
    lim = obj(
        meta["limits"],
        "smt_timeout_ms egraph_timeout_ms max_iterations max_enodes",
        "max_instances",
    )
    limits = Limits(**{k: integer(v, 1) for k, v in lim.items()})
    part = obj(meta.get("partition", {}), "", "enabled provider max_parts")
    provider = string(part.get("provider", "none"))
    if provider not in ("none", "deterministic", "llm"):
        raise InputError("INVALID_PAIRSPEC", "unknown partition provider")
    partition = PartitionConfig(
        boolean(part.get("enabled", False)), provider, integer(part.get("max_parts", 16), 1)
    )
    llm_raw = obj(
        meta.get("llm", {"enabled": False}),
        "enabled",
        "provider model base_url_env api_key_env timeout_ms",
    )
    enabled = boolean(llm_raw["enabled"])
    if enabled:
        obj(llm_raw, "enabled provider model base_url_env api_key_env timeout_ms")
    llm = LLMConfig(
        enabled,
        string(llm_raw.get("provider", "none")),
        string(llm_raw["model"]) if enabled else "",
        string(llm_raw["base_url_env"]) if enabled else "",
        string(llm_raw["api_key_env"]) if enabled else "",
        integer(llm_raw.get("timeout_ms", 10000), 1),
    )
    if enabled and llm.provider != "openai_compatible":
        raise InputError("INVALID_PAIRSPEC", "unknown LLM provider")
    pred = obj(top["predicates"], "roles bindings parameters constraints disjoint custom")
    roles: dict[str, Role] = {}
    for name, raw_role in sorted(mapping(pred["roles"]).items()):
        role = obj(raw_role, "kind element")
        kind = string(role["kind"])
        if kind not in ("buffer", "scalar"):
            raise InputError("INVALID_PAIRSPEC", "invalid role kind")
        try:
            typ = parse_type(string(role["element"]))
        except ValueError as exc:
            raise InputError("INVALID_PAIRSPEC", str(exc)) from exc
        roles[name] = Role(name, cast(Literal["buffer", "scalar"], kind), typ)
    params: list[Parameter] = []
    symbols: dict[str, Expr] = {}
    for name, raw_param in sorted(mapping(pred["parameters"]).items()):
        param = obj(raw_param, "type min max")
        try:
            typ = parse_type(string(param["type"]))
        except ValueError as exc:
            raise InputError("INVALID_PAIRSPEC", str(exc)) from exc
        low, high = integer(param["min"]), integer(param["max"])
        if (
            typ.kind != "int"
            or typ.bits is None
            or not -(1 << (typ.bits - 1)) <= low <= high < (1 << (typ.bits - 1))
        ):
            raise InputError("INVALID_PAIRSPEC", f"invalid parameter domain: {name}")
        params.append(Parameter(name, typ, low, high))
        symbols[name] = var(name)
    bindings = _bindings(pred["bindings"], symbols)
    symbols.update(bindings)
    predicates: list[Predicate] = []
    for parameter in params:
        expr = Expr(
            "and",
            BOOL,
            (
                Expr("ge", BOOL, (var(parameter.name), const(parameter.minimum))),
                Expr("le", BOOL, (var(parameter.name), const(parameter.maximum))),
            ),
        )
        predicates.append(
            Predicate(f"parameter.{parameter.name}", "parameter", str(parameter), expr)
        )
    for name, expr in bindings:
        predicates.append(Predicate(f"binding.global.{name}", "binding", name, expr))
    for collection in ("constraints", "custom"):
        for raw in array(pred[collection]):
            item = (
                obj(raw, "id expr")
                if collection == "constraints"
                else obj(raw, "id encoder", "expr")
            )
            encoder = item.get("encoder", "z3")
            if encoder not in ("z3", "trusted") or (encoder == "z3" and "expr" not in item):
                raise InputError("INVALID_PAIRSPEC", "invalid predicate encoder")
            constraint = config_expr(item["expr"], symbols) if "expr" in item else None
            if constraint is not None and constraint.type != BOOL:
                raise InputError("INVALID_PAIRSPEC", "constraint must be boolean")
            predicates.append(
                Predicate(
                    string(item["id"]), collection, str(encoder), constraint, encoder == "trusted"
                )
            )
    disjoint: list[tuple[str, ...]] = []
    for raw in array(pred["disjoint"]):
        group = tuple(sorted(string(v) for v in array(raw)))
        if len(group) < 2 or len(set(group)) != len(group) or not set(group) <= roles.keys():
            raise InputError("INVALID_PAIRSPEC", "invalid disjoint group")
        disjoint.append(group)
        predicates.append(Predicate("disjoint." + ".".join(group), "disjoint", ",".join(group)))
    programs = obj(top["programs"], "lhs rhs")
    lhs = _launches(
        programs["lhs"], "lhs", path.parent, roles, symbols, bits, limits, allow_external
    )
    rhs = _launches(
        programs["rhs"], "rhs", path.parent, roles, symbols, bits, limits, allow_external
    )
    if len({x.id for x in lhs + rhs}) != len(lhs + rhs):
        raise InputError("INVALID_PAIRSPEC", "launch ids must be globally unique")
    for launch in lhs + rhs:
        for ep in launch.abi:
            predicates.append(Predicate(f"abi.{launch.id}.{ep.role}", "abi", str(ep)))
        for name, expr in launch.bindings:
            predicates.append(Predicate(f"binding.{launch.id}.{name}", "binding", name, expr))
    observation = obj(top["observation"], "role numel require_full_coverage require_disjoint")
    obs_role = string(observation["role"])
    if obs_role not in roles or roles[obs_role].kind != "buffer":
        raise InputError("INVALID_PAIRSPEC", "undeclared observation buffer")
    for launches in (lhs, rhs):
        if not any(s.role == obs_role for launch in launches for s in launch.stores):
            raise InputError("INVALID_PAIRSPEC", "observation has no selected store")
    obs_numel = config_expr(observation["numel"], symbols)
    if obs_numel.type != INDEX:
        raise InputError("INVALID_PAIRSPEC", "numel must be integer")
    required = tuple(sorted(string(x) for x in array(observation["require_disjoint"])))
    if len(set(required)) != len(required) or not set(required) <= roles.keys():
        raise InputError("INVALID_PAIRSPEC", "unknown/duplicate disjoint role")
    sources = []
    for raw in array(top["rewrites"]):
        source = obj(raw, "file", "expected_sha256")
        file = resolve_file(path.parent, string(source["file"]), allow_external)
        actual = digest(file)
        if "expected_sha256" in source and source["expected_sha256"] != actual:
            raise InputError("FILE_HASH_MISMATCH", str(file))
        sources.append(RewriteSource(file, actual))
    if len({p.id for p in predicates}) != len(predicates):
        raise InputError("INVALID_PAIRSPEC", "duplicate predicate id")
    assumptions = obj(top["assumptions"], "for_llm")
    return PairSpec(
        path,
        string(meta["pair_id"]),
        "abstract_float",
        bits,
        limits,
        lhs,
        rhs,
        tuple(roles.values()),
        bindings,
        tuple(params),
        tuple(sorted(predicates, key=lambda p: p.id)),
        tuple(sorted(disjoint)),
        Observation(obs_role, obs_numel, boolean(observation["require_full_coverage"]), required),
        tuple(sources),
        tuple(string(x) for x in array(assumptions["for_llm"])),
        partition,
        llm,
    )
