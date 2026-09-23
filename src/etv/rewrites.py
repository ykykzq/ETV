"""Per-run candidate parsing and formal/trusted admission."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import z3

from .errors import InputError, UnsupportedSemantics
from .ir import BOOL, FLOAT, Expr, RootPair, const, parse_type, var, walk
from .jsonio import JSON, array, digest, mapping, obj, read_json, string
from .pairspec import PairSpec
from .result import RuleFact
from .semantics import validate_expr
from .smt import Encoder, query


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    lhs: Expr
    rhs: Expr
    requires: tuple[str, ...]
    source: str
    sha256: str
    generated: bool = False


@dataclass(frozen=True)
class RuleRegistry:
    rules: tuple[Rule, ...]
    facts: tuple[RuleFact, ...]


def pattern(raw: JSON) -> Expr:
    data = mapping(raw)
    try:
        typ = parse_type(string(data["type"]))
        if "pvar" in data:
            obj(data, "pvar type")
            return Expr("pvar", typ, attrs=(("name", string(data["pvar"])),))
        if "const" in data:
            obj(data, "const type")
            value = data["const"]
            if typ == BOOL:
                if type(value) is not bool:
                    raise ValueError("bool constant must be boolean")
                return const(value, typ)
            if not isinstance(value, (str, int)) or type(value) is bool:
                raise ValueError("numeric constant must be an integer or rational string")
            rational = Fraction(value)
            if typ.kind != "abstract_float" and rational.denominator != 1:
                raise ValueError("fractional integer constant")
            return const(rational if typ.kind == "abstract_float" else int(rational), typ)
        obj(data, "op type args")
        op = string(data["op"])
        if op in ("load", "input", "read", "undefined", "var", "pvar", "coordinate", "to_index"):
            raise ValueError("operator is not public in rewrite patterns")
        result = Expr(op, typ, tuple(pattern(x) for x in array(data["args"])))
        validate_expr(result)
        return result
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise InputError("INVALID_REWRITE", str(exc)) from exc


def parse_rules(path: Path) -> tuple[Rule, ...]:
    return parse_rule_data(read_json(path), str(path), digest(path))


def parse_rule_data(
    raw: JSON, source: str, sha256: str, generated: bool = False
) -> tuple[Rule, ...]:
    data = obj(raw, "format rules")
    if data["format"] != "etv-rewrite-v1":
        raise InputError("INVALID_REWRITE", "expected etv-rewrite-v1")
    result: list[Rule] = []
    for raw_rule in array(data["rules"]):
        rule = obj(raw_rule, "id kind lhs rhs requires validation")
        validation = obj(rule["validation"], "method")
        if validation["method"] not in ("z3", "trusted"):
            raise InputError("INVALID_REWRITE", "unknown validator request")
        kind = string(rule["kind"])
        if kind not in ("algebraic", "relation", "layout"):
            raise InputError("INVALID_REWRITE", "unknown rule kind")
        lhs, rhs = pattern(rule["lhs"]), pattern(rule["rhs"])
        left = {(str(e.attr("name")), e.type) for e in walk(lhs) if e.op == "pvar"}
        right = {(str(e.attr("name")), e.type) for e in walk(rhs) if e.op == "pvar"}
        if (
            left != right
            or len({n for n, _ in left}) != len(left)
            or lhs.type != rhs.type
            or lhs.op == "pvar"
        ):
            raise InputError(
                "INVALID_REWRITE", "pattern variable/sort mismatch or unrestricted lhs"
            )
        result.append(
            Rule(
                string(rule["id"]),
                kind,
                lhs,
                rhs,
                tuple(string(v) for v in array(rule["requires"])),
                source,
                sha256,
                generated,
            )
        )
    if len({rule.id for rule in result}) != len(result):
        raise InputError("INVALID_REWRITE", "duplicate rule id")
    return tuple(result)


def _variables(expr: Expr) -> Expr:
    if expr.op == "pvar":
        return var("rule." + str(expr.attr("name")), expr.type)
    return Expr(expr.op, expr.type, tuple(_variables(a) for a in expr.args), expr.attrs)


def builtin_rules(roots: tuple[RootPair, ...]) -> tuple[Rule, ...]:
    x, y, z = (Expr("pvar", FLOAT, attrs=(("name", name),)) for name in ("x", "y", "z"))

    def term(op: str, *args: Expr) -> Expr:
        return Expr(op, FLOAT, args)

    candidates = [
        ("fadd_commute", term("fadd", x, y), term("fadd", y, x)),
        ("fmul_commute", term("fmul", x, y), term("fmul", y, x)),
        (
            "fadd_associate",
            term("fadd", term("fadd", x, y), z),
            term("fadd", x, term("fadd", y, z)),
        ),
        (
            "fmul_associate",
            term("fmul", term("fmul", x, y), z),
            term("fmul", x, term("fmul", y, z)),
        ),
        ("fadd_zero", term("fadd", x, const(0, FLOAT)), x),
        ("fmul_one", term("fmul", x, const(1, FLOAT)), x),
        ("fsub_zero", term("fsub", x, const(0, FLOAT)), x),
        ("fma_expand", term("fma", x, y, z), term("fadd", term("fmul", x, y), z)),
    ]
    casts = sorted(
        {
            (e.op, e.attrs)
            for root in roots
            for expr in (root.lhs, root.rhs)
            for e in walk(expr)
            if e.op in ("extf", "truncf")
        },
        key=str,
    )
    for index, (op, attrs) in enumerate(casts):
        candidates.append((f"{op}.{index}", Expr(op, FLOAT, (x,), attrs), x))
    return tuple(
        Rule("builtin." + name, "algebraic", lhs, rhs, (), "builtin", "")
        for name, lhs, rhs in candidates
    )


def admit_candidates(candidates: tuple[Rule, ...], spec: PairSpec) -> RuleRegistry:
    ids: set[str] = set()
    predicates = {p.id: p for p in spec.predicates}
    admitted: list[Rule] = []
    facts: list[RuleFact] = []
    for rule in candidates:
        if rule.id in ids:
            raise InputError("INVALID_REWRITE", "duplicate rule id: " + rule.id)
        ids.add(rule.id)
        if not set(rule.requires) <= predicates.keys():
            raise InputError("INVALID_REWRITE", "unknown predicate gate: " + rule.id)
        status, validator, detail = "rejected", "none", ""
        if rule.generated or (rule.kind in ("relation", "layout") and rule.requires):
            status, validator = "admitted_unverified", "trusted"
        elif rule.kind == "algebraic":
            encoder = Encoder(spec.index_bits)
            lhs, rhs = _variables(rule.lhs), _variables(rule.rhs)
            try:
                left, right = encoder.encode(lhs), encoder.encode(rhs)
                dl, dr = encoder.defined(lhs), encoder.defined(rhs)
                if encoder.exact:
                    result = query(
                        z3.Or(dl != dr, z3.And(dl, dr, left != right)),
                        (),
                        spec.limits.smt_timeout_ms,
                    )
                    validator, detail = (
                        "z3",
                        result.status + (": " + result.detail if result.detail else ""),
                    )
                    if result.status == "unsat":
                        status = "formal"
                else:
                    detail = "exact algebraic encoding unavailable"
            except UnsupportedSemantics as exc:
                detail = exc.reason
        else:
            detail = "unverified relation/layout rule requires explicit predicate gates"
        if status != "rejected":
            admitted.append(rule)
        facts.append(
            RuleFact(
                rule.id,
                rule.source,
                rule.sha256,
                validator,
                status,
                rule.requires,
                detail=detail,
                lhs=rule.lhs,
                rhs=rule.rhs,
            )
        )
    return RuleRegistry(tuple(admitted), tuple(facts))


def admit_rules(
    spec: PairSpec, roots: RootPair | tuple[RootPair, ...], generated: tuple[Rule, ...] = ()
) -> RuleRegistry:
    root_tuple = (roots,) if isinstance(roots, RootPair) else roots
    candidates = (
        builtin_rules(root_tuple)
        + tuple(rule for source in spec.rewrites for rule in parse_rules(source.file))
        + generated
    )
    return admit_candidates(candidates, spec)
