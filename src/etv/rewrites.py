"""Per-run candidate parsing and formal/trusted admission."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
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


@lru_cache(maxsize=100000)
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
        ("fmax_commute", term("fmax", x, y), term("fmax", y, x)),
        ("fmin_commute", term("fmin", x, y), term("fmin", y, x)),
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
        ("fmax_select", term("fmax", x, y), term("select", Expr("gt", BOOL, (x, y)), x, y)),
        ("fmin_select", term("fmin", x, y), term("select", Expr("lt", BOOL, (x, y)), x, y)),
    ]
    p = Expr("pvar", BOOL, attrs=(("name", "p"),))
    candidates.extend(
        [
            ("bool_or_false", Expr("or", BOOL, (p, const(False, BOOL))), p),
            ("bool_false_or", Expr("or", BOOL, (const(False, BOOL), p)), p),
            ("bool_and_true", Expr("and", BOOL, (p, const(True, BOOL))), p),
            ("bool_true_and", Expr("and", BOOL, (const(True, BOOL), p)), p),
            ("float_eq_reflexive", Expr("eq", BOOL, (x, x)), const(True, BOOL)),
            ("float_ne_reflexive", Expr("ne", BOOL, (x, x)), const(False, BOOL)),
            ("select_true", term("select", const(True, BOOL), x, y), x),
            ("select_false", term("select", const(False, BOOL), x, y), y),
            (
                "select_gt_reverse",
                term("select", Expr("gt", BOOL, (x, y)), x, y),
                term("select", Expr("lt", BOOL, (y, x)), x, y),
            ),
            (
                "select_le_complement",
                term("select", Expr("le", BOOL, (x, y)), z, x),
                term("select", Expr("gt", BOOL, (x, y)), x, z),
            ),
        ]
    )
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
    # Abstract memory leaves into independent typed variables. These are only
    # candidates: admit_candidates must prove universal equality AND equal
    # definedness in Z3 before any instance enters egglog.
    seen: set[tuple[Expr, Expr]] = set()
    ground_seen: set[tuple[Expr, Expr]] = set()
    memory_seen: set[tuple[Expr, Expr]] = set()
    for root in roots:
        # A ground rule retains each exact read/variable term. It cannot match
        # an arbitrary floating subtree, so finite external-input semantics may
        # be used without extending a finite-only theorem to +/-infinity.
        if (
            root.lhs != root.rhs
            and (root.lhs, root.rhs) not in ground_seen
            and len(ground_seen) < 16
        ):
            ground_seen.add((root.lhs, root.rhs))
            candidates.append((f"observation_ground.{len(ground_seen)}", root.lhs, root.rhs))
        # Keep the read constructor and logical storage role in this template.
        # Only its address is generalized: unlike a floating pvar, a read is a
        # finite external input under the declared model. Z3 must establish the
        # candidate for every address before the e-graph may apply it to any
        # lane. This avoids one concrete rule per output without introducing a
        # finite-only theorem that could match an arbitrary infinite subtree.
        addresses: dict[Expr, Expr] = {}
        memory_cache: dict[Expr, Expr] = {}
        reserved_names = {
            str(e.attr("key" if e.op == "input" else "name"))
            for expression in (root.lhs, root.rhs)
            for e in walk(expression)
            if e.op in ("var", "input")
        }

        def memory_template(expr: Expr) -> Expr:
            if expr in memory_cache:
                return memory_cache[expr]
            if expr.op == "read":
                address = expr.args[0]
                if address not in addresses:
                    name = f"address{len(addresses)}"
                    while "rule." + name in reserved_names:
                        name += "_"
                    addresses[address] = Expr("pvar", address.type, attrs=(("name", name),))
                result = Expr("read", expr.type, (addresses[address],), expr.attrs)
            else:
                result = Expr(
                    expr.op, expr.type, tuple(memory_template(a) for a in expr.args), expr.attrs
                )
            memory_cache[expr] = result
            return result

        if root.lhs != root.rhs and len(memory_seen) < 16:
            left, right = memory_template(root.lhs), memory_template(root.rhs)
            if addresses and (left, right) not in memory_seen:
                memory_seen.add((left, right))
                candidates.append((f"observation_memory.{len(memory_seen)}", left, right))
        leaves: dict[Expr, Expr] = {}
        abstract_cache: dict[Expr, Expr] = {}

        def abstract(expr: Expr) -> Expr:
            if expr in abstract_cache:
                return abstract_cache[expr]
            if expr.op in ("var", "read", "input"):
                if expr not in leaves:
                    leaves[expr] = Expr("pvar", expr.type, attrs=(("name", f"v{len(leaves)}"),))
                result = leaves[expr]
            else:
                result = Expr(expr.op, expr.type, tuple(abstract(a) for a in expr.args), expr.attrs)
            abstract_cache[expr] = result
            return result

        if root.lhs == root.rhs:
            continue
        left, right = abstract(root.lhs), abstract(root.rhs)
        if left.op == "pvar" or (left, right) in seen:
            continue
        seen.add((left, right))
        candidates.append((f"observation_candidate.{len(seen)}", left, right))
        if len(seen) >= 16:
            break
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
            has_patterns = any(
                e.op == "pvar" for expression in (rule.lhs, rule.rhs) for e in walk(expression)
            )
            encoder = Encoder(spec.index_bits, extended_float_variables=has_patterns)
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
