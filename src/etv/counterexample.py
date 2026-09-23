"""Replay exact value counterexamples against the scalar evaluator."""

from dataclasses import dataclass
from fractions import Fraction

import z3

from .errors import UnsupportedSemantics
from .ir import Expr, RootPair, const
from .pairspec import PairSpec
from .result import Counterexample
from .semantics import Value, evaluate
from .smt import Encoder


@dataclass(frozen=True)
class ValueCheck:
    status: str
    counterexample: Counterexample | None = None
    detail: str = ""


def _number(value: z3.ExprRef) -> Value:
    if z3.is_true(value):
        return True
    if z3.is_false(value):
        return False
    if z3.is_int_value(value):
        return value.as_long()
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise ValueError("non-rational model value")


def find_counterexample(root: RootPair, spec: PairSpec) -> ValueCheck:
    encoder = Encoder(spec.index_bits)
    lhs, rhs = encoder.encode(root.lhs), encoder.encode(root.rhs)
    premises = tuple(encoder.encode(p) for p in root.premises)
    if not encoder.exact:
        return ValueCheck(
            "unknown", detail="non-algebraic functions lack an exact counterexample encoding"
        )
    solver = z3.Solver()
    solver.set(timeout=spec.limits.smt_timeout_ms, random_seed=0)
    query = z3.And(*premises, encoder.defined(root.lhs), encoder.defined(root.rhs), lhs != rhs)
    solver.add(query)
    status = solver.check()
    if status == z3.unsat:
        return ValueCheck("unsat")
    if status == z3.unknown:
        return ValueCheck("unknown", detail=solver.reason_unknown())
    model = solver.model()
    if not z3.is_true(model.eval(query, model_completion=True)):
        return ValueCheck("unknown", detail="model did not satisfy query")

    def concrete(expr: Expr) -> Expr:
        if expr.op in ("var", "input", "read"):
            return const(
                _number(model.eval(encoder.encode(expr), model_completion=True)), expr.type
            )
        return Expr(expr.op, expr.type, tuple(concrete(a) for a in expr.args), expr.attrs)

    try:
        left = evaluate(concrete(root.lhs), index_bits=spec.index_bits)
        right = evaluate(concrete(root.rhs), index_bits=spec.index_bits)
        index = _number(model.eval(encoder.encode(root.index), model_completion=True))
        if left == right:
            return ValueCheck("unknown", detail="independent evaluation did not reproduce mismatch")
    except (ValueError, KeyError, UnsupportedSemantics, ZeroDivisionError) as exc:
        return ValueCheck("unknown", detail="exact replay unavailable: " + str(exc))
    assignments = tuple(sorted((str(d), str(model[d])) for d in model.decls()))
    return ValueCheck(
        "sat",
        Counterexample(
            str(index),
            assignments,
            str(index),
            str(index),
            True,
            True,
            str(left),
            str(right),
            "Evaluate the typed observation roots at the reported index with the exact scalar/array model. Both evaluators reproduce unequal rational values.",
        ),
    )
