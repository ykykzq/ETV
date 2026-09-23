"""Exact arithmetic encodings and counterexample-oriented proof queries."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import z3

from .errors import UnsupportedSemantics
from .ir import BOOL, Expr
from .semantics import CASTS, width


def _wrap(value: z3.ArithRef, bits: int) -> z3.ArithRef:
    half = 1 << (bits - 1)
    return (value + half) % (1 << bits) - half


def _trunc(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    magnitude = z3.Abs(a) / z3.Abs(b)
    return z3.If((a < 0) != (b < 0), -magnitude, magnitude)


class Encoder:
    def __init__(
        self,
        index_bits: int | None = None,
        environment: Mapping[str, z3.ExprRef] | None = None,
        mathematical_int: bool = False,
    ) -> None:
        self.index_bits = index_bits
        self.environment = dict(environment or {})
        self.cache: dict[Expr, z3.ExprRef] = {}
        self.exact = True
        self.mathematical_int = mathematical_int

    def encode(self, expr: Expr) -> z3.ExprRef:
        if expr not in self.cache:
            self.cache[expr] = z3.simplify(self._encode(expr))
        return self.cache[expr]

    def _encode(self, expr: Expr) -> z3.ExprRef:
        op = expr.op
        if op == "const":
            value = expr.attr("value")
            if expr.type == BOOL:
                return z3.BoolVal(bool(value))
            if expr.type.kind == "abstract_float":
                assert isinstance(value, (int, Fraction))
                fraction = Fraction(value)
                return z3.RealVal(f"{fraction.numerator}/{fraction.denominator}")
            return z3.IntVal(value)
        if op in ("var", "input", "undefined"):
            name = (
                str(expr.attr("key" if op == "input" else "name"))
                if op != "undefined"
                else "undefined"
            )
            if name in self.environment:
                return self.environment[name]
            sort = (
                z3.BoolSort()
                if expr.type == BOOL
                else z3.RealSort()
                if expr.type.kind == "abstract_float"
                else z3.IntSort()
            )
            return z3.Const(name, sort)
        if op == "load":
            raise UnsupportedSemantics("MEMORY_NOT_COMPOSED")
        if op == "read":
            memory = z3.Array("memory." + str(expr.attr("role")), z3.IntSort(), z3.RealSort())
            return z3.Select(memory, self.encode(expr.args[0]))
        args = [self.encode(a) for a in expr.args]
        a = args[0]
        b = args[1] if len(args) > 1 else None
        if op in ("coordinate", "to_index"):
            return a
        if op in CASTS:
            if op in ("extf", "truncf"):
                return a
            if expr.args[0].type == BOOL:
                a = z3.If(a, 1, 0)
            source = width(expr.args[0].type, self.index_bits)
            a = a % (1 << source) if op in ("extui", "uitofp", "index_castui") else _wrap(a, source)
            if op in ("sitofp", "uitofp"):
                return z3.ToReal(a)
            if expr.type == BOOL:
                return a % 2 == 1
            return _wrap(a, width(expr.type, self.index_bits))
        if op in ("add", "fadd"):
            result = a + b
        elif op in ("sub", "fsub"):
            result = a - b
        elif op in ("mul", "fmul"):
            result = a * b
        elif op in ("div", "rem"):
            q = _trunc(a, b)
            result = q if op == "div" else a - q * b
        elif op == "ceildiv":
            result = -((-a) / b)
        elif op == "fdiv":
            result = a / b
        elif op in ("eq", "ne", "lt", "le", "gt", "ge"):
            return {
                "eq": lambda: a == b,
                "ne": lambda: a != b,
                "lt": lambda: a < b,
                "le": lambda: a <= b,
                "gt": lambda: a > b,
                "ge": lambda: a >= b,
            }[op]()
        elif op == "and":
            return z3.And(a, b)
        elif op == "or":
            return z3.Or(a, b)
        elif op == "xor":
            return z3.Xor(a, b)
        elif op == "not":
            return z3.Not(a)
        elif op == "select":
            return z3.If(a, args[1], args[2])
        elif op == "fneg":
            result = -a
        elif op == "absf":
            result = z3.If(a >= 0, a, -a)
        elif op in ("fmin", "fmax"):
            result = z3.If(a <= b if op == "fmin" else a >= b, a, b)
        elif op == "floor":
            result = z3.ToReal(z3.ToInt(a))
        elif op == "ceil":
            result = -z3.ToReal(z3.ToInt(-a))
        elif op == "nearbyint":
            low = z3.ToInt(a)
            result = z3.ToReal(
                z3.If(
                    a - low < z3.RealVal("1/2"),
                    low,
                    z3.If(a - low > z3.RealVal("1/2"), low + 1, z3.If(low % 2 == 0, low, low + 1)),
                )
            )
        elif op == "fma":
            result = a * b + args[2]
        else:
            self.exact = False
            function = z3.Function("math." + op, *[v.sort() for v in args], z3.RealSort())
            result = function(*args)
        if expr.type.kind == "int" and not self.mathematical_int:
            assert expr.type.bits is not None
            return _wrap(result, expr.type.bits)
        return result

    def defined(self, expr: Expr) -> z3.BoolRef:
        if expr.op == "undefined":
            return z3.BoolVal(False)
        if expr.op == "select":
            condition = self.encode(expr.args[0])
            return z3.And(
                self.defined(expr.args[0]),
                z3.If(condition, self.defined(expr.args[1]), self.defined(expr.args[2])),
            )
        conditions = [self.defined(arg) for arg in expr.args]
        op = expr.op
        if op in ("div", "rem", "fdiv"):
            a, b = (self.encode(arg) for arg in expr.args)
            conditions.append(b != 0)
            if expr.type.kind == "int":
                assert expr.type.bits is not None
                conditions.append(z3.Not(z3.And(a == -(1 << (expr.type.bits - 1)), b == -1)))
        elif op == "ceildiv":
            conditions.append(self.encode(expr.args[1]) > 0)
        elif op in ("sqrt", "rsqrt", "log", "acosh", "pow"):
            a = self.encode(expr.args[0])
            conditions.append(a >= 0 if op == "sqrt" else a >= 1 if op == "acosh" else a > 0)
        if expr.op == "var" and expr.type.kind == "int":
            assert expr.type.bits is not None
            a = self.encode(expr)
            conditions.extend((a >= -(1 << (expr.type.bits - 1)), a < (1 << (expr.type.bits - 1))))
        if expr.op == "coordinate" and expr.type.kind == "int":
            assert expr.type.bits is not None
            a = self.encode(expr)
            conditions.extend((a >= -(1 << (expr.type.bits - 1)), a < (1 << (expr.type.bits - 1))))
        if (
            self.mathematical_int
            and expr.type.kind == "int"
            and expr.op in ("add", "sub", "mul", "div", "rem")
        ):
            assert expr.type.bits is not None
            a = self.encode(expr)
            conditions.extend((a >= -(1 << (expr.type.bits - 1)), a < (1 << (expr.type.bits - 1))))
        return z3.And(*conditions)


@dataclass(frozen=True)
class QueryResult:
    status: str
    model: tuple[tuple[str, str], ...] = ()
    detail: str = ""


def query(
    counterexample: z3.BoolRef, premises: tuple[z3.BoolRef, ...], timeout_ms: int
) -> QueryResult:
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=0)
    solver.add(*premises, counterexample)
    answer = solver.check()
    if answer == z3.unsat:
        return QueryResult("unsat")
    if answer == z3.unknown:
        return QueryResult("unknown", detail=solver.reason_unknown())
    model = solver.model()
    if not z3.is_true(model.eval(z3.And(*premises, counterexample), model_completion=True)):
        return QueryResult("unknown", detail="model did not replay the query")
    assignments = tuple(sorted((str(d), str(model[d])) for d in model.decls()))
    return QueryResult("sat", assignments)
