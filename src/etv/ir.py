"""The sole typed semantic representation, independent of solvers and frontends."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Literal, TypeAlias
from types import NotImplementedType


@dataclass(frozen=True, order=True)
class Type:
    kind: Literal["bool", "int", "index", "abstract_float"]
    bits: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "int" and (self.bits is None or self.bits < 2):
            raise ValueError("integer type requires at least two bits")
        if self.kind in ("bool", "index") and self.bits is not None:
            raise ValueError("bool/index do not carry a fixed width")

    def __str__(self) -> str:
        return f"i{self.bits}" if self.kind == "int" else self.kind


BOOL = Type("bool")
INDEX = Type("index")
FLOAT = Type("abstract_float")


def parse_type(name: str) -> Type:
    if name in ("bool", "i1"):
        return BOOL
    if name == "index":
        return INDEX
    if name == "abstract_float":
        return FLOAT
    if name.startswith("i") and name[1:].isdigit():
        return Type("int", int(name[1:]))
    raise ValueError(f"invalid type: {name}")


@dataclass(frozen=True, order=True)
class StorageRef:
    side: Literal["lhs", "rhs"]
    launch_id: str
    physical_arg: str
    logical_role: str
    base_offset: int = 0
    scalar_index: int | None = None


ScalarAttr: TypeAlias = str | int | bool | Fraction | StorageRef


@dataclass(frozen=True)
class Expr:
    op: str
    type: Type
    args: tuple["Expr", ...] = ()
    attrs: tuple[tuple[str, ScalarAttr], ...] = ()

    def __post_init__(self) -> None:
        if self.attrs != tuple(sorted(self.attrs, key=lambda item: item[0])):
            raise ValueError("attributes must be sorted")
        if len({key for key, _ in self.attrs}) != len(self.attrs):
            raise ValueError("duplicate attribute")

    def __hash__(self) -> int:
        # Expr graphs share subexpressions. Dataclass's recursive tuple hash
        # expands that DAG into a tree on every dictionary/cache lookup.
        # This memo is deliberately not a dataclass field: it is implementation
        # state, never semantic identity or part of JSON report serialization.
        cached = self.__dict__.get("_structural_hash")
        if cached is None:
            cached = hash((self.op, self.type, self.args, self.attrs))
            object.__setattr__(self, "_structural_hash", cached)
        assert isinstance(cached, int)
        return cached

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if self is other:
            return True
        if not isinstance(other, Expr) or type(self) is not type(other):
            return NotImplemented
        if hash(self) != hash(other):
            return False
        # Structural equality must visit a pair of shared DAG nodes once. A
        # hash match alone is never evidence of equality (collisions are legal).
        pending = [(self, other)]
        seen: set[tuple[int, int]] = set()
        while pending:
            left, right = pending.pop()
            if left is right:
                continue
            key = (id(left), id(right))
            if key in seen:
                continue
            seen.add(key)
            if (
                type(left) is not type(right)
                or left.op != right.op
                or left.type != right.type
                or left.attrs != right.attrs
                or len(left.args) != len(right.args)
            ):
                return False
            pending.extend(zip(left.args, right.args, strict=True))
        return True

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        # Python randomizes string hashes per interpreter. Do not send the
        # parent's cached hash to egglog's multiprocessing-spawn worker.
        return (type(self), (self.op, self.type, self.args, self.attrs))

    def attr(self, name: str) -> ScalarAttr:
        return dict(self.attrs)[name]


def const(value: int | bool | Fraction | str, typ: Type = INDEX) -> Expr:
    if typ.kind == "abstract_float":
        if value not in ("+inf", "-inf"):
            value = Fraction(value)
    elif isinstance(value, str):
        raise ValueError("special constants require abstract_float")
    elif typ.kind == "int":
        assert typ.bits is not None
        value = signed(int(value), typ.bits)
    elif typ == BOOL:
        value = bool(value)
    return Expr("const", typ, attrs=(("value", value),))


def var(name: str, typ: Type = INDEX) -> Expr:
    return Expr("var", typ, attrs=(("name", name),))


def signed(value: int, bits: int) -> int:
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def trunc_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)


def _free_variables(expr: Expr) -> frozenset[str]:
    cached = expr.__dict__.get("_free_variables")
    if cached is None:
        cached = (
            frozenset((str(expr.attr("name")),))
            if expr.op == "var"
            else frozenset().union(*(_free_variables(arg) for arg in expr.args))
        )
        object.__setattr__(expr, "_free_variables", cached)
    assert isinstance(cached, frozenset)
    return cached


@lru_cache(maxsize=100000)
def substitute(expr: Expr, items: tuple[tuple[str, Expr], ...]) -> Expr:
    if not items:
        return expr
    if expr.op == "var":
        return dict(items).get(str(expr.attr("name")), expr)
    if not any(name in _free_variables(expr) for name, _ in items):
        return expr
    args = tuple(substitute(arg, items) for arg in expr.args)
    if all(new is old for new, old in zip(args, expr.args, strict=True)):
        return expr
    return Expr(expr.op, expr.type, args, expr.attrs)


def walk(expr: Expr) -> tuple[Expr, ...]:
    result: list[Expr] = []
    seen: set[Expr] = set()

    def visit(node: Expr) -> None:
        if node in seen:
            return
        seen.add(node)
        for arg in node.args:
            visit(arg)
        result.append(node)

    visit(expr)
    return tuple(result)


@dataclass(frozen=True)
class Store:
    store_index: int
    storage: StorageRef
    logical_index: Expr
    offset: Expr
    mask: Expr
    value: Expr
    lane_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class Kernel:
    launch_id: str
    function: str
    programs: Expr
    lane_shape: tuple[int, ...]
    stores: tuple[Store, ...]
    source_hash: str
    parameters: tuple[Expr, ...] = ()
    effects: tuple[Expr, ...] = ()


@dataclass(frozen=True)
class ProgramSequence:
    steps: tuple[tuple[Kernel, ...], ...]


@dataclass(frozen=True)
class ProgramPair:
    lhs: ProgramSequence
    rhs: ProgramSequence


@dataclass(frozen=True)
class RootPair:
    lhs: Expr
    rhs: Expr
    index: Expr
    premises: tuple[Expr, ...] = ()
