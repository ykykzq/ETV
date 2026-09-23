"""Official libtriton parsing followed by a read-only structured MLIR snapshot."""

from dataclasses import dataclass
from fractions import Fraction
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

from .errors import InputError, ResourceLimit, UnsupportedSemantics
from .jsonio import digest


@dataclass(frozen=True)
class IRType:
    kind: str
    assembly: str
    bits: int | None = None
    shape: tuple[int, ...] = ()
    element: "IRType | None" = None
    encoding: str = ""
    address_space: int | None = None
    signedness: int = 0


@dataclass(frozen=True)
class Attribute:
    name: str
    kind: str
    assembly: str
    values: tuple[int | str | Fraction | None, ...] = ()
    splat: bool = False
    children: tuple["Attribute", ...] = ()


@dataclass(frozen=True)
class Value:
    id: str
    type: IRType


@dataclass(frozen=True)
class Block:
    id: int
    arguments: tuple[Value, ...]
    operations: tuple["Operation", ...]


@dataclass(frozen=True)
class Operation:
    ordinal: int
    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: tuple[Attribute, ...]
    block: int | None
    regions: tuple[tuple[Block, ...], ...]
    location: str
    assembly: str

    def attribute(self, name: str) -> Attribute:
        for attr in self.attributes:
            if attr.name == name:
                return attr
        raise UnsupportedSemantics("TTIR_ATTRIBUTE_UNSUPPORTED", f"{self.name}: missing {name}")

    def integer(self, name: str) -> int:
        attr = self.attribute(name)
        if attr.kind != "integer" or type(attr.values[0]) is not int:
            raise UnsupportedSemantics("TTIR_ATTRIBUTE_UNSUPPORTED", f"{self.name}: {name}")
        return attr.values[0]

    def string(self, name: str) -> str:
        attr = self.attribute(name)
        if attr.kind != "string" or not isinstance(attr.values[0], str):
            raise UnsupportedSemantics("TTIR_ATTRIBUTE_UNSUPPORTED", f"{self.name}: {name}")
        return attr.values[0]


@dataclass(frozen=True)
class TTIRModule:
    path: Path
    source_hash: str
    triton_version: str
    module: Operation
    function: Operation


def operations(op: Operation) -> tuple[Operation, ...]:
    return (op,) + tuple(
        child
        for region in op.regions
        for block in region
        for nested in block.operations
        for child in operations(nested)
    )


def _type(raw: dict[str, Any]) -> IRType:
    return IRType(
        raw["kind"],
        raw["assembly"],
        raw.get("bits"),
        tuple(raw.get("shape", ())),
        _type(raw["element"]) if "element" in raw else None,
        raw.get("encoding", ""),
        raw.get("address_space"),
        raw.get("signedness", 0),
    )


def _value(raw: dict[str, Any]) -> Value:
    return Value(raw["id"], _type(raw["type"]))


def _attribute(name: str, raw: dict[str, Any]) -> Attribute:
    kind = raw["kind"]
    values: tuple[int | str | Fraction | None, ...] = ()
    if kind == "integer":
        values = (int(raw["value"]),)
    elif kind == "float":
        values = (Fraction(raw["value"]) if raw["value"] is not None else None,)
    elif kind == "string":
        values = (raw["value"],)
    elif kind == "dense":
        values = tuple(
            int(v) if isinstance(v, str) else Fraction(v) if v is not None else None
            for v in raw["value"]
        )
    children = (
        tuple(_attribute(str(i), v) for i, v in enumerate(raw["value"]))
        if kind == "array"
        else tuple(_attribute(k, v) for k, v in sorted(raw["value"].items()))
        if kind == "dictionary"
        else ()
    )
    return Attribute(name, kind, raw["assembly"], values, raw.get("splat", False), children)


def _operation(raw: dict[str, Any]) -> Operation:
    regions = tuple(
        tuple(
            Block(
                b["id"],
                tuple(_value(a) for a in b["arguments"]),
                tuple(_operation(o) for o in b["operations"]),
            )
            for b in r
        )
        for r in raw["regions"]
    )
    return Operation(
        raw["ordinal"],
        raw["name"],
        tuple(_value(v) for v in raw["operands"]),
        tuple(_value(v) for v in raw["results"]),
        tuple(_attribute(k, v) for k, v in sorted(raw["attributes"].items())),
        raw["block"],
        regions,
        raw["location"],
        raw["assembly"],
    )


def parse_ttir(path: Path, function: str) -> TTIRModule:
    source_hash = digest(path)
    try:
        version = importlib.metadata.version("triton")
        if version.split("+")[0] != "3.7.1":
            raise InputError("TTIR_VERSION_MISMATCH", version)
        ir = importlib.import_module("triton._C.libtriton").ir
        native = importlib.import_module("etv._mlir_native")
        if native.triton_version != "3.7.1":
            raise InputError("TTIR_VERSION_MISMATCH", native.triton_version)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise InputError(
            "TTIR_VERSION_MISMATCH", "pinned Triton and MLIR adapter are required"
        ) from exc
    context = ir.context()
    ir.load_dialects(context)
    try:
        module = ir.parse_mlir_module(str(path), context)
    except RuntimeError as exc:
        raise InputError("TTIR_PARSE_FAILED", str(exc)) from exc
    if not module.verify():
        raise InputError("TTIR_VERIFY_FAILED", str(path))
    try:
        snapshot = _operation(native.snapshot(module))
    except RuntimeError as exc:
        if "RESOURCE_LIMIT" in str(exc):
            raise ResourceLimit(str(exc)) from exc
        raise InputError("TTIR_VERIFY_FAILED", str(exc)) from exc
    functions = [
        op
        for region in snapshot.regions
        for block in region
        for op in block.operations
        if op.name == "tt.func" and op.string("sym_name") == function
    ]
    if len(functions) != 1:
        raise InputError("TTIR_FUNCTION_NOT_FOUND", function)
    if digest(path) != source_hash:
        raise InputError("FILE_HASH_MISMATCH", str(path))
    return TTIRModule(path, source_hash, version, snapshot, functions[0])
