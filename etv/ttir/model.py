"""Stable, backend-neutral snapshot of a libtriton-parsed TTIR module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TTIRValue:
    id: int
    type: str
    location: Optional[str] = None

    def to_json(self) -> dict:
        value = {"id": self.id, "type": self.type}
        if self.location is not None:
            value["location"] = self.location
        return value


@dataclass(frozen=True)
class TTIRArgument:
    index: int
    value: TTIRValue

    def to_json(self) -> dict:
        return {"index": self.index, "value": self.value.to_json()}


@dataclass(frozen=True)
class TTIROperation:
    index: int
    name: str
    operands: Tuple[TTIRValue, ...]
    results: Tuple[TTIRValue, ...]
    block_id: int
    regions: int
    attributes: Mapping[str, Any]
    assembly: Optional[str] = None

    def to_json(self, include_assembly: bool = True) -> dict:
        value = {
            "index": self.index,
            "name": self.name,
            "operands": [operand.to_json() for operand in self.operands],
            "results": [result.to_json() for result in self.results],
            "block_id": self.block_id,
            "regions": self.regions,
            "attributes": dict(self.attributes),
        }
        if include_assembly and self.assembly is not None:
            value["assembly"] = self.assembly
        return value


@dataclass(frozen=True)
class TTIRModule:
    source: Path
    source_sha256: str
    parser: str
    parser_version: str
    function: str
    arguments: Tuple[TTIRArgument, ...]
    operations: Tuple[TTIROperation, ...]
    canonical_assembly: str

    def to_json(self, include_assembly: bool = True) -> dict:
        value = {
            "format": "etv-ttir-snapshot-v1",
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "function": self.function,
            "arguments": [argument.to_json() for argument in self.arguments],
            "operations": [
                operation.to_json(include_assembly=include_assembly)
                for operation in self.operations
            ],
            "statistics": {
                "arguments": len(self.arguments),
                "operations": len(self.operations),
                "operation_kinds": sorted({operation.name for operation in self.operations}),
            },
        }
        if include_assembly:
            value["canonical_assembly"] = self.canonical_assembly
        return value
