"""Pinned libtriton adapter for complete TTIR syntax parsing and verification."""

from __future__ import annotations

import hashlib
import importlib
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from ..model import InputError
from ..observability import get_logger, log_event
from .model import TTIRArgument, TTIRModule, TTIROperation, TTIRValue


REQUIRED_TRITON_VERSION = "3.7.1"


LOGGER = get_logger(__name__)

_ATTRIBUTE_NAMES = (
    "axis",
    "start",
    "end",
    "predicate",
    "value",
    "sym_name",
    "sym_visibility",
    "cache",
    "evict",
    "isVolatile",
    "num-warps",
    "num-ctas",
    "ttg.num-warps",
    "ttg.num-ctas",
)

_OP_LINE = re.compile(
    r"^\s*(?:(?:%[-\w.$]+(?:\s*:\s*\d+)?(?:#\d+)?)\s*=\s*)?"
    r"(?P<quoted>\"(?P<qname>[A-Za-z_][\w.]*)\"|(?P<name>[A-Za-z_][\w.]*))\b"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assembly_lines(text: str) -> Dict[str, Deque[str]]:
    """Index single-line operation assembly emitted by libtriton.

    This text is diagnostic metadata. Parsing and verification have already
    happened through MLIR before it is consulted by the semantic lifter.
    """

    result: Dict[str, Deque[str]] = defaultdict(deque)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        match = _OP_LINE.match(line)
        if match is None:
            continue
        name = match.group("qname") or match.group("name")
        if name in {"module", "attributes"}:
            continue
        result[name].append(line)
    return result


def _value(value: Any, stable_id: int) -> TTIRValue:
    location = str(value.get_loc())
    if location == "loc(unknown)":
        location = None
    return TTIRValue(id=stable_id, type=str(value.get_type()), location=location)


class LibTritonParser:
    """Load and verify raw TTIR with the exact supported libtriton version."""

    def __init__(self, required_version: str = REQUIRED_TRITON_VERSION) -> None:
        self.required_version = required_version

    def _runtime(self):
        try:
            triton = importlib.import_module("triton")
            libtriton = importlib.import_module("triton._C.libtriton")
            ir = libtriton.ir
        except (ImportError, OSError, AttributeError) as exc:
            raise InputError(
                "raw TTIR requires Triton/libtriton 3.7.1; install the 'ttir' extra "
                "on Linux or follow docs/dependencies.md for a macOS source build",
                "TTIR_FRONTEND_UNAVAILABLE",
            ) from exc
        version = getattr(triton, "__version__", None)
        if version != self.required_version:
            raise InputError(
                f"libtriton version mismatch: expected {self.required_version}, found {version!r}",
                "TTIR_VERSION_MISMATCH",
            )
        return ir, version

    def parse(self, path: Path, function: Optional[str] = None) -> TTIRModule:
        path = Path(path).resolve()
        log_event(
            LOGGER,
            logging.INFO,
            "ttir_parse_started",
            "starting raw TTIR parsing",
            path=str(path),
            function=function,
            required_version=self.required_version,
        )
        if path.suffix not in {".ttir", ".mlir"}:
            raise InputError(f"raw TTIR input must end in .ttir or .mlir: {path}")
        if not path.is_file():
            raise InputError(f"cannot read {path}: file does not exist", "READ_ERROR")

        ir, version = self._runtime()
        context = ir.context()
        ir.load_dialects(context)
        try:
            module = ir.parse_mlir_module(str(path), context)
        except RuntimeError as exc:
            raise InputError(f"libtriton could not parse {path}: {exc}", "TTIR_PARSE_ERROR") from exc
        module.context = context
        if not module.verify():
            raise InputError(f"libtriton verification failed for {path}", "TTIR_VERIFY_ERROR")

        log_event(
            LOGGER,
            logging.INFO,
            "ttir_module_verified",
            "libtriton parsed and verified the module",
            path=str(path),
            triton_version=version,
        )

        canonical = module.str()
        assembly = _assembly_lines(canonical)
        operations = []
        function_names = []
        value_ids: Dict[int, int] = {}
        block_ids: Dict[int, int] = {}

        def snapshot_value(value: Any) -> TTIRValue:
            raw_id = int(value.id())
            if raw_id not in value_ids:
                value_ids[raw_id] = len(value_ids)
            return _value(value, value_ids[raw_id])

        def visit(operation: Any) -> None:
            name = operation.get_name()
            attributes: Dict[str, Any] = {}
            for attr_name in _ATTRIBUTE_NAMES:
                for getter in (
                    operation.get_int_attr,
                    operation.get_bool_attr,
                    operation.get_str_attr,
                    operation.get_flat_symbol_ref_attr,
                ):
                    value = getter(attr_name)
                    if value is not None:
                        attributes[attr_name] = value
                        break
            if name == "tt.func" and isinstance(attributes.get("sym_name"), str):
                function_names.append(attributes["sym_name"])
            text = assembly[name].popleft() if assembly[name] else None
            block = operation.get_block()
            block_id = 0
            if block is not None:
                raw_block_id = int(block.id())
                if raw_block_id not in block_ids:
                    block_ids[raw_block_id] = len(block_ids) + 1
                block_id = block_ids[raw_block_id]
            operations.append(
                TTIROperation(
                    index=len(operations),
                    name=name,
                    operands=tuple(
                        snapshot_value(operation.get_operand(i))
                        for i in range(operation.get_num_operands())
                    ),
                    results=tuple(snapshot_value(operation.get_result(i)) for i in range(operation.get_num_results())),
                    block_id=block_id,
                    regions=int(operation.get_num_regions()),
                    attributes=attributes,
                    assembly=text,
                )
            )

        module.walk(visit)
        selected = function or module.get_entry_func_name()
        if not selected and len(function_names) == 1:
            selected = function_names[0]
        if not selected:
            raise InputError(
                f"{path} has no unique entry function; set frontends.<side>.function",
                "TTIR_FUNCTION_REQUIRED",
            )
        if not module.has_function(selected):
            raise InputError(f"TTIR function {selected!r} does not exist in {path}", "TTIR_FUNCTION_NOT_FOUND")
        func = module.get_function(selected)
        arguments = tuple(
            TTIRArgument(index=index, value=snapshot_value(func.args(index)))
            for index in range(func.get_num_args())
        )
        result = TTIRModule(
            source=path,
            source_sha256=_sha256(path),
            parser="triton._C.libtriton.ir",
            parser_version=version,
            function=selected,
            arguments=arguments,
            operations=tuple(operations),
            canonical_assembly=canonical,
        )
        log_event(
            LOGGER,
            logging.INFO,
            "ttir_parse_finished",
            "raw TTIR snapshot completed",
            path=str(path),
            function=result.function,
            operations=len(result.operations),
            arguments=len(result.arguments),
        )
        return result


def parse_ttir(path: Path, function: Optional[str] = None) -> TTIRModule:
    return LibTritonParser().parse(path, function=function)
