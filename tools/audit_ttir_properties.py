#!/usr/bin/env python3
"""Audit store and loop syntax in the captured 75-operator TTIR corpus.

This is deliberately a syntactic audit.  It establishes facts such as the
number of store operations and the shape of SCF loops, but it does not claim
that distinct dynamic store instances have distinct addresses.  That latter
property requires the address/mask proof obligations described in
docs/formal_program_properties_research.md.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable


SSA = r"%[-a-zA-Z$._0-9]+"
FOR_RE = re.compile(
    rf"\bscf\.for\s+(?:unsigned\s+)?(?P<iv>{SSA})\s*=\s*"
    rf"(?P<lb>{SSA})\s+to\s+(?P<ub>{SSA})\s+step\s+(?P<step>{SSA})\b"
)
RESULT_RE = re.compile(rf"^\s*(?P<result>{SSA})(?::\d+)?\s*=\s*(?P<rhs>.*)$")
SSA_RE = re.compile(SSA)
INT_CONSTANT_RE = re.compile(
    r"\barith\.constant\s+(?P<value>-?(?:0x[0-9a-fA-F]+|\d+))\s*:\s*(?:i\d+|index)\b"
)
INT_BINARY_RE = re.compile(
    rf"\barith\.(?P<op>addi|subi|muli|divsi|divui|ceildivsi|ceildivui|floordivsi|"
    rf"remsi|remui|maxsi|maxui|minsi|minui)\s+(?P<a>{SSA})\s*,\s*(?P<b>{SSA})\b"
)
INT_CAST_RE = re.compile(
    rf"\barith\.(?:index_cast|index_castui|extsi|extui|trunci)\s+(?P<a>{SSA})\b"
)


@dataclass(frozen=True)
class LoopRecord:
    line: int
    lower: str
    upper: str
    step: str
    classification: str
    trip_count: int | None


@dataclass(frozen=True)
class FileRecord:
    path: str
    operator: str
    side: str
    sha256: str
    stores: int
    stores_in_scf_for: int
    atomics: int
    scf_for: int
    scf_while: int
    scf_parallel: int
    loops: tuple[LoopRecord, ...]


@dataclass(frozen=True)
class Definition:
    op: str
    operands: tuple[str, ...]
    constant: int | None = None


def _integer_definition(rhs: str) -> Definition | None:
    constant_match = INT_CONSTANT_RE.search(rhs)
    if constant_match:
        return Definition("constant", (), int(constant_match.group("value"), 0))

    binary_match = INT_BINARY_RE.search(rhs)
    if binary_match:
        return Definition(
            binary_match.group("op"),
            (binary_match.group("a"), binary_match.group("b")),
        )

    cast_match = INT_CAST_RE.search(rhs)
    if cast_match:
        return Definition("cast", (cast_match.group("a"),))

    operands = tuple(SSA_RE.findall(rhs))
    return Definition("unknown", operands) if operands else None


def _trunc_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)


def _evaluate(name: str, definitions: dict[str, Definition], seen: set[str] | None = None) -> int | None:
    seen = set() if seen is None else seen
    if name in seen:
        return None
    definition = definitions.get(name)
    if definition is None:
        return None
    if definition.op == "constant":
        return definition.constant

    next_seen = seen | {name}
    values = [_evaluate(operand, definitions, next_seen) for operand in definition.operands]
    if any(value is None for value in values):
        return None
    ints = [int(value) for value in values if value is not None]
    try:
        if definition.op == "cast":
            return ints[0]
        a, b = ints
        if definition.op == "addi":
            return a + b
        if definition.op == "subi":
            return a - b
        if definition.op == "muli":
            return a * b
        if definition.op in {"divsi", "divui"}:
            return _trunc_div(a, b)
        if definition.op in {"ceildivsi", "ceildivui"}:
            return -((-a) // b)
        if definition.op == "floordivsi":
            return a // b
        if definition.op in {"remsi", "remui"}:
            return a - _trunc_div(a, b) * b
        if definition.op in {"maxsi", "maxui"}:
            return max(a, b)
        if definition.op in {"minsi", "minui"}:
            return min(a, b)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _trip_count(lower: int, upper: int, step: int) -> int | None:
    if step <= 0:
        return None
    return max(0, (upper - lower + step - 1) // step)


def _side(path: Path) -> str:
    if "lhs" in path.parts:
        return "lhs"
    if "rhs" in path.parts:
        return "rhs"
    return "other"


def audit_file(path: Path, root: Path) -> FileRecord:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    relative = path.relative_to(root)
    operator = relative.parts[0]

    definitions: dict[str, Definition] = {}
    loops: list[LoopRecord] = []
    brace_depth = 0
    loop_start_depths: list[int] = []
    stores = 0
    stores_in_scf_for = 0

    for line_number, line in enumerate(lines, start=1):
        loop_match = FOR_RE.search(line)
        if loop_match:
            lower_name = loop_match.group("lb")
            upper_name = loop_match.group("ub")
            step_name = loop_match.group("step")
            lower = _evaluate(lower_name, definitions)
            upper = _evaluate(upper_name, definitions)
            step = _evaluate(step_name, definitions)
            trip_count = (
                _trip_count(lower, upper, step)
                if lower is not None and upper is not None and step is not None
                else None
            )
            if trip_count is not None:
                classification = "fixed"
            elif step is not None and step > 0:
                classification = "symbolic_explicit_bound"
            elif step is not None:
                classification = "invalid_nonpositive_step"
            else:
                classification = "symbolic_step"
            loops.append(
                LoopRecord(
                    line=line_number,
                    lower=lower_name,
                    upper=upper_name,
                    step=step_name,
                    classification=classification,
                    trip_count=trip_count,
                )
            )
            loop_start_depths.append(brace_depth)

        if re.search(r"\btt\.store\b", line):
            stores += 1
            if loop_start_depths:
                stores_in_scf_for += 1

        result_match = RESULT_RE.match(line)
        if result_match and not loop_match:
            definition = _integer_definition(result_match.group("rhs"))
            if definition is not None:
                definitions[result_match.group("result")] = definition

        brace_depth += line.count("{") - line.count("}")
        while loop_start_depths and brace_depth <= loop_start_depths[-1]:
            loop_start_depths.pop()

    return FileRecord(
        path=str(relative),
        operator=operator,
        side=_side(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        stores=stores,
        stores_in_scf_for=stores_in_scf_for,
        atomics=len(re.findall(r"\btt\.atomic[_a-zA-Z0-9.]*\b", text)),
        scf_for=len(loops),
        scf_while=len(re.findall(r"\bscf\.while\b", text)),
        scf_parallel=len(re.findall(r"\bscf\.(?:parallel|forall)\b", text)),
        loops=tuple(loops),
    )


def _count_by(records: Iterable[FileRecord], field: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(record, field) for record in records).items()))


def _operator_summary(records: list[FileRecord]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        grouped[record.operator].append(record)

    result: dict[str, dict[str, object]] = {}
    for operator, items in sorted(grouped.items()):
        loop_classes = Counter(
            loop.classification for item in items for loop in item.loops
        )
        result[operator] = {
            "files": len(items),
            "lhs_files": sum(item.side == "lhs" for item in items),
            "rhs_files": sum(item.side == "rhs" for item in items),
            "single_store_files": sum(item.stores == 1 for item in items),
            "multi_store_files": sum(item.stores > 1 for item in items),
            "max_stores_per_file": max(item.stores for item in items),
            "store_ops": sum(item.stores for item in items),
            "store_ops_in_scf_for": sum(item.stores_in_scf_for for item in items),
            "atomic_ops": sum(item.atomics for item in items),
            "files_with_scf_for": sum(item.scf_for > 0 for item in items),
            "scf_for_ops": sum(item.scf_for for item in items),
            "loop_classes": dict(sorted(loop_classes.items())),
            "scf_while_ops": sum(item.scf_while for item in items),
            "scf_parallel_or_forall_ops": sum(item.scf_parallel for item in items),
        }
    return result


def _validate_with_libtriton(
    records: list[FileRecord], benchmark: Path
) -> dict[str, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from etv.ttir.libtriton import REQUIRED_TRITON_VERSION, parse_ttir

    errors: list[dict[str, str]] = []
    mismatches: list[dict[str, object]] = []
    operation_counts: Counter[str] = Counter()
    for record in records:
        path = benchmark / record.path
        try:
            module = parse_ttir(path)
        except Exception as exc:  # The audit must report every invalid artifact.
            errors.append(
                {
                    "path": record.path,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        names = [operation.name for operation in module.operations]
        actual = {
            "tt.store": names.count("tt.store"),
            "scf.for": names.count("scf.for"),
            "scf.while": names.count("scf.while"),
            "scf.parallel_or_forall": names.count("scf.parallel")
            + names.count("scf.forall"),
            "tt.atomic": sum(name.startswith("tt.atomic") for name in names),
        }
        expected = {
            "tt.store": record.stores,
            "scf.for": record.scf_for,
            "scf.while": record.scf_while,
            "scf.parallel_or_forall": record.scf_parallel,
            "tt.atomic": record.atomics,
        }
        operation_counts.update(actual)
        if actual != expected:
            mismatches.append(
                {"path": record.path, "libtriton": actual, "syntactic": expected}
            )

    return {
        "enabled": True,
        "required_triton_version": REQUIRED_TRITON_VERSION,
        "files_requested": len(records),
        "files_parsed_and_verified": len(records) - len(errors),
        "parse_or_verify_errors": errors,
        "count_mismatches": mismatches,
        "operation_counts": dict(sorted(operation_counts.items())),
    }


def build_report(
    benchmark: Path,
    specs: Path,
    *,
    include_file_records: bool = False,
    verify_libtriton: bool = False,
) -> dict[str, object]:
    expected_operators = sorted(path.stem for path in specs.glob("*.yaml"))
    benchmark_operators = sorted(path.name for path in benchmark.iterdir() if path.is_dir())
    included = sorted(set(expected_operators) & set(benchmark_operators))
    records = [
        audit_file(path, benchmark)
        for operator in included
        for path in sorted((benchmark / operator).rglob("*.ttir"))
    ]
    corpus_fingerprint = hashlib.sha256(
        "".join(f"{record.path}\0{record.sha256}\n" for record in records).encode()
    ).hexdigest()
    loop_classes = Counter(loop.classification for record in records for loop in record.loops)
    fixed_trip_counts = Counter(
        loop.trip_count
        for record in records
        for loop in record.loops
        if loop.trip_count is not None
    )
    store_distribution = Counter(record.stores for record in records)

    report: dict[str, object] = {
        "scope": {
            "benchmark": str(benchmark),
            "specs": str(specs),
            "expected_operator_count": len(expected_operators),
            "benchmark_operator_count": len(benchmark_operators),
            "included_operator_count": len(included),
            "included_operators": included,
            "manifest_files": sum(
                1
                for operator in included
                for _ in (benchmark / operator).glob("*/manifest.json")
            ),
            "benchmark_only_operators": sorted(set(benchmark_operators) - set(expected_operators)),
            "spec_only_operators": sorted(set(expected_operators) - set(benchmark_operators)),
        },
        "totals": {
            "ttir_files": len(records),
            "corpus_fingerprint_sha256": corpus_fingerprint,
            "distinct_file_sha256": len({record.sha256 for record in records}),
            "files_by_side": _count_by(records, "side"),
            "files_with_exactly_one_store": sum(record.stores == 1 for record in records),
            "files_with_multiple_stores": sum(record.stores > 1 for record in records),
            "max_stores_per_file": max(record.stores for record in records),
            "store_count_distribution": {
                str(count): files for count, files in sorted(store_distribution.items())
            },
            "store_ops": sum(record.stores for record in records),
            "store_ops_in_scf_for": sum(record.stores_in_scf_for for record in records),
            "atomic_ops": sum(record.atomics for record in records),
            "files_with_scf_for": sum(record.scf_for > 0 for record in records),
            "scf_for_ops": sum(record.scf_for for record in records),
            "loop_classes": dict(sorted(loop_classes.items())),
            "fixed_trip_count_distribution": {
                str(count): loops for count, loops in sorted(fixed_trip_counts.items())
            },
            "scf_while_ops": sum(record.scf_while for record in records),
            "scf_parallel_or_forall_ops": sum(record.scf_parallel for record in records),
        },
        "operators": _operator_summary(records),
        "witnesses": {
            "multi_store_files": [
                asdict(record) for record in records if record.stores > 1
            ][:5],
            "loop_files": [
                asdict(record) for record in records if record.scf_for > 0
            ][:5],
        },
        "limitations": [
            "Aggregate classification is syntactic; --verify-libtriton additionally parses/verifies every module and cross-checks operation counts.",
            "A single tt.store can write many lanes and execute in many programs/iterations.",
            "Multiple tt.store operations may write disjoint outputs or slices.",
            "Dynamic single-writer requires SMT proof over addresses, masks, program IDs, lanes, loop iterations, and aliases.",
            "A symbolic_explicit_bound scf.for has an explicit SSA upper bound and a statically positive step, but not a compile-time-fixed trip count.",
        ],
    }
    if include_file_records:
        report["file_records"] = [asdict(record) for record in records]
    report["libtriton_validation"] = (
        _validate_with_libtriton(records, benchmark)
        if verify_libtriton
        else {"enabled": False}
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmark",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "specs" / "ntops" / "operators",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-file-records",
        action="store_true",
        help="include all per-file records instead of only aggregate data and witnesses",
    )
    parser.add_argument(
        "--verify-libtriton",
        action="store_true",
        help="parse and verify every TTIR and cross-check operation counts (requires Triton 3.7.1)",
    )
    args = parser.parse_args()

    report = build_report(
        args.benchmark.resolve(),
        args.specs.resolve(),
        include_file_records=args.include_file_records,
        verify_libtriton=args.verify_libtriton,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
