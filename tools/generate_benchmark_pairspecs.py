#!/usr/bin/env python3
"""Generate conservative, manually authored PairSpec-v2 mappings for captured pairs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etv.ttir import parse_ttir


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_numel(description: Any) -> int | None:
    if not isinstance(description, dict) or description.get("kind") != "tensor":
        return None
    shape = description.get("shape")
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in shape
    ):
        return None
    return math.prod(shape)


def _pointer_arguments(module: Any) -> list[int]:
    return [
        argument.index
        for argument in module.arguments
        if "!tt.ptr<" in argument.value.type
    ]


def _argument_origin(module: Any, value_id: int) -> set[int]:
    argument_by_id = {
        argument.value.id: argument.index for argument in module.arguments
    }
    operation_by_result = {
        result.id: operation
        for operation in module.operations
        for result in operation.results
    }
    seen: set[int] = set()

    def visit(current: int) -> set[int]:
        if current in argument_by_id:
            return {argument_by_id[current]}
        if current in seen:
            return set()
        seen.add(current)
        operation = operation_by_result.get(current)
        if operation is None:
            return set()
        origins: set[int] = set()
        for operand in operation.operands:
            origins.update(visit(operand.id))
        return origins

    return visit(value_id)


def _output_argument(module: Any) -> int:
    stores = [operation for operation in module.operations if operation.name == "tt.store"]
    if len(stores) != 1:
        raise ValueError(f"ETV requires one tt.store, found {len(stores)}")
    origins = _argument_origin(module, stores[0].operands[0].id)
    pointer_origins = origins.intersection(_pointer_arguments(module))
    if len(pointer_origins) != 1:
        raise ValueError(
            f"store pointer does not have one physical pointer origin: {sorted(pointer_origins)}"
        )
    return next(iter(pointer_origins))


def _runtime_values(arguments: Any) -> dict[int, Any]:
    if not isinstance(arguments, list):
        return {}
    values: dict[int, Any] = {}
    for item in arguments:
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        value = item.get("value")
        if isinstance(position, int) and not isinstance(position, bool):
            values[position] = value
    return values


def _side_bindings(module: Any, runtime_arguments: Any) -> dict[str, Any]:
    runtime_values = _runtime_values(runtime_arguments)
    bindings: dict[str, Any] = {}
    for argument in module.arguments:
        if "!tt.ptr<" in argument.value.type:
            continue
        value = runtime_values.get(argument.index)
        if isinstance(value, bool) or isinstance(value, int):
            bindings[f"arg{argument.index}"] = value
        elif isinstance(value, float) and math.isfinite(value):
            bindings[f"arg{argument.index}"] = value
    return bindings


def _endpoint(name: str) -> dict[str, str]:
    return {"kind": "block", "name": name}


def _pair_spec(
    *,
    case_dir: Path,
    lhs_path: Path,
    rhs_path: Path,
    lhs_module: Any,
    rhs_module: Any,
    lhs_launch: dict[str, Any],
    rhs_launch: dict[str, Any],
    output_numel: int,
    nodeid: str,
) -> dict[str, Any]:
    lhs_output = _output_argument(lhs_module)
    rhs_output = _output_argument(rhs_module)
    lhs_inputs = [index for index in _pointer_arguments(lhs_module) if index != lhs_output]
    rhs_inputs = [index for index in _pointer_arguments(rhs_module) if index != rhs_output]
    if len(lhs_inputs) != len(rhs_inputs):
        raise ValueError(
            f"input pointer count differs: lhs={len(lhs_inputs)}, rhs={len(rhs_inputs)}"
        )
    lhs_programs = lhs_launch.get("programs")
    rhs_programs = rhs_launch.get("programs")
    if not isinstance(lhs_programs, int) or lhs_programs <= 0:
        raise ValueError(f"invalid LHS program count: {lhs_programs!r}")
    if not isinstance(rhs_programs, int) or rhs_programs <= 0:
        raise ValueError(f"invalid RHS program count: {rhs_programs!r}")
    roles = {
        f"Input{index}": {
            "lhs": _endpoint(f"arg{lhs_index}"),
            "rhs": _endpoint(f"arg{rhs_index}"),
        }
        for index, (lhs_index, rhs_index) in enumerate(
            zip(lhs_inputs, rhs_inputs, strict=True)
        )
    }
    roles["Output"] = {
        "lhs": _endpoint(f"arg{lhs_output}"),
        "rhs": _endpoint(f"arg{rhs_output}"),
    }
    block_roles = list(roles)
    pair_dir = case_dir / "pairspec"
    return {
        "format": "etv-pair-v2",
        "metadata": {
            "pair_id": nodeid,
            "lhs": str(Path("..").joinpath(lhs_path.relative_to(case_dir))),
            "rhs": str(Path("..").joinpath(rhs_path.relative_to(case_dir))),
            "semantic_mode": "abstract_float",
            "frontends": {
                "lhs": {
                    "kind": "ttir",
                    "function": lhs_module.function,
                    "programs": lhs_programs,
                },
                "rhs": {
                    "kind": "ttir",
                    "function": rhs_module.function,
                    "programs": rhs_programs,
                },
            },
            "limits": {
                "max_iterations": 8,
                "max_enodes": 20000,
                "timeout_ms": 10000,
            },
        },
        "assumptions": {
            "for_llm": [
                "This pair was captured from the same specialized ntops pytest node; "
                "pointer roles are mapped by one-store dataflow and remaining ABI order."
            ]
        },
        "predicates": {
            "abi": roles,
            "bindings": {"X": output_numel},
            "side_bindings": {
                "lhs": _side_bindings(
                    lhs_module, lhs_launch.get("runtime_arguments")
                ),
                "rhs": _side_bindings(
                    rhs_module, rhs_launch.get("runtime_arguments")
                ),
            },
            "parameters": {},
            "constraints": [],
            "disjoint": [block_roles] if len(block_roles) > 1 else [],
            "custom": [],
        },
        "observation": {
            "output_role": "Output",
            "output_numel": {"var": "X"},
            "require_full_coverage": True,
            "require_disjoint": block_roles if len(block_roles) > 1 else [],
        },
        "rewrites": [],
    }


def _generate(case_dir: Path) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status: dict[str, Any] = {
        "format": "etv-benchmark-pairspec-status-v1",
        "nodeid": manifest.get("nodeid"),
        "case_dir": str(case_dir),
        "status": "unavailable",
    }
    lhs_launches = [
        item
        for item in manifest.get("lhs_launches", [])
        if item.get("status") == "captured" and item.get("ttir")
    ]
    rhs_references = [
        item
        for item in manifest.get("rhs_references", [])
        if item.get("ttir_status") == "captured"
    ]
    completed_references = [
        item for item in rhs_references if item.get("captured_output") is not None
    ]
    if len(completed_references) == 1:
        # Older captures can contain one incomplete duplicate caused by CPython
        # revisiting the first line of a multiline assignment.
        rhs_references = completed_references
    try:
        if len(lhs_launches) != 1:
            raise ValueError(f"requires one captured LHS launch, found {len(lhs_launches)}")
        if len(rhs_references) != 1:
            raise ValueError(
                f"requires one reference with RHS TTIR, found {len(rhs_references)}"
            )
        reference = rhs_references[0]
        selected = reference.get("selected_ttir", [])
        runtime_launches = reference.get("runtime_launches", [])
        if len(selected) != 1 or len(runtime_launches) != 1:
            raise ValueError(
                "requires one selected RHS TTIR and one runtime launch, found "
                f"{len(selected)} and {len(runtime_launches)}"
            )
        output_numel = _tensor_numel(reference.get("captured_output"))
        if output_numel is None or output_numel <= 0:
            raise ValueError("reference output is not one non-empty tensor")
        lhs_path = Path(lhs_launches[0]["ttir"]).resolve()
        rhs_path = Path(selected[0]["path"]).resolve()
        lhs_module = parse_ttir(lhs_path)
        rhs_module = parse_ttir(rhs_path)
        spec = _pair_spec(
            case_dir=case_dir,
            lhs_path=lhs_path,
            rhs_path=rhs_path,
            lhs_module=lhs_module,
            rhs_module=rhs_module,
            lhs_launch=lhs_launches[0],
            rhs_launch=runtime_launches[0],
            output_numel=output_numel,
            nodeid=manifest["nodeid"],
        )
        spec_path = case_dir / "pairspec" / "pair.json"
        _write_json(spec_path, spec)
        status.update({"status": "generated", "pairspec": str(spec_path)})
    except Exception as error:
        status["reason"] = str(error)
        (case_dir / "pairspec" / "pair.json").unlink(missing_ok=True)
    _write_json(case_dir / "pairspec" / "status.json", status)
    return status


def _manifest_unavailable(case_dir: Path, nodeid: str | None) -> dict[str, Any]:
    status = {
        "format": "etv-benchmark-pairspec-status-v1",
        "nodeid": nodeid,
        "case_dir": str(case_dir),
        "status": "unavailable",
        "reason": "capture manifest unavailable (test did not reach capture hook)",
    }
    (case_dir / "pairspec" / "pair.json").unlink(missing_ok=True)
    _write_json(case_dir / "pairspec" / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve()
    collection_summary_path = benchmark_root / "collection-summary.json"
    if collection_summary_path.exists():
        collection_summary = json.loads(
            collection_summary_path.read_text(encoding="utf-8")
        )
        cases = sorted(
            (
                Path(record["case_dir"]).resolve(),
                record.get("nodeid"),
            )
            for record in collection_summary.get("records", [])
        )
    else:
        cases = [
            (path.parent, None)
            for path in sorted(benchmark_root.glob("*/*/manifest.json"))
        ]
    records = [
        (
            _generate(case_dir)
            if (case_dir / "manifest.json").exists()
            else _manifest_unavailable(case_dir, nodeid)
        )
        for case_dir, nodeid in cases
    ]
    counts = Counter(record["status"] for record in records)
    summary = {
        "format": "etv-benchmark-pairspec-summary-v1",
        "counts": dict(sorted(counts.items())),
        "records": records,
    }
    _write_json(benchmark_root / "pairspec-summary.json", summary)
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
