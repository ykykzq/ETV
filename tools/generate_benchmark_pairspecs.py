#!/usr/bin/env python3
"""Generate provenance-aware PairSpec-v2 observations for captured TTIR pairs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etv.ttir import parse_ttir


@dataclass(frozen=True)
class PointerInfo:
    argument: int
    description: dict[str, Any] | None

    @property
    def name(self) -> str:
        return f"arg{self.argument}"

    @property
    def provenance(self) -> dict[str, Any]:
        if not isinstance(self.description, dict):
            return {}
        value = self.description.get("provenance")
        return value if isinstance(value, dict) else {}

    @property
    def tensor_id(self) -> str | None:
        value = self.provenance.get("tensor_id")
        return value if isinstance(value, str) else None

    @property
    def storage_id(self) -> str | None:
        value = self.provenance.get("storage_id")
        return value if isinstance(value, str) else None

    @property
    def storage_offset(self) -> int:
        value = self.provenance.get("storage_offset")
        if not isinstance(value, int) or isinstance(value, bool):
            value = (self.description or {}).get("storage_offset", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def runtime_role(self) -> str | None:
        value = self.provenance.get("runtime_role")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class StoreInfo:
    index: int
    output: PointerInfo
    dependencies: tuple[PointerInfo, ...]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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


def _argument_origins(module: Any, value_ids: list[int]) -> set[int]:
    argument_by_id = {
        argument.value.id: argument.index for argument in module.arguments
    }
    operation_by_result = {
        result.id: operation
        for operation in module.operations
        for result in operation.results
    }

    def visit(current: int, seen: set[int]) -> set[int]:
        if current in argument_by_id:
            return {argument_by_id[current]}
        if current in seen:
            return set()
        seen.add(current)
        operation = operation_by_result.get(current)
        if operation is None:
            return set()
        result: set[int] = set()
        for operand in operation.operands:
            result.update(visit(operand.id, seen))
        return result

    origins: set[int] = set()
    for value_id in value_ids:
        origins.update(visit(value_id, set()))
    return origins


def _runtime_values(arguments: Any) -> list[dict[str, Any]]:
    if not isinstance(arguments, list):
        return []
    return [item for item in arguments if isinstance(item, dict)]


def _runtime_tensor_values(arguments: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _runtime_values(arguments)
        if isinstance(item.get("value"), dict) and item["value"].get("kind") == "tensor"
    ]


def _pointer_runtime_map(
    module: Any, launch: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    pointers = _pointer_arguments(module)
    runtime_tensors = _runtime_tensor_values(launch.get("runtime_arguments"))
    explicitly_mapped = {
        item.get("ttir_argument_index"): item["value"]
        for item in runtime_tensors
        if isinstance(item.get("ttir_argument_index"), int)
    }
    if set(explicitly_mapped) == set(pointers):
        return explicitly_mapped
    if len(pointers) != len(runtime_tensors):
        raise ValueError(
            "cannot align TTIR pointer arguments with runtime tensors: "
            f"ttir={len(pointers)}, runtime={len(runtime_tensors)}"
        )
    return {
        pointer: runtime["value"]
        for pointer, runtime in zip(pointers, runtime_tensors, strict=True)
    }


def _store_infos(module: Any, launch: dict[str, Any]) -> list[StoreInfo]:
    runtime = _pointer_runtime_map(module, launch)
    pointer_arguments = set(_pointer_arguments(module))
    stores = [
        operation for operation in module.operations if operation.name == "tt.store"
    ]
    result: list[StoreInfo] = []
    for store_index, store in enumerate(stores):
        output_origins = _argument_origins(module, [store.operands[0].id])
        output_pointers = sorted(output_origins.intersection(pointer_arguments))
        if len(output_pointers) != 1:
            raise ValueError(
                f"store {store_index} pointer has {len(output_pointers)} physical origins"
            )
        output_argument = output_pointers[0]
        dependency_origins = _argument_origins(
            module,
            [operand.id for operand in store.operands[1:]],
        )
        dependency_arguments = sorted(
            dependency_origins.intersection(pointer_arguments)
        )
        if not dependency_arguments and any(
            operation.regions for operation in module.operations
        ):
            # libtriton snapshots expose region counts but not every nested SSA
            # edge. Preserve the captured input provenance for an auditable
            # PairSpec; lifting will still honestly report unsupported region
            # semantics instead of claiming a proof.
            dependency_arguments = sorted(
                argument
                for argument, description in runtime.items()
                if argument != output_argument
                and PointerInfo(argument, description).runtime_role == "input"
            )
        result.append(
            StoreInfo(
                index=store_index,
                output=PointerInfo(output_argument, runtime.get(output_argument)),
                dependencies=tuple(
                    PointerInfo(argument, runtime.get(argument))
                    for argument in dependency_arguments
                ),
            )
        )
    return result


def _pointer_inventory(
    module: Any,
    launch: dict[str, Any],
    stores: list[StoreInfo],
    selected_store: StoreInfo | None = None,
) -> list[dict[str, Any]]:
    runtime = _pointer_runtime_map(module, launch)
    store_outputs = {store.output.argument for store in stores}
    store_dependencies = {
        pointer.argument for store in stores for pointer in store.dependencies
    }
    selected_dependencies = (
        {pointer.argument for pointer in selected_store.dependencies}
        if selected_store is not None
        else set()
    )
    result: list[dict[str, Any]] = []
    for argument in _pointer_arguments(module):
        pointer = PointerInfo(argument, runtime.get(argument))
        if selected_store is not None and argument == selected_store.output.argument:
            semantic_role = "selected_output"
        elif argument in selected_dependencies:
            semantic_role = (
                "selected_scratch_or_intermediate_dependency"
                if pointer.runtime_role in {"scratch", "unclassified_output_or_scratch"}
                else "selected_input_dependency"
            )
        elif argument in store_outputs:
            semantic_role = (
                "store_output" if selected_store is None else "other_store_output"
            )
        elif argument in store_dependencies:
            semantic_role = (
                "scratch_or_intermediate_store_dependency"
                if pointer.runtime_role in {"scratch", "unclassified_output_or_scratch"}
                else (
                    "store_dependency"
                    if selected_store is None
                    else "other_store_dependency"
                )
            )
        else:
            semantic_role = "unused_by_store_values"
        result.append(
            {
                "argument": argument,
                "name": pointer.name,
                "semantic_role": semantic_role,
                "runtime_role": pointer.runtime_role,
                "tensor_id": pointer.tensor_id,
                "storage_id": pointer.storage_id,
                "alias_group": pointer.provenance.get("alias_group"),
                "storage_offset": pointer.storage_offset,
                "byte_offset": pointer.provenance.get("byte_offset"),
                "output_leaf_indices": pointer.provenance.get(
                    "output_leaf_indices", []
                ),
            }
        )
    return result


def _store_inventory(stores: list[StoreInfo]) -> list[dict[str, Any]]:
    return [
        {
            "store_index": store.index,
            "output_argument": store.output.argument,
            "dependency_arguments": [
                pointer.argument for pointer in store.dependencies
            ],
        }
        for store in stores
    ]


def _description_signature(description: Any) -> tuple[Any, ...] | None:
    if not isinstance(description, dict) or description.get("kind") != "tensor":
        return None
    return (
        description.get("dtype"),
        tuple(description.get("shape", [])),
        tuple(description.get("stride", [])),
    )


def _flatten_tensor_descriptions(
    value: Any, path: str = "reference_output"
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    if value.get("kind") == "tensor":
        return [{"leaf_index": 0, "path": path, "value": value}]
    items = value.get("items")
    result: list[dict[str, Any]] = []
    if isinstance(items, list):
        for index, item in enumerate(items):
            result.extend(_flatten_tensor_descriptions(item, f"{path}[{index}]"))
    elif isinstance(items, dict):
        for key, item in items.items():
            result.extend(_flatten_tensor_descriptions(item, f"{path}[{key!r}]"))
    for index, item in enumerate(result):
        item["leaf_index"] = index
    return result


def _output_leaves(reference: dict[str, Any]) -> list[dict[str, Any]]:
    leaves = reference.get("output_leaves")
    if isinstance(leaves, list) and all(isinstance(item, dict) for item in leaves):
        return leaves
    return _flatten_tensor_descriptions(reference.get("captured_output"))


def _leaf_for_rhs_store(
    store: StoreInfo,
    leaves: list[dict[str, Any]],
    stores: list[StoreInfo],
) -> dict[str, Any]:
    declared = store.output.provenance.get("output_leaf_indices")
    if isinstance(declared, list):
        matches = [leaf for leaf in leaves if leaf.get("leaf_index") in declared]
        if len(matches) == 1:
            return matches[0]
    storage_id = store.output.storage_id
    if storage_id is not None:
        matches = [
            leaf
            for leaf in leaves
            if PointerInfo(-1, leaf.get("value")).storage_id == storage_id
        ]
        if len(matches) == 1:
            return matches[0]
    if len(stores) == len(leaves):
        return leaves[store.index]
    raise ValueError(f"cannot associate RHS store {store.index} with one output leaf")


def _lhs_store_for_leaf(
    lhs_stores: list[StoreInfo],
    rhs_store: StoreInfo,
    leaf: dict[str, Any],
    rhs_stores: list[StoreInfo],
) -> StoreInfo:
    leaf_index = leaf.get("leaf_index")
    declared_matches = [
        store
        for store in lhs_stores
        if leaf_index in (store.output.provenance.get("output_leaf_indices") or [])
    ]
    if len(declared_matches) == 1:
        return declared_matches[0]
    if len(lhs_stores) == len(rhs_stores):
        return lhs_stores[rhs_store.index]
    signature = _description_signature(leaf.get("value"))
    matches = [
        store
        for store in lhs_stores
        if _description_signature(store.output.description) == signature
    ]
    if len(matches) == 1:
        return matches[0]
    if len(lhs_stores) == 1:
        return lhs_stores[0]
    raise ValueError(
        f"cannot associate output leaf {leaf.get('leaf_index')} with one LHS store"
    )


def _pair_dependencies(
    lhs_store: StoreInfo,
    rhs_store: StoreInfo,
) -> list[tuple[PointerInfo, PointerInfo]]:
    lhs_inputs = [
        item
        for item in lhs_store.dependencies
        if item.argument != lhs_store.output.argument
    ]
    rhs_inputs = [
        item
        for item in rhs_store.dependencies
        if item.argument != rhs_store.output.argument
    ]
    has_provenance = any(
        item.storage_id is not None or item.tensor_id is not None
        for item in (*lhs_inputs, *rhs_inputs)
    )
    if has_provenance:
        pairs: list[tuple[PointerInfo, PointerInfo]] = []
        remaining_lhs = list(lhs_inputs)
        remaining_rhs = list(rhs_inputs)

        # Exact tensor identities are strongest. A view reconstructed from the
        # same storage can legitimately have a different tensor identity, so
        # unmatched entries fall through to storage-level matching.
        for lhs in list(remaining_lhs):
            if lhs.tensor_id is None:
                continue
            matches = [rhs for rhs in remaining_rhs if rhs.tensor_id == lhs.tensor_id]
            if len(matches) == 1:
                rhs = matches[0]
                pairs.append((lhs, rhs))
                remaining_lhs.remove(lhs)
                remaining_rhs.remove(rhs)

        lhs_by_storage: dict[str, list[PointerInfo]] = {}
        rhs_by_storage: dict[str, list[PointerInfo]] = {}
        for item in remaining_lhs:
            if item.storage_id is not None:
                lhs_by_storage.setdefault(item.storage_id, []).append(item)
        for item in remaining_rhs:
            if item.storage_id is not None:
                rhs_by_storage.setdefault(item.storage_id, []).append(item)
        for storage_id in sorted(set(lhs_by_storage).intersection(rhs_by_storage)):
            lhs_group = lhs_by_storage[storage_id]
            rhs_group = rhs_by_storage[storage_id]
            for lhs, rhs in zip(lhs_group, rhs_group):
                pairs.append((lhs, rhs))
                remaining_lhs.remove(lhs)
                remaining_rhs.remove(rhs)
    else:
        remaining_rhs = list(rhs_inputs)
        pairs = []
        for lhs in lhs_inputs:
            signature = _description_signature(lhs.description)
            matches = [
                rhs
                for rhs in remaining_rhs
                if _description_signature(rhs.description) == signature
            ]
            if matches:
                rhs = matches[0]
                pairs.append((lhs, rhs))
                remaining_rhs.remove(rhs)

    lhs_output_alias = {
        index
        for index, (lhs, _) in enumerate(pairs)
        if lhs.storage_id is not None and lhs.storage_id == lhs_store.output.storage_id
    }
    rhs_output_alias = {
        index
        for index, (_, rhs) in enumerate(pairs)
        if rhs.storage_id is not None and rhs.storage_id == rhs_store.output.storage_id
    }
    if lhs_output_alias != rhs_output_alias:
        raise ValueError(
            "input/output alias relation differs between sides: "
            f"lhs={sorted(lhs_output_alias)}, rhs={sorted(rhs_output_alias)}"
        )
    return pairs


def _scalar_compatible(type_text: str, value: Any) -> bool:
    if isinstance(value, bool):
        return "i1" in type_text
    if isinstance(value, int):
        return any(token in type_text for token in ("i8", "i16", "i32", "i64", "index"))
    if isinstance(value, float):
        return any(token in type_text for token in ("f16", "f32", "f64", "bf16"))
    return False


def _specialized_away(value: Any) -> bool:
    """Return whether Triton folded a runtime scalar into a constexpr.

    Triton specializes integer arguments equal to 1 into constexprs, removing
    them from the compiled TTIR signature (for example contiguous strides).
    Such values must not consume a slot when aligning TTIR scalar arguments
    with the wrapper-level runtime argument list.
    """
    return type(value) is int and value == 1


def _side_bindings(module: Any, runtime_arguments: Any) -> dict[str, Any]:
    scalar_arguments = [
        argument
        for argument in module.arguments
        if "!tt.ptr<" not in argument.value.type
    ]
    runtime_scalars = [
        item.get("value")
        for item in _runtime_values(runtime_arguments)
        if isinstance(item.get("value"), (bool, int, float))
        and not _specialized_away(item.get("value"))
    ]
    bindings: dict[str, Any] = {}
    cursor = 0
    for argument in scalar_arguments:
        while cursor < len(runtime_scalars) and not _scalar_compatible(
            argument.value.type, runtime_scalars[cursor]
        ):
            cursor += 1
        if cursor >= len(runtime_scalars):
            continue
        value = runtime_scalars[cursor]
        cursor += 1
        # PairSpec side bindings are integer launch/shape facts. Floating
        # runtime scalars require an explicit cross-side scalar ABI role.
        if isinstance(value, (bool, int)):
            bindings[f"arg{argument.index}"] = value
    return bindings


def _endpoint(pointer: PointerInfo) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": "block", "name": pointer.name}
    if pointer.storage_offset:
        result["offset"] = pointer.storage_offset
    return result


def _disjoint_groups(
    roles: dict[str, tuple[PointerInfo, PointerInfo]],
) -> list[list[str]]:
    result: list[list[str]] = []
    for lhs_name, rhs_name in combinations(sorted(roles), 2):
        lhs_pair = roles[lhs_name]
        rhs_pair = roles[rhs_name]
        known_disjoint = all(
            first.storage_id is not None
            and second.storage_id is not None
            and first.storage_id != second.storage_id
            for first, second in zip(lhs_pair, rhs_pair, strict=True)
        )
        if known_disjoint:
            result.append([lhs_name, rhs_name])
    return result


def _pair_spec(
    *,
    case_dir: Path,
    lhs_path: Path,
    rhs_path: Path,
    lhs_module: Any,
    rhs_module: Any,
    lhs_launch: dict[str, Any],
    rhs_launch: dict[str, Any],
    lhs_store: StoreInfo,
    rhs_store: StoreInfo,
    leaf: dict[str, Any],
    nodeid: str,
    pair_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_numel = _tensor_numel(leaf.get("value"))
    if output_numel is None or output_numel <= 0:
        raise ValueError(
            f"output leaf {leaf.get('leaf_index')} is empty or not a tensor"
        )
    lhs_programs = lhs_launch.get("programs")
    rhs_programs = rhs_launch.get("programs")
    if not isinstance(lhs_programs, int) or lhs_programs <= 0:
        raise ValueError(f"invalid LHS program count: {lhs_programs!r}")
    if not isinstance(rhs_programs, int) or rhs_programs <= 0:
        raise ValueError(f"invalid RHS program count: {rhs_programs!r}")

    dependencies = _pair_dependencies(lhs_store, rhs_store)
    role_pointers: dict[str, tuple[PointerInfo, PointerInfo]] = {
        f"Input{index}": pair for index, pair in enumerate(dependencies)
    }
    role_pointers["Output"] = (lhs_store.output, rhs_store.output)
    roles = {
        logical: {"lhs": _endpoint(lhs), "rhs": _endpoint(rhs)}
        for logical, (lhs, rhs) in role_pointers.items()
    }
    disjoint = _disjoint_groups(role_pointers)
    spec = {
        "format": "etv-pair-v2",
        "metadata": {
            "pair_id": pair_id,
            "lhs": str(Path("../..").joinpath(lhs_path.relative_to(case_dir))),
            "rhs": str(Path("../..").joinpath(rhs_path.relative_to(case_dir))),
            "semantic_mode": "abstract_float",
            "frontends": {
                "lhs": {
                    "kind": "ttir",
                    "function": lhs_module.function,
                    "programs": lhs_programs,
                    "store_index": lhs_store.index,
                },
                "rhs": {
                    "kind": "ttir",
                    "function": rhs_module.function,
                    "programs": rhs_programs,
                    "store_index": rhs_store.index,
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
                "Both kernels were captured from the same specialized pytest node.",
                "Input roles are aligned by captured tensor/storage provenance when "
                "available, otherwise by exact runtime tensor signatures; endpoint "
                "offsets normalize tensor views to their logical storage bases.",
                "Selected-store dependencies without a reliable counterpart are left "
                "unmapped and must prevent a proof if they affect the observation.",
                f"This PairSpec observes output leaf {leaf.get('path')!r} only.",
            ]
        },
        "predicates": {
            "abi": roles,
            "bindings": {"X": output_numel},
            "side_bindings": {
                "lhs": _side_bindings(lhs_module, lhs_launch.get("runtime_arguments")),
                "rhs": _side_bindings(rhs_module, rhs_launch.get("runtime_arguments")),
            },
            "parameters": {},
            "constraints": [],
            "disjoint": disjoint,
            "custom": [],
        },
        "observation": {
            "output_role": "Output",
            "output_numel": {"var": "X"},
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "rewrites": [],
    }
    mapping = {
        "nodeid": nodeid,
        "output_leaf": leaf,
        "lhs_store_index": lhs_store.index,
        "rhs_store_index": rhs_store.index,
        "pointer_inventory": {
            "lhs": _pointer_inventory(lhs_module, lhs_launch, [lhs_store], lhs_store),
            "rhs": _pointer_inventory(rhs_module, rhs_launch, [rhs_store], rhs_store),
        },
        "unmapped_selected_dependencies": {
            "lhs": sorted(
                {pointer.argument for pointer in lhs_store.dependencies}
                - {pointer.argument for pointer, _ in dependencies}
            ),
            "rhs": sorted(
                {pointer.argument for pointer in rhs_store.dependencies}
                - {pointer.argument for _, pointer in dependencies}
            ),
        },
        "roles": {
            logical: {
                "lhs_argument": lhs.argument,
                "rhs_argument": rhs.argument,
                "lhs_tensor_id": lhs.tensor_id,
                "rhs_tensor_id": rhs.tensor_id,
                "lhs_storage_id": lhs.storage_id,
                "rhs_storage_id": rhs.storage_id,
                "lhs_storage_offset": lhs.storage_offset,
                "rhs_storage_offset": rhs.storage_offset,
                "lhs_runtime_role": lhs.runtime_role,
                "rhs_runtime_role": rhs.runtime_role,
            }
            for logical, (lhs, rhs) in role_pointers.items()
        },
        "disjoint": disjoint,
    }
    return spec, mapping


def _runtime_ttir_launches(reference: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = reference.get("runtime_launches", [])
    selected = reference.get("selected_ttir", [])
    if not isinstance(runtime, list) or not all(
        isinstance(item, dict) for item in runtime
    ):
        return []
    result = []
    for index, launch in enumerate(runtime):
        path = launch.get("ttir")
        if not isinstance(path, str) and len(selected) == len(runtime):
            selected_item = selected[index]
            path = (
                selected_item.get("path") if isinstance(selected_item, dict) else None
            )
        if not isinstance(path, str):
            raise ValueError(
                f"cannot associate runtime launch {index} with one selected TTIR"
            )
        result.append({**launch, "_ttir_path": path})
    return result


def _storage_key(pointer: PointerInfo) -> str:
    if pointer.storage_id is None:
        raise ValueError(
            "multi-launch slicing requires capture-v2 storage provenance for every "
            f"selected pointer (arg{pointer.argument})"
        )
    return pointer.storage_id


def _qualified_endpoint(pointer: PointerInfo, launch_id: str) -> dict[str, Any]:
    result = _endpoint(pointer)
    result["name"] = f"{launch_id}::{result['name']}"
    return result


def _multilaunch_pair_spec(
    *,
    case_dir: Path,
    lhs_path: Path,
    lhs_module: Any,
    lhs_launch: dict[str, Any],
    lhs_store: StoreInfo,
    rhs_nodes: list[tuple[int, Path, Any, dict[str, Any], StoreInfo]],
    final_node: tuple[int, Path, Any, dict[str, Any], StoreInfo],
    leaf: dict[str, Any],
    nodeid: str,
    pair_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_numel = _tensor_numel(leaf.get("value"))
    if output_numel is None or output_numel <= 0:
        raise ValueError(
            f"output leaf {leaf.get('leaf_index')} is empty or not a tensor"
        )
    lhs_programs = lhs_launch.get("programs")
    if not isinstance(lhs_programs, int) or lhs_programs <= 0:
        raise ValueError(f"invalid LHS program count: {lhs_programs!r}")

    producer_by_storage: dict[
        str, list[tuple[int, Path, Any, dict[str, Any], StoreInfo]]
    ] = {}
    for node in rhs_nodes:
        producer_by_storage.setdefault(_storage_key(node[4].output), []).append(node)

    relevant: dict[
        tuple[int, int], tuple[int, Path, Any, dict[str, Any], StoreInfo]
    ] = {}

    def include(node: tuple[int, Path, Any, dict[str, Any], StoreInfo]) -> None:
        key = (node[0], node[4].index)
        if key in relevant:
            return
        relevant[key] = node
        launch_index, _, _, _, store = node
        for dependency in store.dependencies:
            storage = _storage_key(dependency)
            producers = [
                candidate
                for candidate in producer_by_storage.get(storage, [])
                if candidate[0] < launch_index
            ]
            if producers:
                latest_step = max(candidate[0] for candidate in producers)
                for candidate in producers:
                    if candidate[0] == latest_step:
                        include(candidate)

    include(final_node)
    ordered = [relevant[key] for key in sorted(relevant)]
    if len(ordered) < 2:
        raise ValueError(
            "multiple runtime launches were captured, but only one launch contributes "
            "to this observed output leaf"
        )

    internal_storage = {
        _storage_key(node[4].output) for node in ordered if node != final_node
    }
    final_storage = _storage_key(final_node[4].output)
    external_occurrences: dict[
        str, tuple[tuple[int, Path, Any, dict[str, Any], StoreInfo], PointerInfo]
    ] = {}
    for node in ordered:
        for pointer in node[4].dependencies:
            storage = _storage_key(pointer)
            if storage not in internal_storage and storage != final_storage:
                external_occurrences.setdefault(storage, (node, pointer))

    synthetic_rhs = StoreInfo(
        index=final_node[4].index,
        output=final_node[4].output,
        dependencies=tuple(pointer for _, pointer in external_occurrences.values()),
    )
    dependencies = _pair_dependencies(lhs_store, synthetic_rhs)
    role_pointers: dict[str, tuple[PointerInfo, PointerInfo]] = {
        f"Input{index}": pair for index, pair in enumerate(dependencies)
    }
    role_pointers["Output"] = (lhs_store.output, final_node[4].output)
    rhs_role_by_storage = {
        _storage_key(rhs): logical for logical, (_, rhs) in role_pointers.items()
    }
    internal_role_by_storage = {
        storage: f"Internal{index}"
        for index, storage in enumerate(sorted(internal_storage))
    }

    launches = []
    launch_pointer_inventories = []
    for launch_index, rhs_path, module, launch, store in ordered:
        launch_id = f"launch{launch_index:03d}.store{store.index:03d}"
        programs = launch.get("programs")
        if not isinstance(programs, int) or programs <= 0:
            raise ValueError(
                f"invalid RHS program count for launch {launch_index}: {programs!r}"
            )
        output_storage = _storage_key(store.output)
        output_role = (
            "Output"
            if (launch_index, store.index) == (final_node[0], final_node[4].index)
            else internal_role_by_storage[output_storage]
        )
        abi: dict[str, dict[str, Any]] = {output_role: _endpoint(store.output)}
        unmapped = []
        for pointer in store.dependencies:
            if pointer.argument == store.output.argument:
                continue
            storage = _storage_key(pointer)
            logical = internal_role_by_storage.get(storage) or rhs_role_by_storage.get(
                storage
            )
            if logical is None:
                logical = f"Unmapped.{launch_id}.arg{pointer.argument}"
                unmapped.append(pointer.argument)
            existing = abi.get(logical)
            endpoint = _endpoint(pointer)
            if existing is not None and existing != endpoint:
                raise ValueError(
                    f"launch {launch_id} uses logical role {logical} through multiple pointers"
                )
            abi[logical] = endpoint
        launches.append(
            {
                "id": launch_id,
                "step": launch_index,
                "file": str(Path("../..").joinpath(rhs_path.relative_to(case_dir))),
                "frontend": {
                    "kind": "ttir",
                    "function": module.function,
                    "programs": programs,
                    "store_index": store.index,
                },
                "semantic": f"captured launch {launch_index}, store {store.index}",
                "abi": abi,
                "bindings": _side_bindings(module, launch.get("runtime_arguments")),
            }
        )
        launch_pointer_inventories.append(
            {
                "launch_id": launch_id,
                "pointers": _pointer_inventory(module, launch, [store], store),
                "unmapped_selected_dependencies": sorted(unmapped),
            }
        )

    first_rhs = ordered[0]
    final_launch_id = f"launch{final_node[0]:03d}.store{final_node[4].index:03d}"
    roles = {}
    for logical, (lhs, rhs) in role_pointers.items():
        rhs_occurrence = (
            final_node
            if logical == "Output"
            else external_occurrences[_storage_key(rhs)][0]
        )
        rhs_launch_id = (
            final_launch_id
            if logical == "Output"
            else f"launch{rhs_occurrence[0]:03d}.store{rhs_occurrence[4].index:03d}"
        )
        roles[logical] = {
            "lhs": _endpoint(lhs),
            "rhs": _qualified_endpoint(rhs, rhs_launch_id),
        }
    disjoint = _disjoint_groups(role_pointers)
    spec = {
        "format": "etv-pair-v2",
        "metadata": {
            "pair_id": pair_id,
            "lhs": str(Path("../..").joinpath(lhs_path.relative_to(case_dir))),
            "rhs": str(Path("../..").joinpath(first_rhs[1].relative_to(case_dir))),
            "semantic_mode": "abstract_float",
            "frontends": {
                "lhs": {
                    "kind": "ttir",
                    "function": lhs_module.function,
                    "programs": lhs_programs,
                    "store_index": lhs_store.index,
                },
                "rhs": launches[0]["frontend"],
            },
            "limits": {
                "max_iterations": 8,
                "max_enodes": 20000,
                "timeout_ms": 10000,
            },
            "llm": {"enabled": True, "generate_rules": False},
            "partition": {
                "enabled": True,
                "min_partitions": 2,
                "max_partitions": max(16, len(launches) * 4),
            },
            "launches": {"rhs": launches},
        },
        "assumptions": {
            "for_llm": [
                "The RHS launches were captured in execution order from one compiled reference call.",
                "RHS launch boundaries are fixed; the LLM may only select corresponding LHS expression paths.",
                "Intermediate storage edges come from capture-v2 storage provenance and exact element offsets.",
                f"This PairSpec observes output leaf {leaf.get('path')!r} only.",
            ]
        },
        "predicates": {
            "abi": roles,
            "bindings": {"X": output_numel},
            "side_bindings": {
                "lhs": _side_bindings(lhs_module, lhs_launch.get("runtime_arguments")),
                "rhs": {},
            },
            "parameters": {},
            "constraints": [],
            "disjoint": disjoint,
            "custom": [],
        },
        "observation": {
            "output_role": "Output",
            "output_numel": {"var": "X"},
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "rewrites": [],
    }
    mapping = {
        "nodeid": nodeid,
        "mode": "prepartitioned_rhs_launch_sequence",
        "output_leaf": leaf,
        "lhs_store_index": lhs_store.index,
        "rhs_final_launch": final_node[0],
        "rhs_final_store_index": final_node[4].index,
        "ordered_components": [item["id"] for item in launches],
        "launch_pointer_inventories": launch_pointer_inventories,
        "roles": {
            logical: {
                "lhs_argument": lhs.argument,
                "rhs_argument": rhs.argument,
                "lhs_storage_id": lhs.storage_id,
                "rhs_storage_id": rhs.storage_id,
                "lhs_storage_offset": lhs.storage_offset,
                "rhs_storage_offset": rhs.storage_offset,
            }
            for logical, (lhs, rhs) in role_pointers.items()
        },
        "disjoint": disjoint,
    }
    return spec, mapping


def _reference_self_check(reference: dict[str, Any]) -> None:
    check = (reference.get("checks") or {}).get("compiled_vs_fx_eager")
    if not isinstance(check, dict):
        return
    if not check.get("structure_equal") or not all(
        leaf.get("allclose_1e-3") for leaf in check.get("tensor_leaves", [])
    ):
        raise ValueError("RHS compiled-vs-FX self-check did not pass")


def _attempt_multilaunch_reference(
    *,
    case_dir: Path,
    manifest: dict[str, Any],
    lhs_launch: dict[str, Any],
    reference: dict[str, Any],
    runtime_launches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lhs_path = Path(lhs_launch["ttir"]).resolve()
    lhs_module = parse_ttir(lhs_path)
    lhs_stores = _store_infos(lhs_module, lhs_launch)
    rhs_nodes: list[tuple[int, Path, Any, dict[str, Any], StoreInfo]] = []
    for launch_index, launch in enumerate(runtime_launches):
        rhs_path = Path(launch["_ttir_path"]).resolve()
        module = parse_ttir(rhs_path)
        stores = _store_infos(module, launch)
        if not stores:
            raise ValueError(f"RHS runtime launch {launch_index} has no tt.store")
        rhs_nodes.extend(
            (launch_index, rhs_path, module, launch, store) for store in stores
        )
    leaves = _output_leaves(reference)
    if not leaves:
        raise ValueError("reference output has no tensor leaves")

    reference_id = (
        f"r{reference.get('site_index', 0):03d}-"
        f"e{reference.get('execution', 0):03d}"
    )
    diagnostic_path = (
        case_dir / "pairspec" / "pairs" / f"diagnostic-{reference_id}.json"
    )
    _write_json(
        diagnostic_path,
        {
            "nodeid": manifest["nodeid"],
            "mode": "prepartitioned_rhs_launch_sequence",
            "reference_site": reference.get("site_index"),
            "reference_execution": reference.get("execution"),
            "lhs": {
                "stores": _store_inventory(lhs_stores),
                "pointers": _pointer_inventory(lhs_module, lhs_launch, lhs_stores),
            },
            "rhs_launches": [
                {
                    "launch_index": launch_index,
                    "path": str(nodes[0][1]),
                    "function": nodes[0][2].function,
                    "stores": _store_inventory([item[4] for item in nodes]),
                }
                for launch_index in range(len(runtime_launches))
                for nodes in [[item for item in rhs_nodes if item[0] == launch_index]]
            ],
            "output_leaves": leaves,
        },
    )

    generated = []
    for leaf in leaves:
        leaf_index = leaf.get("leaf_index", 0)
        leaf_storage = PointerInfo(-1, leaf.get("value")).storage_id
        candidates = []
        for node in rhs_nodes:
            declared = node[4].output.provenance.get("output_leaf_indices") or []
            if leaf_index in declared or (
                leaf_storage is not None and node[4].output.storage_id == leaf_storage
            ):
                candidates.append(node)
        if len(candidates) != 1:
            raise ValueError(
                f"cannot associate output leaf {leaf_index} with one final RHS launch store; "
                f"found {len(candidates)}"
            )
        final_node = candidates[0]
        all_rhs_stores = [item[4] for item in rhs_nodes]
        lhs_store = _lhs_store_for_leaf(lhs_stores, final_node[4], leaf, all_rhs_stores)
        pair_slug = f"{reference_id}-o{leaf_index:03d}"
        pair_id = f"{manifest['nodeid']}::{pair_slug}"
        spec, mapping = _multilaunch_pair_spec(
            case_dir=case_dir,
            lhs_path=lhs_path,
            lhs_module=lhs_module,
            lhs_launch=lhs_launch,
            lhs_store=lhs_store,
            rhs_nodes=rhs_nodes,
            final_node=final_node,
            leaf=leaf,
            nodeid=manifest["nodeid"],
            pair_id=pair_id,
        )
        spec_path = case_dir / "pairspec" / "pairs" / f"pair-{pair_slug}.json"
        mapping_path = case_dir / "pairspec" / "pairs" / f"mapping-{pair_slug}.json"
        _write_json(spec_path, spec)
        _write_json(mapping_path, mapping)
        generated.append(
            {
                "pair_id": pair_id,
                "pairspec": str(spec_path),
                "mapping": str(mapping_path),
                "reference_site": reference.get("site_index"),
                "reference_execution": reference.get("execution"),
                "output_leaf_index": leaf_index,
                "output_leaf_path": leaf.get("path"),
                "lhs_pointer_count": len(_pointer_arguments(lhs_module)),
                "rhs_pointer_count": sum(
                    len(_pointer_arguments(nodes[0][2]))
                    for launch_index in range(len(runtime_launches))
                    for nodes in [
                        [item for item in rhs_nodes if item[0] == launch_index]
                    ]
                ),
                "lhs_store_count": len(lhs_stores),
                "rhs_store_count": len(rhs_nodes),
                "rhs_launch_count": len(runtime_launches),
                "prepartitioned_side": "rhs",
                "output_leaf_count": len(leaves),
                "has_nonzero_endpoint_offset": any(
                    role.get("lhs_storage_offset") or role.get("rhs_storage_offset")
                    for role in mapping["roles"].values()
                ),
                "unmapped_selected_dependencies": {
                    "lhs": [],
                    "rhs": sorted(
                        {
                            argument
                            for inventory in mapping["launch_pointer_inventories"]
                            for argument in inventory["unmapped_selected_dependencies"]
                        }
                    ),
                },
            }
        )
    return generated


def _attempt_reference(
    *,
    case_dir: Path,
    manifest: dict[str, Any],
    lhs_launch: dict[str, Any],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    _reference_self_check(reference)
    runtime_launches = _runtime_ttir_launches(reference)
    if not runtime_launches:
        raise ValueError("requires at least one runtime launch with selected RHS TTIR")
    if len(runtime_launches) > 1:
        return _attempt_multilaunch_reference(
            case_dir=case_dir,
            manifest=manifest,
            lhs_launch=lhs_launch,
            reference=reference,
            runtime_launches=runtime_launches,
        )
    lhs_path = Path(lhs_launch["ttir"]).resolve()
    rhs_path = Path(runtime_launches[0]["_ttir_path"]).resolve()
    lhs_module = parse_ttir(lhs_path)
    rhs_module = parse_ttir(rhs_path)
    lhs_stores = _store_infos(lhs_module, lhs_launch)
    rhs_stores = _store_infos(rhs_module, runtime_launches[0])
    leaves = _output_leaves(reference)
    if not leaves and len(rhs_stores) == 1:
        description = rhs_stores[0].output.description
        if _tensor_numel(description):
            leaves = [
                {
                    "leaf_index": 0,
                    "path": "inferred_single_rhs_store_output",
                    "value": description,
                    "inference": "single RHS store target",
                }
            ]
    if not leaves:
        raise ValueError("reference output has no tensor leaves")

    reference_id = (
        f"r{reference.get('site_index', 0):03d}-e{reference.get('execution', 0):03d}"
    )
    diagnostic_path = (
        case_dir / "pairspec" / "pairs" / f"diagnostic-{reference_id}.json"
    )
    _write_json(
        diagnostic_path,
        {
            "nodeid": manifest["nodeid"],
            "reference_site": reference.get("site_index"),
            "reference_execution": reference.get("execution"),
            "lhs": {
                "pointer_count": len(_pointer_arguments(lhs_module)),
                "stores": _store_inventory(lhs_stores),
                "pointers": _pointer_inventory(lhs_module, lhs_launch, lhs_stores),
            },
            "rhs": {
                "pointer_count": len(_pointer_arguments(rhs_module)),
                "stores": _store_inventory(rhs_stores),
                "pointers": _pointer_inventory(
                    rhs_module, runtime_launches[0], rhs_stores
                ),
            },
            "output_leaves": leaves,
        },
    )

    generated: list[dict[str, Any]] = []
    for rhs_store in rhs_stores:
        leaf = _leaf_for_rhs_store(rhs_store, leaves, rhs_stores)
        lhs_store = _lhs_store_for_leaf(lhs_stores, rhs_store, leaf, rhs_stores)
        leaf_index = leaf.get("leaf_index", rhs_store.index)
        pair_slug = f"{reference_id}-o{leaf_index:03d}"
        pair_id = f"{manifest['nodeid']}::{pair_slug}"
        spec, mapping = _pair_spec(
            case_dir=case_dir,
            lhs_path=lhs_path,
            rhs_path=rhs_path,
            lhs_module=lhs_module,
            rhs_module=rhs_module,
            lhs_launch=lhs_launch,
            rhs_launch=runtime_launches[0],
            lhs_store=lhs_store,
            rhs_store=rhs_store,
            leaf=leaf,
            nodeid=manifest["nodeid"],
            pair_id=pair_id,
        )
        spec_path = case_dir / "pairspec" / "pairs" / f"pair-{pair_slug}.json"
        mapping_path = case_dir / "pairspec" / "pairs" / f"mapping-{pair_slug}.json"
        _write_json(spec_path, spec)
        _write_json(mapping_path, mapping)
        unmapped = mapping["unmapped_selected_dependencies"]
        generated.append(
            {
                "pair_id": pair_id,
                "pairspec": str(spec_path),
                "mapping": str(mapping_path),
                "reference_site": reference.get("site_index"),
                "reference_execution": reference.get("execution"),
                "output_leaf_index": leaf_index,
                "output_leaf_path": leaf.get("path"),
                "lhs_pointer_count": len(_pointer_arguments(lhs_module)),
                "rhs_pointer_count": len(_pointer_arguments(rhs_module)),
                "lhs_store_count": len(lhs_stores),
                "rhs_store_count": len(rhs_stores),
                "output_leaf_count": len(leaves),
                "output_leaf_inferred": bool(leaf.get("inference")),
                "has_nonzero_endpoint_offset": any(
                    pointer.storage_offset
                    for pointer_pair in _pair_dependencies(lhs_store, rhs_store)
                    for pointer in pointer_pair
                )
                or bool(lhs_store.output.storage_offset)
                or bool(rhs_store.output.storage_offset),
                "unmapped_selected_dependencies": unmapped,
            }
        )
    return generated


def _generate(case_dir: Path) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    status: dict[str, Any] = {
        "format": "etv-benchmark-pairspec-status-v2",
        "nodeid": manifest.get("nodeid"),
        "case_dir": str(case_dir),
        "status": "unavailable",
        "pairspecs": [],
        "attempts": [],
    }
    pair_root = case_dir / "pairspec" / "pairs"
    shutil.rmtree(pair_root, ignore_errors=True)
    (case_dir / "pairspec" / "pair.json").unlink(missing_ok=True)
    lhs_launches = [
        item
        for item in manifest.get("lhs_launches", [])
        if item.get("status") == "captured" and item.get("ttir")
    ]
    if len(lhs_launches) != 1:
        status["reason"] = (
            f"requires one captured LHS launch, found {len(lhs_launches)}"
        )
        _write_json(case_dir / "pairspec" / "status.json", status)
        return status
    references = [
        item
        for item in manifest.get("rhs_references", [])
        if item.get("ttir_status") == "captured"
    ]
    if not references:
        status["reason"] = "requires at least one reference with RHS TTIR, found 0"
        _write_json(case_dir / "pairspec" / "status.json", status)
        return status

    for reference in references:
        reference_id = (
            f"r{reference.get('site_index', 0):03d}-"
            f"e{reference.get('execution', 0):03d}"
        )
        diagnostic_path = pair_root / f"diagnostic-{reference_id}.json"
        attempt = {
            "reference_site": reference.get("site_index"),
            "reference_execution": reference.get("execution"),
            "status": "unavailable",
        }
        try:
            generated = _attempt_reference(
                case_dir=case_dir,
                manifest=manifest,
                lhs_launch=lhs_launches[0],
                reference=reference,
            )
            status["pairspecs"].extend(generated)
            attempt.update({"status": "generated", "generated": len(generated)})
        except Exception as error:
            attempt["reason"] = str(error)
        if diagnostic_path.exists():
            attempt["diagnostics"] = str(diagnostic_path)
        status["attempts"].append(attempt)

    if status["pairspecs"]:
        status["status"] = "generated"
        status["generated_specs"] = len(status["pairspecs"])
        failed = [item for item in status["attempts"] if item["status"] != "generated"]
        if failed:
            status["partial_failures"] = failed
        status["features"] = {
            "multiple_references": len(references) > 1,
            "multiple_references_fully_generated": len(references) > 1
            and all(item.get("status") == "generated" for item in status["attempts"]),
            "unequal_pointer_abi": any(
                item.get("lhs_pointer_count") != item.get("rhs_pointer_count")
                for item in status["pairspecs"]
            ),
            "multiple_stores": any(
                item.get("lhs_store_count", 0) > 1 or item.get("rhs_store_count", 0) > 1
                for item in status["pairspecs"]
            ),
            "multiple_launches": any(
                item.get("rhs_launch_count", 1) > 1 for item in status["pairspecs"]
            ),
            "tuple_output": any(
                item.get("output_leaf_count", 0) > 1 for item in status["pairspecs"]
            ),
            "nonzero_endpoint_offset": any(
                item.get("has_nonzero_endpoint_offset") for item in status["pairspecs"]
            ),
            "incomplete_pointer_mapping": any(
                item.get("unmapped_selected_dependencies", {}).get("lhs")
                or item.get("unmapped_selected_dependencies", {}).get("rhs")
                for item in status["pairspecs"]
            ),
            "inferred_single_store_output": any(
                item.get("output_leaf_inferred") for item in status["pairspecs"]
            ),
        }
    else:
        reasons = [item.get("reason", "unknown") for item in status["attempts"]]
        status["reason"] = "; ".join(dict.fromkeys(reasons))
    _write_json(case_dir / "pairspec" / "status.json", status)
    return status


def _manifest_unavailable(case_dir: Path, nodeid: str | None) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    status = {
        "format": "etv-benchmark-pairspec-status-v2",
        "nodeid": nodeid,
        "case_dir": str(case_dir),
        "status": "unavailable",
        "pairspecs": [],
        "attempts": [],
        "reason": "capture manifest unavailable (test did not reach capture hook)",
    }
    shutil.rmtree(case_dir / "pairspec" / "pairs", ignore_errors=True)
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
            (Path(record["case_dir"]).resolve(), record.get("nodeid"))
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
        "format": "etv-benchmark-pairspec-summary-v2",
        "counts": dict(sorted(counts.items())),
        "spec_counts": {
            "generated": sum(len(record.get("pairspecs", [])) for record in records),
            "failed_reference_attempts": sum(
                item.get("status") != "generated"
                for record in records
                for item in record.get("attempts", [])
            ),
        },
        "recovered_case_counts": {
            feature: sum(
                bool(record.get("features", {}).get(feature)) for record in records
            )
            for feature in (
                "multiple_references",
                "multiple_references_fully_generated",
                "unequal_pointer_abi",
                "multiple_stores",
                "tuple_output",
                "nonzero_endpoint_offset",
                "incomplete_pointer_mapping",
                "inferred_single_store_output",
            )
        },
        "records": records,
    }
    _write_json(benchmark_root / "pairspec-summary.json", summary)
    print(json.dumps({**summary["counts"], **summary["spec_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
