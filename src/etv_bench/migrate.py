"""Deterministic, audited migration of explicitly mapped v2 fixtures."""

import hashlib
import copy
import json
from pathlib import Path
import shutil
from typing import Any


def migrate_v2(source: Path, destination: Path, store_roles: dict[str, str] | None = None) -> Path:
    old = json.loads(source.read_text())
    if old["format"] != "etv-pair-v2":
        raise ValueError("expected v2 input")
    destination.mkdir(parents=True, exist_ok=True)
    metadata, predicates = old["metadata"], old["predicates"]
    roles: dict[str, dict[str, str]] = {}
    launches: dict[str, list[dict[str, Any]]] = {"lhs": [], "rhs": []}
    artifacts = []
    store_roles = store_roles or {}
    for side in ("lhs", "rhs"):
        components = metadata.get("launches", {}).get(side)
        if components is None:
            components = [
                {
                    "id": side + ".0",
                    "file": metadata[side],
                    "step": 0,
                    "frontend": metadata["frontends"][side],
                    "abi": {
                        role: endpoints[side]
                        for role, endpoints in predicates["abi"].items()
                        if side in endpoints
                    },
                    "bindings": predicates.get("side_bindings", {}).get(side, {}),
                }
            ]
        groups: dict[tuple[str, str, int], dict[str, Any]] = {}
        for component in components:
            front = component["frontend"]
            key = (component["file"], front["function"], component["step"])
            if key not in groups:
                ordinal = len(groups)
                original = (source.parent / component["file"]).resolve()
                target = destination / f"{side}-{ordinal}.ttir"
                shutil.copyfile(original, target)
                sha = hashlib.sha256(original.read_bytes()).hexdigest()
                artifacts.append(
                    {
                        "original": component["file"],
                        "path": target.name,
                        "before_sha256": sha,
                        "after_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
                groups[key] = {
                    "id": f"{side}.{ordinal}",
                    "file": target.name,
                    "function": front["function"],
                    "grid": {"programs": front["programs"]},
                    "step": component["step"],
                    "abi": copy.deepcopy(component["abi"]),
                    "bindings": component["bindings"],
                    "stores": [],
                }
            group = groups[key]
            if group["bindings"] != component["bindings"]:
                raise ValueError("components of a physical launch disagree on bindings")
            for role, endpoint in component["abi"].items():
                if role in group["abi"] and group["abi"][role] != endpoint:
                    raise ValueError("conflicting physical launch ABI")
                group["abi"][role] = endpoint
            index = front.get("store_index", 0)
            role = store_roles.get(f"{side}:{component['id']}")
            if role is None:
                if metadata.get("launches", {}).get(side) is not None:
                    raise ValueError(
                        "explicit store role required for legacy multilaunch component"
                    )
                role = old["observation"]["output_role"]
            group["stores"].append({"index": index, "role": role})
            for name, endpoint in group["abi"].items():
                roles[name] = {
                    "kind": "buffer" if endpoint["kind"] == "block" else "scalar",
                    "element": "abstract_float",
                }
        launches[side] = list(groups.values())
    old_limits = metadata["limits"]
    result = {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": metadata["pair_id"],
            "semantic_mode": "abstract_float",
            "target": {"index_bits": 64},
            "limits": {
                "smt_timeout_ms": 2000,
                "egraph_timeout_ms": old_limits["timeout_ms"],
                "max_iterations": old_limits["max_iterations"],
                "max_enodes": old_limits["max_enodes"],
            },
            "partition": {"enabled": False, "provider": "none", "max_parts": 16},
            "llm": {"enabled": False},
        },
        "programs": launches,
        "assumptions": old["assumptions"],
        "predicates": {
            "roles": roles,
            "bindings": predicates.get("bindings", {}),
            "parameters": predicates.get("parameters", {}),
            "constraints": predicates.get("constraints", []),
            "disjoint": predicates["disjoint"],
            "custom": predicates.get("custom", []),
        },
        "observation": {
            "role": old["observation"]["output_role"],
            "numel": old["observation"]["output_numel"],
            "require_full_coverage": old["observation"]["require_full_coverage"],
            "require_disjoint": old["observation"]["require_disjoint"],
        },
        "rewrites": [],
    }
    output = destination / "pair.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    audit = {
        "format": "etv-v2-migration-v1",
        "original_pair_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "migrated_pair_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "artifacts": artifacts,
        "original_pair": old,
        "store_roles": store_roles,
        "review": [],
        "facts": "ABI and bindings are trusted input premises; migration is not a proof.",
    }
    (destination / "provenance.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return output
