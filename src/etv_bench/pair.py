"""Pair launch/storage records using compiler slots and the public snapshot CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _contained(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact path missing or escapes capture root")
    return path


def _snapshot(root: Path, launch: dict[str, Any], command: list[str]) -> dict[str, Any]:
    if (
        launch["side"] not in ("lhs", "rhs")
        or type(launch["ordinal"]) is not int
        or launch["ordinal"] < 0
    ):
        raise ValueError("invalid physical launch identity")
    artifact = _contained(root, launch["file"])
    path = root / f"snapshot-{launch['side']}-{launch['ordinal']}.json"
    process = subprocess.run(
        command + ["parse", str(artifact), "--function", launch["function"], "--out", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode:
        raise ValueError("official TTIR snapshot unavailable")
    return _read(path)["function"]


def _memory_slots(function: dict[str, Any]) -> tuple[list[int], list[int], int]:
    blocks = function["regions"][0]
    if len(blocks) != 1:
        raise ValueError("ambiguous function blocks")
    arguments = blocks[0]["arguments"]
    owner = {value["id"]: index for index, value in enumerate(arguments)}
    operations: list[dict[str, Any]] = []

    def visit(block: dict[str, Any]) -> None:
        for operation in block["operations"]:
            operations.append(operation)
            for region in operation["regions"]:
                for nested in region:
                    visit(nested)

    visit(blocks[0])
    definitions = {
        value["id"]: operation for operation in operations for value in operation["results"]
    }

    def slot(value: str) -> int:
        if value in owner:
            return owner[value]
        operation = definitions[value]
        if operation["name"] not in (
            "tt.addptr",
            "tt.splat",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
        ):
            raise ValueError("pointer provenance is not a unique argument")
        return slot(operation["operands"][0]["id"])

    reads = [slot(op["operands"][0]["id"]) for op in operations if op["name"] == "tt.load"]
    writes = [slot(op["operands"][0]["id"]) for op in operations if op["name"] == "tt.store"]
    return reads, writes, len(arguments)


def pair_capture(capture_path: Path, command: list[str]) -> Path:
    root = capture_path.parent.resolve()
    capture = _read(capture_path)
    if capture.get("format") != "etv-runtime-capture-v1":
        raise ValueError("expected runtime capture v1")
    audit: list[dict[str, Any]] = []
    diagnostics = list(capture.get("diagnostics", []))
    analyzed = []
    for launch in capture["launches"]:
        try:
            if not launch["selected_runtime_launch"] or launch["grid"][1:] != [1, 1]:
                raise ValueError("launch selection or grid unsupported")
            artifact = _contained(root, launch["file"])
            if (
                launch.get("sha256")
                and hashlib.sha256(artifact.read_bytes()).hexdigest() != launch["sha256"]
            ):
                raise ValueError("captured artifact hash mismatch")
            function = _snapshot(root, launch, command)
            reads, writes, arity = _memory_slots(function)
            if len(launch["slots"]) != arity or [s["slot"] for s in launch["slots"]] != list(
                range(arity)
            ):
                raise ValueError("compiler signature does not match TTIR arguments")
            analyzed.append((launch, reads, writes))
        except (ValueError, KeyError, subprocess.SubprocessError) as exc:
            diagnostics.append(str(exc))
    programs: dict[str, list[dict[str, Any]]] = {"lhs": [], "rhs": []}
    roles: dict[str, dict[str, str]] = {}
    storage_roles: dict[tuple[str, str], str] = {}
    for side in ("lhs", "rhs"):
        for storage, role in capture.get("input_roles", {}).items():
            storage_roles[(side, storage)] = role
        for storage, role in capture["outputs"].get(side, {}).items():
            if (side, storage) in storage_roles and storage_roles[(side, storage)] != role:
                diagnostics.append("output aliases an input; explicit reviewed contract required")
            storage_roles[(side, storage)] = role
        for launch, _, writes in analyzed:
            if launch["side"] != side:
                continue
            for slot in writes:
                runtime = launch["slots"][slot]["runtime"]
                identity = runtime.get("storage_id")
                if identity is not None and (side, identity) not in storage_roles:
                    storage_roles[(side, identity)] = f"Internal_{side}_{len(storage_roles)}"
    for launch, reads, writes in analyzed:
        side = launch["side"]
        abi, bindings = {}, {}
        slot_roles = {}
        for slot in launch["slots"]:
            name = f"arg{slot['slot']}"
            runtime = slot["runtime"]
            if runtime["kind"] == "integer":
                bindings[name] = runtime["value"]
            elif runtime["kind"] == "buffer":
                if runtime.get("dtype") not in (
                    "torch.float16",
                    "torch.bfloat16",
                    "torch.float32",
                    "torch.float64",
                ):
                    diagnostics.append("nonfloat memory requires review")
                role = storage_roles.get((side, runtime["storage_id"]))
                if role is None:
                    diagnostics.append("external input lacks a unique declared storage role")
                    continue
                if role in abi:
                    diagnostics.append("duplicate physical endpoints for one role require review")
                    continue
                abi[role] = {"kind": "block", "name": name, "offset": runtime["offset"]}
                slot_roles[slot["slot"]] = role
                roles[role] = {"kind": "buffer", "element": "abstract_float"}
            else:
                diagnostics.append("noninteger scalar requires explicit reviewed ABI")
            audit.append(
                {
                    "launch": launch["ordinal"],
                    "side": side,
                    "slot": slot["slot"],
                    "source_name": slot["source_name"],
                    "compiler_type": slot["compiler_type"],
                    "runtime": runtime,
                }
            )
        if any(slot not in slot_roles for slot in reads + writes):
            diagnostics.append("memory role incomplete")
            continue
        programs[side].append(
            {
                "id": f"{side}.{launch['ordinal']}",
                "file": launch["file"],
                "function": launch["function"],
                "step": launch["step"],
                "grid": {"programs": launch["grid"][0]},
                "abi": abi,
                "bindings": bindings,
                "stores": [{"index": i, "role": slot_roles[slot]} for i, slot in enumerate(writes)],
            }
        )
    if not programs["lhs"] or not programs["rhs"]:
        diagnostics.append("both launch sequences are required")
    output_roles = [set(capture["outputs"].get(side, {}).values()) for side in ("lhs", "rhs")]
    if output_roles != [{"Output"}, {"Output"}] or type(capture.get("output_numel")) is not int:
        diagnostics.append("single Output role and explicit output_numel are required")
    pair = {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": capture["source"]["test_id"],
            "semantic_mode": "abstract_float",
            "target": {"index_bits": 64},
            "limits": {
                "smt_timeout_ms": 2000,
                "egraph_timeout_ms": 10000,
                "max_iterations": 8,
                "max_enodes": 20000,
            },
        },
        "programs": programs,
        "assumptions": {"for_llm": []},
        "predicates": {
            "roles": roles,
            "bindings": {},
            "parameters": {},
            "constraints": [],
            "disjoint": [],
            "custom": [],
        },
        "observation": {
            "role": "Output",
            "numel": capture.get("output_numel", 0),
            "require_full_coverage": True,
            "require_disjoint": sorted(roles),
        },
        "rewrites": [],
    }
    # Captured allocation identities establish separation premises, not proved facts.
    identities_by_role: dict[str, set[str]] = {}
    for (_, identity), role in storage_roles.items():
        identities_by_role.setdefault(role, set()).add(identity)
    for i, role in enumerate(sorted(roles)):
        for other in sorted(roles)[i + 1 :]:
            if identities_by_role[role].isdisjoint(identities_by_role[other]):
                pair["predicates"]["disjoint"].append([role, other])
    output = root / "pair.json"
    _write(output, pair)
    status = "needs_review" if diagnostics else "ready"
    if status == "ready":
        # The public verify command validates the entire strict input contract.
        checked = subprocess.run(
            command + ["verify", str(output), "--out", str(root / "validation"), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            report = json.loads(checked.stdout)
            if report["reason"] in (
                "INVALID_PAIRSPEC",
                "INPUT_NOT_FOUND",
                "FILE_HASH_MISMATCH",
                "ABI_ROLE_MISSING",
                "TTIR_VERSION_MISMATCH",
                "TTIR_PARSE_FAILED",
                "TTIR_VERIFY_FAILED",
                "INTERNAL_ERROR",
            ):
                status = "needs_review"
                diagnostics.append(report["reason"])
        except (ValueError, KeyError):
            status = "needs_review"
            diagnostics.append("verifier did not produce a schema validation report")
    artifacts = [
        {
            "path": launch["file"],
            "sha256": hashlib.sha256(_contained(root, launch["file"]).read_bytes()).hexdigest(),
            "side": launch["side"],
            "launch": launch["ordinal"],
            "step": launch["step"],
        }
        for launch in capture["launches"]
    ]
    _write(
        root / "manifest.json",
        {
            "format": "etv-benchmark-case-v1",
            "case_id": capture["source"]["test_id"],
            "status": status,
            "generation": {"mode": "automatic", "tool_version": "0.3.0"},
            "source": capture["source"],
            "environment": capture["environment"],
            "artifacts": artifacts,
            "pair": "pair.json",
            "provenance": "provenance.json",
            "expected": "expected.json",
        },
    )
    _write(
        root / "provenance.json",
        {
            "format": "etv-artifact-provenance-v1",
            "capture": capture,
            "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "pair_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "facts": audit,
            "artifacts": artifacts,
            "diagnostics": diagnostics,
            "review": [],
        },
    )
    _write(root / "expected.json", {"status": "UNSPECIFIED"})
    return root / "manifest.json"
