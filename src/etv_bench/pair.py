"""Pair launch/storage records using compiler slots and the public snapshot CLI."""

import hashlib
import json
from pathlib import Path
import re
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
            "tt.bitcast",
        ):
            raise ValueError("pointer provenance is not a unique argument")
        return slot(operation["operands"][0]["id"])

    reads = [slot(op["operands"][0]["id"]) for op in operations if op["name"] == "tt.load"]
    writes = [slot(op["operands"][0]["id"]) for op in operations if op["name"] == "tt.store"]
    return reads, writes, len(arguments)


def _element(runtime: dict[str, Any]) -> str | None:
    dtype = runtime.get("dtype")
    if not isinstance(dtype, str):
        return None
    return {
        "torch.float16": "abstract_float",
        "torch.bfloat16": "abstract_float",
        "torch.float32": "abstract_float",
        "torch.float64": "abstract_float",
        "torch.bool": "bool",
        "torch.int8": "i8",
        "torch.int16": "i16",
        "torch.int32": "i32",
        "torch.int64": "i64",
    }.get(dtype)


def _append_endpoint(abi: dict[str, Any], role: str, name: str, offset: int) -> None:
    """Retain every physical pointer to a logical allocation, including views."""
    endpoint = {"kind": "block", "name": name, "offset": offset}
    previous = abi.get(role)
    if previous is None:
        abi[role] = endpoint
    elif isinstance(previous, list):
        previous.append(endpoint)
    else:
        abi[role] = [previous, endpoint]


def _check_storage_views(launches: list[dict[str, Any]], diagnostics: list[str]) -> None:
    """Aliases share an element address space, not only an abstract value type."""
    seen: dict[str, dict[str, Any]] = {}
    for launch in launches:
        for slot in launch["slots"]:
            runtime = slot["runtime"]
            if runtime.get("kind") != "buffer":
                continue
            identity = runtime["storage_id"]
            prior = seen.setdefault(identity, {})
            for key in ("dtype", "element_size", "allocation_base", "device", "storage_nbytes"):
                if key not in runtime:
                    continue
                if key in prior and prior[key] != runtime[key]:
                    diagnostic = (
                        "aliased storage has inconsistent element interpretation"
                        if key in ("dtype", "element_size")
                        else "captured storage identity has inconsistent allocation metadata"
                    )
                    if diagnostic not in diagnostics:
                        diagnostics.append(diagnostic)
                prior[key] = runtime[key]


def _unused_arguments(function: dict[str, Any]) -> set[int]:
    """Only a literally unreferenced SSA argument is certified unused here."""
    if not isinstance(function, dict) or not function.get("regions"):
        return set()
    blocks = function["regions"][0]
    if len(blocks) != 1:
        return set()
    referenced: set[str] = set()

    def visit(block: dict[str, Any]) -> None:
        for operation in block["operations"]:
            referenced.update(operand["id"] for operand in operation["operands"])
            for region in operation["regions"]:
                for nested in region:
                    visit(nested)

    visit(blocks[0])
    return {
        index
        for index, argument in enumerate(blocks[0]["arguments"])
        if argument["id"] not in referenced
    }


_BOOLEAN_OUTPUT_OPERATORS = frozenset(("eq", "ne", "lt", "le", "gt", "ge", "isinf", "isnan"))


def _boolean_output_representation(
    capture: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select an explicit injective output encoding only for known predicates."""
    source = capture["source"]
    operator = source.get("operator")
    source_field = "capture.source.operator"
    source_value = operator
    if operator is None:
        source_value = source.get("test_id", "")
        matched = re.fullmatch(r"(?:[^:]+/)?test_([a-z_]+)\.py::test_\1(?:\[.*\])?", source_value)
        operator = matched.group(1) if matched else None
        source_field = "capture.source.test_id"
    if operator not in _BOOLEAN_OUTPUT_OPERATORS:
        return None, None
    sides: dict[str, dict[str, Any]] = {}
    for side in ("lhs", "rhs"):
        outputs = capture["outputs"].get(side, {})
        if len(outputs) != 1 or set(outputs.values()) != {"Output"}:
            return None, None
        identity = next(iter(outputs))
        descriptors = [
            slot["runtime"]
            for launch in capture["launches"]
            if launch["side"] == side
            for slot in launch["slots"]
            if slot["runtime"].get("storage_id") == identity
        ]
        elements = {_element(runtime) for runtime in descriptors}
        if len(elements) != 1:
            return None, None
        sides[side] = {
            "storage_id": identity,
            "element": next(iter(elements)),
            "dtypes": sorted({runtime["dtype"] for runtime in descriptors}),
        }
    if {value["element"] for value in sides.values()} != {"abstract_float", "bool"}:
        return None, None
    representation = {
        "kind": "boolean_numeric_01",
        **{
            side: {
                "role": f"Output_{side}",
                "encoding": "bool_to_numeric_01" if value["element"] == "bool" else "identity",
            }
            for side, value in sides.items()
        },
    }
    evidence = {
        "operator": operator,
        "operator_source": source_field,
        "source_value": source_value,
        "physical_outputs": sides,
        "value_representation": representation,
        "contract": "full output equality after injective bool-to-0/1 encoding; dtype equality is not claimed",
    }
    return representation, evidence


def _host_events(capture: dict[str, Any], diagnostics: list[str]) -> list[tuple[int, dict]]:
    """Keep capture indices stable: PairSpec refers to the hashed source event."""
    if "host_operations" not in capture:
        return []
    if capture.get("host_capture_format") != "etv-host-operations-v1":
        diagnostics.append("unsupported host capture format")
        return []
    events = []
    steps: set[tuple[str, int]] = set()
    for launch in capture["launches"]:
        key = (launch["side"], launch["step"])
        if key in steps:
            diagnostics.append("host capture requires a unique ordered step per event")
        steps.add(key)
    for index, event in enumerate(capture["host_operations"]):
        try:
            if event["kind"] not in ("constant", "copy", "div_scalar", "pad"):
                raise ValueError("unsupported host operation: " + str(event.get("operator")))
            if event["side"] not in ("lhs", "rhs") or type(event["step"]) is not int:
                raise ValueError("invalid host event identity")
            key = (event["side"], event["step"])
            if event["step"] < 0 or key in steps:
                raise ValueError("host capture requires a unique ordered step per event")
            steps.add(key)
            if not isinstance(event["inputs"], list) or any(
                value.get("kind") != "buffer" for value in event["inputs"] + [event["output"]]
            ):
                raise ValueError("host event requires captured tensor descriptors")
            events.append((index, event))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            diagnostics.append("host event incomplete: " + str(exc))
    return events


EXECUTION_SCOPES = ("ttir-only", "ttir-with-host-ops")


def _capture_diagnostics(
    capture: dict[str, Any], execution_scope: str
) -> tuple[list[str], list[str]]:
    """Separate unsupported host semantics from missing capture provenance."""
    original = list(capture.get("diagnostics", []))
    if execution_scope != "ttir-only":
        return original, []
    host_diagnostics = set()
    events = capture.get("host_operations", [])
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or event.get("kind") != "unsupported":
            continue
        try:
            host_diagnostics.add(
                f"unsupported host operation ({event['side']} step {event['step']}): "
                f"{event['operator']}: {event['attributes']['reason']}"
            )
        except (KeyError, TypeError):
            continue
    return (
        [item for item in original if item not in host_diagnostics],
        [item for item in original if item in host_diagnostics],
    )


def _host_output_steps(
    capture: dict[str, Any], diagnostics: list[str]
) -> dict[tuple[str, str], list[int]]:
    """Use host evidence only to reject unavailable TTIR-only input relations."""
    result: dict[tuple[str, str], list[int]] = {}
    events = capture.get("host_operations", [])
    if (
        "host_operations" in capture
        and capture.get("host_capture_format") != "etv-host-operations-v1"
    ):
        diagnostics.append("unsupported host capture format")
    if not isinstance(events, list):
        diagnostics.append("host evidence outside ttir-only scope has incomplete effect provenance")
        return result
    steps = {(launch["side"], launch["step"]) for launch in capture["launches"]}
    for event in events:
        try:
            if event.get("kind") not in ("constant", "copy", "div_scalar", "pad", "unsupported"):
                raise ValueError("host event effect kind is unknown")
            side, step = event["side"], event["step"]
            if side not in ("lhs", "rhs") or type(step) is not int or step < 0:
                raise ValueError("invalid host event ordering")
            if (side, step) in steps:
                raise ValueError("ambiguous host/TTIR event ordering")
            steps.add((side, step))
            outputs = (
                [event["output"]] if event.get("output") is not None else event.get("outputs", [])
            )
            if not outputs:
                raise ValueError("host event output effect is unknown")
            if event.get("kind") == "unsupported":
                writes = event["attributes"]["writes"]
                if type(writes) is not bool:
                    raise ValueError("host event mutation effect is unknown")
                if writes:
                    # The capture knows that some tensor argument may be mutated,
                    # but not which one; retain all possible affected allocations.
                    outputs = outputs + event["inputs"]
            for output in outputs:
                identity = output["storage_id"]
                if not isinstance(identity, str):
                    raise ValueError("invalid storage identity")
                result.setdefault((side, identity), []).append(step)
        except (ValueError, TypeError, KeyError, AttributeError):
            diagnostics.append(
                "host evidence outside ttir-only scope has incomplete effect provenance"
            )
    return result


def pair_capture(
    capture_path: Path,
    command: list[str],
    *,
    execution_scope: str = "ttir-only",
) -> Path:
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("unsupported execution scope: " + execution_scope)
    root = capture_path.parent.resolve()
    capture = _read(capture_path)
    if capture.get("format") != "etv-runtime-capture-v1":
        raise ValueError("expected runtime capture v1")
    audit: list[dict[str, Any]] = []
    diagnostics, excluded_capture_diagnostics = _capture_diagnostics(capture, execution_scope)
    _check_storage_views(capture["launches"], diagnostics)
    host_events = (
        _host_events(capture, diagnostics) if execution_scope == "ttir-with-host-ops" else []
    )
    excluded_host_writes = (
        _host_output_steps(capture, diagnostics) if execution_scope == "ttir-only" else {}
    )
    capture_hash = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    analyzed = []
    unused_arguments: dict[tuple[str, int], set[int]] = {}
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
            unused_arguments[(launch["side"], launch["ordinal"])] = _unused_arguments(function)
            if len(launch["slots"]) != arity or [s["slot"] for s in launch["slots"]] != list(
                range(arity)
            ):
                raise ValueError("compiler signature does not match TTIR arguments")
            analyzed.append((launch, reads, writes))
        except (ValueError, KeyError, subprocess.SubprocessError) as exc:
            diagnostics.append(str(exc))
    ttir_writes: dict[tuple[str, str], list[int]] = {}
    for launch, _, writes in analyzed:
        for slot in writes:
            identity = launch["slots"][slot]["runtime"].get("storage_id")
            if identity is not None:
                ttir_writes.setdefault((launch["side"], identity), []).append(launch["step"])

    def excluded_host_dependency(side: str, identity: str, before: int | None = None) -> bool:
        def latest(steps: list[int]) -> int:
            return max((step for step in steps if before is None or step < before), default=-1)

        key = (side, identity)
        return latest(excluded_host_writes.get(key, [])) > latest(ttir_writes.get(key, []))

    programs: dict[str, list[dict[str, Any]]] = {"lhs": [], "rhs": []}
    roles: dict[str, dict[str, str]] = {}
    storage_roles: dict[tuple[str, str], str] = {}
    role_identities: dict[tuple[str, str], set[str]] = {}
    unused_storage_roles: dict[tuple[str, str], str] = {}
    unused_pointer_evidence: list[dict[str, Any]] = []
    value_representation, representation_evidence = _boolean_output_representation(capture)
    for side in ("lhs", "rhs"):
        for storage, role in capture.get("input_roles", {}).items():
            storage_roles[(side, storage)] = role
        for storage, role in capture["outputs"].get(side, {}).items():
            if (side, storage) in storage_roles and storage_roles[(side, storage)] != role:
                diagnostics.append("output aliases an input; explicit reviewed contract required")
            storage_roles[(side, storage)] = (
                value_representation[side]["role"] if value_representation else role
            )
        for launch, _, writes in analyzed:
            if launch["side"] != side:
                continue
            for slot in writes:
                runtime = launch["slots"][slot]["runtime"]
                identity = runtime.get("storage_id")
                if identity is not None and (side, identity) not in storage_roles:
                    storage_roles[(side, identity)] = f"Internal_{side}_{len(storage_roles)}"
        for _, event in host_events:
            if event["side"] == side:
                identity = event["output"]["storage_id"]
                if (side, identity) not in storage_roles:
                    storage_roles[(side, identity)] = f"Internal_{side}_{len(storage_roles)}"
    for launch, reads, writes in analyzed:
        side = launch["side"]
        abi: dict[str, Any] = {}
        bindings: dict[str, Any] = {}
        slot_roles = {}
        for slot in launch["slots"]:
            name = f"arg{slot['slot']}"
            runtime = slot["runtime"]
            if runtime["kind"] == "integer":
                bindings[name] = runtime["value"]
            elif runtime["kind"] == "float":
                bindings[name] = {"constant": runtime["value"], "type": "abstract_float"}
            elif runtime["kind"] == "buffer":
                if slot["slot"] in reads and excluded_host_dependency(
                    side, runtime["storage_id"], launch["step"]
                ):
                    diagnostics.append(
                        "host dependency outside ttir-only execution scope: "
                        f"input {side} {name} storage={runtime['storage_id']}"
                    )
                    continue
                element = _element(runtime)
                if element is None:
                    diagnostics.append("unsupported memory element type")
                    continue
                role = storage_roles.get((side, runtime["storage_id"]))
                if (
                    role is None
                    and slot["slot"] not in reads + writes
                    and slot["slot"] in unused_arguments[(side, launch["ordinal"])]
                ):
                    role = unused_storage_roles.setdefault(
                        (side, runtime["storage_id"]), f"Unused_{side}_{len(unused_storage_roles)}"
                    )
                    unused_pointer_evidence.append(
                        {
                            "side": side,
                            "launch": launch["ordinal"],
                            "slot": slot["slot"],
                            "role": role,
                            "storage_id": runtime["storage_id"],
                            "proof": "compiler snapshot contains no SSA use of this argument",
                        }
                    )
                if role is None:
                    diagnostics.append("external input lacks a unique declared storage role")
                    continue
                _append_endpoint(abi, role, name, runtime["offset"])
                role_identities.setdefault((side, role), set()).add(runtime["storage_id"])
                slot_roles[slot["slot"]] = role
                role_type = {"kind": "buffer", "element": element}
                if role in roles and roles[role] != role_type:
                    diagnostics.append("storage role has inconsistent element types")
                roles[role] = role_type
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
    for event_index, event in host_events:
        side = event["side"]
        abi = {}
        valid = True
        output_role = None
        for index, runtime in enumerate(event["inputs"] + [event["output"]]):
            role = storage_roles.get((side, runtime["storage_id"]))
            element = _element(runtime)
            if role is None:
                diagnostics.append("host input lacks a declared producer or input storage role")
                valid = False
                continue
            if element is None:
                diagnostics.append("unsupported host memory element type")
                valid = False
                continue
            if role in abi:
                diagnostics.append("duplicate host endpoints for one role require review")
                valid = False
                continue
            role_type = {"kind": "buffer", "element": element}
            if role in roles and roles[role] != role_type:
                diagnostics.append("storage role has inconsistent element types")
                valid = False
                continue
            roles[role] = role_type
            role_identities.setdefault((side, role), set()).add(runtime["storage_id"])
            abi[role] = {"kind": "block", "name": f"arg{index}", "offset": runtime["offset"]}
            if index == len(event["inputs"]):
                output_role = role
        audit.append({"host_event": event_index, "side": side, "event": event})
        if valid and output_role is not None:
            programs[side].append(
                {
                    "kind": "host_op",
                    "id": f"{side}.host.{event_index}",
                    "file": capture_path.name,
                    "expected_sha256": capture_hash,
                    "event": event_index,
                    "step": event["step"],
                    "abi": abi,
                    "stores": [{"index": 0, "role": output_role}],
                }
            )
    for program in programs.values():
        program.sort(key=lambda launch: launch["step"])
    for side in ("lhs", "rhs"):
        for identity in capture["outputs"].get(side, {}):
            if excluded_host_dependency(side, identity):
                diagnostics.append(
                    "host dependency outside ttir-only execution scope: "
                    f"final output {side} storage={identity}"
                )
    if not programs["lhs"] or not programs["rhs"]:
        diagnostics.append("both launch sequences are required")
    output_roles = [set(capture["outputs"].get(side, {}).values()) for side in ("lhs", "rhs")]
    if output_roles != [{"Output"}, {"Output"}] or type(capture.get("output_numel")) is not int:
        diagnostics.append("single Output role and explicit output_numel are required")
    pair: dict[str, Any] = {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": capture["source"]["test_id"],
            "semantic_mode": "abstract_float",
            "execution_scope": execution_scope,
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
            "require_disjoint": (
                []
                if value_representation
                else sorted(set(roles) - set(unused_storage_roles.values()))
            ),
        },
        "rewrites": [],
    }
    if value_representation:
        pair["observation"]["value_representation"] = value_representation
    if any(role["element"] == "bool" for role in roles.values()):
        # Representation premise, not a fact established by launch capture.
        # It is surfaced in the verifier report alongside the ABI premises.
        pair["metadata"]["target"]["boolean_storage"] = "canonical_i8"
    if "output_views" in capture:
        layouts = {}
        for side in ("lhs", "rhs"):
            view = capture["output_views"].get(side, {}).get("Output")
            if not view or capture["outputs"].get(side, {}).get(view["storage_id"]) != "Output":
                diagnostics.append("output view lacks the declared Output storage")
                continue
            layouts[side] = {key: view[key] for key in ("shape", "stride", "offset")}
        if len(layouts) == 2:
            pair["observation"]["layouts"] = layouts
    # Captured allocation identities establish separation premises, not proved facts.
    identities_by_role: dict[str, set[str]] = {}
    for (_, role), identities in role_identities.items():
        identities_by_role.setdefault(role, set()).update(identities)
    for i, role in enumerate(sorted(roles)):
        for other in sorted(roles)[i + 1 :]:
            if value_representation and not any(
                (side, role) in role_identities and (side, other) in role_identities
                for side in ("lhs", "rhs")
            ):
                continue
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
                if report.get("diagnostic"):
                    diagnostics.append("PairSpec validation: " + report["diagnostic"])
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
    if "host_operations" in capture:
        artifacts.append(
            {"path": capture_path.name, "sha256": capture_hash, "kind": "host_capture"}
        )
    _write(
        root / "manifest.json",
        {
            "format": "etv-benchmark-case-v1",
            "case_id": capture["source"]["test_id"],
            "status": status,
            "generation": {
                "mode": "automatic",
                "tool_version": "0.3.0",
                "execution_scope": execution_scope,
            },
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
            "execution_scope": execution_scope,
            "excluded_capture_diagnostics": excluded_capture_diagnostics,
            "excluded_host_events": (
                list(range(len(capture.get("host_operations", []))))
                if execution_scope == "ttir-only"
                and isinstance(capture.get("host_operations", []), list)
                else []
            ),
            "capture": capture,
            "capture_sha256": capture_hash,
            "pair_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "facts": audit,
            "unused_pointer_evidence": unused_pointer_evidence,
            "output_representation_evidence": representation_evidence,
            "artifacts": artifacts,
            "diagnostics": diagnostics,
            "review": [],
        },
    )
    _write(root / "expected.json", {"status": "UNSPECIFIED"})
    return root / "manifest.json"
