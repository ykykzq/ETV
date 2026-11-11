"""Automatic TTIR pairing preserves physical storage aliases and their offsets."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from etv.pairspec import parse_pair_spec
from etv_bench.pair import pair_capture


def buffer(identity, *, offset=0, dtype="torch.float32", element_size=4):
    return {
        "kind": "buffer",
        "storage_id": identity,
        "dtype": dtype,
        "element_size": element_size,
        "offset": offset,
        "shape": [4],
        "stride": [1],
    }


def capture_fixture(tmp_path, monkeypatch):
    import etv_bench.pair as module

    slots_by_file = {}

    def snapshot(root, launch, command):
        reads, writes, arity = slots_by_file[launch["file"]]
        return {
            "_test_file": launch["file"],
            "regions": [
                [
                    {
                        "arguments": [{"id": f"arg{i}"} for i in range(arity)],
                        "operations": [
                            {"operands": [{"id": f"arg{i}"}], "regions": []} for i in reads + writes
                        ],
                    }
                ]
            ],
        }

    monkeypatch.setattr(module, "_snapshot", snapshot)
    monkeypatch.setattr(
        module, "_memory_slots", lambda function: slots_by_file[function["_test_file"]]
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"reason": "OBSERVABLE_MEMORY_EQUIVALENT"})
        ),
    )
    capture = {
        "format": "etv-runtime-capture-v1",
        "source": {"test_id": "storage-aliases", "commit": "pinned"},
        "environment": {},
        "launches": [],
        "input_roles": {"input": "Input"},
        "outputs": {"lhs": {"output-lhs": "Output"}, "rhs": {"output-rhs": "Output"}},
        "output_numel": 4,
    }

    def launch(side, buffers, reads, writes, *, step=0):
        path = tmp_path / f"{side}-{step}.ttir"
        path.write_text("TTIR parser is mocked in these automatic-pairing unit tests")
        slots_by_file[path.name] = (reads, writes, len(buffers))
        capture["launches"].append(
            {
                "side": side,
                "step": step,
                "ordinal": step,
                "file": path.name,
                "function": "kernel",
                "grid": [1, 1, 1],
                "selected_runtime_launch": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "slots": [
                    {
                        "slot": index,
                        "source_name": f"ptr{index}",
                        "compiler_type": "*fp32",
                        "runtime": runtime,
                    }
                    for index, runtime in enumerate(buffers)
                ],
            }
        )

    return capture, launch


def generate(tmp_path, capture):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    manifest = pair_capture(path, ["mock-etv"])
    return (
        json.loads(manifest.read_text()),
        json.loads((tmp_path / "pair.json").read_text()),
        json.loads((tmp_path / "provenance.json").read_text()),
    )


def endpoint(name, offset=0):
    return {"kind": "block", "name": name, "offset": offset}


def test_same_input_storage_preserves_all_physical_endpoints_and_view_offsets(
    tmp_path, monkeypatch
):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("input", offset=7), buffer("output-lhs")], [0, 1], [2])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert pair["programs"]["lhs"][0]["abi"] == {
        "Input": [endpoint("arg0"), endpoint("arg1", 7)],
        "Output": endpoint("arg2"),
    }
    assert pair["programs"]["rhs"][0]["abi"]["Input"] == endpoint("arg0")
    assert pair["predicates"]["disjoint"] == [["Input", "Output"]]
    assert pair["predicates"]["constraints"] == pair["predicates"]["custom"] == []
    assert pair["metadata"]["execution_scope"] == "ttir-only"
    assert pair["observation"]["require_full_coverage"] is True
    assert len(provenance["facts"]) == 5
    parsed = parse_pair_spec(tmp_path / "pair.json")
    assert [(ep.name, ep.offset) for ep in parsed.lhs[0].abi if ep.role == "Input"] == [
        ("arg0", 0),
        ("arg1", 7),
    ]


def test_multiple_output_pointers_retain_both_selected_stores(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch(
        "lhs",
        [buffer("input"), buffer("output-lhs"), buffer("output-lhs", offset=2)],
        [0],
        [1, 2],
    )
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert pair["programs"]["lhs"][0]["abi"]["Output"] == [
        endpoint("arg1"),
        endpoint("arg2", 2),
    ]
    assert pair["programs"]["lhs"][0]["stores"] == [
        {"index": 0, "role": "Output"},
        {"index": 1, "role": "Output"},
    ]


def test_earlier_ttir_producer_can_be_read_and_written_through_aliases(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("temp")], [0], [1])
    launch("lhs", [buffer("temp"), buffer("temp", offset=2)], [0], [1], step=1)
    launch("lhs", [buffer("temp"), buffer("output-lhs")], [0], [1], step=2)
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    steps = pair["programs"]["lhs"]
    role = steps[0]["stores"][0]["role"]
    assert steps[1]["abi"] == {role: [endpoint("arg0"), endpoint("arg1", 2)]}
    assert steps[1]["stores"] == [{"index": 0, "role": role}]
    assert steps[2]["abi"][role] == endpoint("arg0")
    assert all(len(set(group)) == len(group) for group in pair["predicates"]["disjoint"])


@pytest.mark.parametrize(
    "changed",
    [
        {"dtype": "torch.float16", "element_size": 2},
        {"dtype": "torch.float32", "element_size": 2},
    ],
)
def test_aliases_cannot_reinterpret_the_element_address_space(tmp_path, monkeypatch, changed):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    alias = buffer("input") | changed
    launch("lhs", [buffer("input"), alias, buffer("output-lhs")], [0, 1], [2])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, _, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert "aliased storage has inconsistent element interpretation" in provenance["diagnostics"]


def test_shared_shapes_or_source_names_do_not_invent_an_external_input_relation(
    tmp_path, monkeypatch
):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("unknown"), buffer("output-lhs")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert "external input lacks a unique declared storage role" in provenance["diagnostics"]
    assert pair["programs"]["lhs"] == []
    assert set(pair["predicates"]["roles"]) == {"Input", "Output"}


def test_output_alias_of_public_input_still_requires_explicit_contract(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    capture["outputs"]["lhs"] = {"input": "Output"}
    launch("lhs", [buffer("input"), buffer("input")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, _, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert (
        "output aliases an input; explicit reviewed contract required" in provenance["diagnostics"]
    )


def test_unreferenced_pointer_uses_side_private_role_without_input_equality(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    for side in ("lhs", "rhs"):
        launch(side, [buffer("unknown"), buffer("input"), buffer(f"output-{side}")], [1], [2])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    evidence = provenance["unused_pointer_evidence"]
    assert len(evidence) == 2
    assert evidence[0]["role"] != evidence[1]["role"]
    assert evidence[0]["role"].startswith("Unused_lhs_")
    assert evidence[1]["role"].startswith("Unused_rhs_")
    assert all(record["slot"] == 0 for record in evidence)
    assert pair["predicates"]["constraints"] == pair["predicates"]["custom"] == []


def test_unreferenced_pointer_does_not_authorize_a_later_read(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("unknown"), buffer("input"), buffer("output-lhs")], [1], [2])
    launch("lhs", [buffer("unknown"), buffer("output-lhs")], [0], [1], step=1)
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert "external input lacks a unique declared storage role" in provenance["diagnostics"]
    assert len(pair["programs"]["lhs"]) == 1


@pytest.mark.parametrize("operator", ["eq", "ne", "lt", "le", "gt", "ge", "isinf", "isnan"])
def test_predicate_outputs_use_explicit_injective_bool_encoding(tmp_path, monkeypatch, operator):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    capture["source"]["test_id"] = f"tests/test_{operator}.py::test_{operator}[shape0-dtype0]"
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch(
        "rhs",
        [buffer("input"), buffer("output-rhs", dtype="torch.bool", element_size=1)],
        [0],
        [1],
    )
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    representation = pair["observation"]["value_representation"]
    assert representation == {
        "kind": "boolean_numeric_01",
        "lhs": {"role": "Output_lhs", "encoding": "identity"},
        "rhs": {"role": "Output_rhs", "encoding": "bool_to_numeric_01"},
    }
    assert pair["programs"]["lhs"][0]["stores"] == [{"index": 0, "role": "Output_lhs"}]
    assert pair["programs"]["rhs"][0]["stores"] == [{"index": 0, "role": "Output_rhs"}]
    assert pair["predicates"]["disjoint"] == [["Input", "Output_lhs"], ["Input", "Output_rhs"]]
    assert pair["observation"]["require_disjoint"] == []
    assert pair["observation"]["role"] == "Output"
    assert pair["observation"]["numel"] == 4
    assert pair["observation"]["require_full_coverage"] is True
    assert pair["metadata"]["target"]["boolean_storage"] == "canonical_i8"
    evidence = provenance["output_representation_evidence"]
    assert evidence["operator_source"] == "capture.source.test_id"
    assert evidence["operator"] == operator
    assert evidence["value_representation"] == representation
    parsed = parse_pair_spec(tmp_path / "pair.json")
    assert parsed.observation.role_for_side("lhs") == "Output_lhs"
    assert parsed.observation.role_for_side("rhs") == "Output_rhs"
    assert parsed.observation.encoding_for_side("rhs") == "bool_to_numeric_01"


def test_runtime_operator_evidence_can_encode_the_opposite_output_direction(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    capture["source"]["operator"] = "eq"
    launch(
        "lhs",
        [buffer("input"), buffer("output-lhs", dtype="torch.bool", element_size=1)],
        [0],
        [1],
    )
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert pair["observation"]["value_representation"]["lhs"]["encoding"] == "bool_to_numeric_01"
    assert pair["observation"]["value_representation"]["rhs"]["encoding"] == "identity"
    assert (
        provenance["output_representation_evidence"]["operator_source"] == "capture.source.operator"
    )


@pytest.mark.parametrize(
    "source",
    [
        {"test_id": "tests/test_add.py::test_add[shape0]"},
        {"test_id": "tests/test_eq.py::test_add[shape0]"},
        {"test_id": "tests/test_eq.py::test_eq[shape0]", "operator": "add"},
    ],
)
def test_dtype_difference_alone_does_not_authorize_boolean_encoding(tmp_path, monkeypatch, source):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    capture["source"].update(source)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch(
        "rhs",
        [buffer("input"), buffer("output-rhs", dtype="torch.bool", element_size=1)],
        [0],
        [1],
    )
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert "storage role has inconsistent element types" in provenance["diagnostics"]
    assert "value_representation" not in pair["observation"]
    assert provenance["output_representation_evidence"] is None


def unsupported_host_event(capture, *, output="unobserved", inputs=(), writes=False, step=1):
    event = {
        "side": "lhs",
        "step": step,
        "operator": "aten.unmodeled.default",
        "kind": "unsupported",
        "inputs": [buffer(identity) for identity in inputs],
        "output": buffer(output),
        "attributes": {"writes": writes, "reason": "no host semantics"},
    }
    diagnostic = (
        f"unsupported host operation (lhs step {step}): aten.unmodeled.default: no host semantics"
    )
    capture["host_capture_format"] = "etv-host-operations-v1"
    capture["host_operations"] = [event]
    capture["diagnostics"] = [diagnostic]
    return event, diagnostic


def test_unrelated_unsupported_host_output_does_not_block_ttir_only(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    _, diagnostic = unsupported_host_event(capture, inputs=("output-lhs",))
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert provenance["excluded_capture_diagnostics"] == [diagnostic]
    assert provenance["capture"]["diagnostics"] == [diagnostic]
    assert provenance["excluded_host_events"] == [0]
    assert len(pair["programs"]["lhs"]) == 1
    assert pair["programs"]["lhs"][0].get("kind", "ttir") == "ttir"
    assert pair["observation"]["require_full_coverage"] is True


@pytest.mark.parametrize("mutates_input_argument", [False, True])
def test_unsupported_host_effect_on_ttir_input_still_blocks(
    tmp_path, monkeypatch, mutates_input_argument
):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1], step=1)
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    unsupported_host_event(
        capture,
        output="unobserved" if mutates_input_argument else "input",
        inputs=("input",) if mutates_input_argument else (),
        writes=mutates_input_argument,
        step=0,
    )
    manifest, pair, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert any(
        item.startswith("host dependency outside ttir-only execution scope: input lhs arg0")
        for item in provenance["diagnostics"]
    )
    assert pair["programs"]["lhs"] == []


def test_host_mutation_after_last_ttir_read_is_outside_observation(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    unsupported_host_event(capture, output="input", inputs=("input",), writes=True)
    manifest, _, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "ready", provenance["diagnostics"]


@pytest.mark.parametrize("missing", ["writes", "output"])
def test_incomplete_host_effect_provenance_cannot_be_ignored(tmp_path, monkeypatch, missing):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    event, _ = unsupported_host_event(capture)
    del (event["attributes"] if missing == "writes" else event)[missing]
    manifest, _, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert (
        "host evidence outside ttir-only scope has incomplete effect provenance"
        in provenance["diagnostics"]
    )


def test_unmatched_host_or_other_capture_diagnostic_is_not_suppressed(tmp_path, monkeypatch):
    capture, launch = capture_fixture(tmp_path, monkeypatch)
    launch("lhs", [buffer("input"), buffer("output-lhs")], [0], [1])
    launch("rhs", [buffer("input"), buffer("output-rhs")], [0], [1])
    _, diagnostic = unsupported_host_event(capture)
    capture["diagnostics"] += [
        "unsupported host operation (lhs step 9): unrecorded mutation",
        "capture mapping incomplete: ValueError",
    ]
    manifest, _, provenance = generate(tmp_path, capture)

    assert manifest["status"] == "needs_review"
    assert provenance["diagnostics"] == capture["diagnostics"][1:]
    assert provenance["excluded_capture_diagnostics"] == [diagnostic]
