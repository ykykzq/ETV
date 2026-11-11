"""Pair generation preserves the runtime producer graph and its provenance."""

import hashlib
import json
from types import SimpleNamespace

from etv_bench.pair import pair_capture


def buffer(identity, shape=(4,), stride=(1,)):
    return {
        "kind": "buffer",
        "storage_id": identity,
        "dtype": "torch.float32",
        "offset": 0,
        "shape": list(shape),
        "stride": list(stride),
        "element_size": 4,
    }


def fixture(tmp_path, monkeypatch):
    import etv_bench.pair as module

    monkeypatch.setattr(module, "_snapshot", lambda *args: {})
    monkeypatch.setattr(module, "_memory_slots", lambda *args: ([0], [1], 2))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"reason": "OBSERVABLE_MEMORY_EQUIVALENT"})
        ),
    )
    data = {
        "format": "etv-runtime-capture-v1",
        "host_capture_format": "etv-host-operations-v1",
        "source": {"test_id": "host-pair", "commit": "pinned"},
        "environment": {},
        "launches": [],
        "input_roles": {"input": "Input"},
        "outputs": {"lhs": {"output-lhs": "Output"}, "rhs": {"output-rhs": "Output"}},
        "output_numel": 4,
        "host_operations": [
            {
                "side": "lhs",
                "step": 0,
                "operator": "aten.clone.default",
                "kind": "copy",
                "inputs": [buffer("input")],
                "output": buffer("derived"),
                "attributes": {},
            },
            {
                "side": "lhs",
                "step": 2,
                "operator": "aten.copy_.default",
                "kind": "copy",
                "inputs": [buffer("temp")],
                "output": buffer("output-lhs"),
                "attributes": {},
            },
        ],
    }
    for side in ("lhs", "rhs"):
        path = tmp_path / f"{side}.ttir"
        path.write_text("official parser is mocked in this pairing-only unit test")
        source = "derived" if side == "lhs" else "input"
        destination = "temp" if side == "lhs" else "output-rhs"
        data["launches"].append(
            {
                "side": side,
                "step": 1 if side == "lhs" else 0,
                "ordinal": 0,
                "file": path.name,
                "function": "copy",
                "grid": [1, 1, 1],
                "selected_runtime_launch": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "slots": [
                    {
                        "slot": i,
                        "source_name": f"arg{i}",
                        "compiler_type": "*fp32",
                        "runtime": buffer(identity),
                    }
                    for i, identity in enumerate((source, destination))
                ],
            }
        )
    return data


def generate(tmp_path, data, *, execution_scope="ttir-only"):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(data))
    manifest = pair_capture(
        path,
        ["unused-in-mocked-unit-test"],
        execution_scope=execution_scope,
    )
    return (
        json.loads(manifest.read_text()),
        json.loads((tmp_path / "pair.json").read_text()),
        json.loads((tmp_path / "provenance.json").read_text()),
    )


def test_host_producers_and_output_postprocessing_are_ordered(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    manifest, pair, provenance = generate(tmp_path, data, execution_scope="ttir-with-host-ops")
    assert manifest["status"] == "ready", provenance["diagnostics"]
    launches = pair["programs"]["lhs"]
    assert [launch["step"] for launch in launches] == [0, 1, 2]
    assert [launch.get("kind", "ttir") for launch in launches] == ["host_op", "ttir", "host_op"]
    assert launches[0]["event"] == 0 and launches[2]["event"] == 1
    assert launches[0]["expected_sha256"] == provenance["capture_sha256"]
    assert launches[0]["stores"][0]["role"] in launches[1]["abi"]
    assert launches[1]["stores"][0]["role"] in launches[2]["abi"]
    assert launches[2]["stores"] == [{"index": 0, "role": "Output"}]
    assert manifest["artifacts"][-1]["kind"] == "host_capture"
    assert provenance["capture"] == data
    assert pair["metadata"]["execution_scope"] == "ttir-with-host-ops"


def test_unknown_host_source_cannot_be_invented_as_input(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["host_operations"][0]["inputs"][0]["storage_id"] = "hidden-rng-state"
    manifest, _, provenance = generate(tmp_path, data, execution_scope="ttir-with-host-ops")
    assert manifest["status"] == "needs_review"
    assert "host input lacks a declared producer or input storage role" in provenance["diagnostics"]


def test_colliding_host_and_kernel_steps_cannot_be_reordered(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["host_operations"][0]["step"] = 1
    manifest, _, provenance = generate(tmp_path, data, execution_scope="ttir-with-host-ops")
    assert manifest["status"] == "needs_review"
    assert any("unique ordered step" in diagnostic for diagnostic in provenance["diagnostics"])


def test_noncontiguous_final_view_has_explicit_observation_projection(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["output_views"] = {
        "lhs": {"Output": buffer("output-lhs", (2, 2), (1, 2))},
        "rhs": {"Output": buffer("output-rhs", (2, 2), (2, 1))},
    }
    manifest, pair, provenance = generate(tmp_path, data, execution_scope="ttir-with-host-ops")
    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert pair["observation"]["layouts"] == {
        "lhs": {"shape": [2, 2], "stride": [1, 2], "offset": 0},
        "rhs": {"shape": [2, 2], "stride": [2, 1], "offset": 0},
    }


def test_unsupported_host_event_is_not_silently_dropped(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["host_operations"][0]["kind"] = "unsupported"
    manifest, _, provenance = generate(tmp_path, data, execution_scope="ttir-with-host-ops")
    assert manifest["status"] == "needs_review"
    assert any("unsupported host operation" in item for item in provenance["diagnostics"])


def test_default_ttir_only_excludes_host_producers_and_preserves_evidence(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    manifest, pair, provenance = generate(tmp_path, data)
    assert manifest["status"] == "needs_review"
    assert pair["metadata"]["execution_scope"] == "ttir-only"
    assert all("kind" not in step for steps in pair["programs"].values() for step in steps)
    assert any(
        "host dependency outside ttir-only execution scope: input lhs arg0" in diagnostic
        for diagnostic in provenance["diagnostics"]
    )
    assert any(
        "host dependency outside ttir-only execution scope: final output lhs" in diagnostic
        for diagnostic in provenance["diagnostics"]
    )
    assert provenance["excluded_host_events"] == [0, 1]
    assert provenance["capture"] == data
    assert manifest["artifacts"][-1]["kind"] == "host_capture"
    assert pair["observation"]["require_full_coverage"] is True
    assert pair["observation"]["numel"] == 4
    assert pair["predicates"]["constraints"] == pair["predicates"]["custom"] == []


def test_ttir_only_does_not_add_zero_initialization_or_shrink_diag_observation(
    tmp_path, monkeypatch
):
    data = fixture(tmp_path, monkeypatch)
    data["launches"][0]["slots"][0]["runtime"] = buffer("input")
    data["launches"][0]["slots"][1]["runtime"] = buffer("output-lhs")
    data["host_operations"] = [
        {
            "side": "lhs",
            "step": 0,
            "operator": "aten.zero_.default",
            "kind": "constant",
            "inputs": [],
            "output": buffer("output-lhs"),
            "attributes": {"value": 0},
        }
    ]
    data["output_views"] = {
        "lhs": {"Output": buffer("output-lhs", (2, 2), (2, 1))},
        "rhs": {"Output": buffer("output-rhs", (2, 2), (2, 1))},
    }
    manifest, pair, provenance = generate(tmp_path, data)
    assert manifest["status"] == "ready", provenance["diagnostics"]
    assert len(pair["programs"]["lhs"]) == len(pair["programs"]["rhs"]) == 1
    assert pair["programs"]["lhs"][0]["step"] == 1
    assert pair["programs"]["lhs"][0]["bindings"] == {}
    assert pair["observation"] == {
        "role": "Output",
        "numel": 4,
        "require_full_coverage": True,
        "require_disjoint": ["Input", "Output"],
        "layouts": {
            side: {"shape": [2, 2], "stride": [2, 1], "offset": 0} for side in ("lhs", "rhs")
        },
    }
    assert pair["predicates"]["bindings"] == {}
    assert pair["predicates"]["constraints"] == pair["predicates"]["custom"] == []
    assert provenance["excluded_host_events"] == [0]


def test_ttir_only_cannot_reuse_input_role_after_host_mutation(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["launches"][0]["slots"][0]["runtime"] = buffer("input")
    data["launches"][0]["slots"][1]["runtime"] = buffer("output-lhs")
    data["host_operations"] = [
        {
            "side": "lhs",
            "step": 0,
            "operator": "aten.fill_.Scalar",
            "kind": "constant",
            "inputs": [],
            "output": buffer("input"),
            "attributes": {"value": 0},
        }
    ]
    manifest, pair, provenance = generate(tmp_path, data)
    assert manifest["status"] == "needs_review"
    assert not pair["programs"]["lhs"]
    assert any(
        "host dependency outside ttir-only execution scope: input lhs arg0" in diagnostic
        for diagnostic in provenance["diagnostics"]
    )


def test_unknown_execution_scope_cannot_fall_back_to_host_semantics(tmp_path, monkeypatch):
    import pytest

    data = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unsupported execution scope"):
        generate(tmp_path, data, execution_scope="anything-else")


def test_strict_scope_rejects_ambiguous_host_ttir_ordering(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    data["host_operations"][0]["step"] = data["launches"][0]["step"]
    manifest, _, provenance = generate(tmp_path, data)
    assert manifest["status"] == "needs_review"
    assert (
        "host evidence outside ttir-only scope has incomplete effect provenance"
        in provenance["diagnostics"]
    )
