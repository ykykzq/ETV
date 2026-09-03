import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module(tmp_path, monkeypatch):
    monkeypatch.setenv("ETV_BENCHMARK_CASE_DIR", str(tmp_path / "case"))
    path = ROOT / "tools/capture_benchmark_test.py"
    spec = importlib.util.spec_from_file_location("capture_benchmark_test_unit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_snapshot_preserves_storage_alias_and_view_offsets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    capture = _load_capture_module(tmp_path, monkeypatch)
    base = torch.arange(12, dtype=torch.float32)
    view = base[3:9]

    base_description = capture._json_value(base, origin="base")
    view_description = capture._json_value(view, origin="view")
    snapshot = capture.ValueSnapshot.capture(["values"], {"values": (base, view)})
    restored_base, restored_view = snapshot.restore()["values"]
    restored_base_description = capture._json_value(
        restored_base, origin="restored_base"
    )
    restored_view_description = capture._json_value(
        restored_view, origin="restored_view"
    )

    base_provenance = base_description["provenance"]
    view_provenance = view_description["provenance"]
    assert base_provenance["storage_id"] == view_provenance["storage_id"]
    assert base_provenance["tensor_id"] != view_provenance["tensor_id"]
    assert base_provenance["storage_offset"] == 0
    assert view_provenance["storage_offset"] == 3
    assert (
        restored_base_description["provenance"]["storage_id"]
        == base_provenance["storage_id"]
    )
    assert (
        restored_view_description["provenance"]["storage_id"]
        == view_provenance["storage_id"]
    )
    assert (
        restored_view_description["provenance"]["tensor_id"]
        == view_provenance["tensor_id"]
    )
    assert restored_view.storage_offset() == 3


def _tensor_description(
    *, tensor_id, storage_id, storage_offset=0, shape=(8,), role="input"
):
    return {
        "kind": "tensor",
        "shape": list(shape),
        "stride": [1] if shape else [],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "provenance": {
            "tensor_id": tensor_id,
            "storage_id": storage_id,
            "alias_group": storage_id,
            "storage_offset": storage_offset,
            "runtime_role": role,
        },
    }


def _module(name, argument_types):
    return SimpleNamespace(
        function=name,
        arguments=tuple(
            SimpleNamespace(index=index, value=SimpleNamespace(type=type_text))
            for index, type_text in enumerate(argument_types)
        ),
    )


def test_pairspec_uses_selected_dependencies_not_total_pointer_count(tmp_path):
    from tools.generate_benchmark_pairspecs import PointerInfo, StoreInfo, _pair_spec

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    lhs_path = case_dir / "lhs.ttir"
    rhs_path = case_dir / "rhs.ttir"
    lhs_path.write_text("lhs", encoding="utf-8")
    rhs_path.write_text("rhs", encoding="utf-8")
    common_lhs = PointerInfo(
        0,
        _tensor_description(
            tensor_id="tensor-input",
            storage_id="storage-input",
            storage_offset=4,
        ),
    )
    unused_lhs = _tensor_description(
        tensor_id="tensor-unused", storage_id="storage-unused"
    )
    common_rhs = PointerInfo(
        0,
        _tensor_description(tensor_id="tensor-input", storage_id="storage-input"),
    )
    lhs_output = PointerInfo(
        2,
        _tensor_description(
            tensor_id="tensor-lhs-output",
            storage_id="storage-lhs-output",
            role="unclassified_output_or_scratch",
        ),
    )
    rhs_output = PointerInfo(
        1,
        _tensor_description(
            tensor_id="tensor-rhs-output",
            storage_id="storage-rhs-output",
            role="output",
        ),
    )
    lhs_module = _module("lhs", ("!tt.ptr<f32>", "!tt.ptr<f32>", "!tt.ptr<f32>"))
    rhs_module = _module("rhs", ("!tt.ptr<f32>", "!tt.ptr<f32>"))
    lhs_launch = {
        "programs": 1,
        "runtime_arguments": [
            {"position": 0, "value": common_lhs.description},
            {"position": 1, "value": unused_lhs},
            {"position": 2, "value": lhs_output.description},
        ],
    }
    rhs_launch = {
        "programs": 1,
        "runtime_arguments": [
            {"position": 0, "value": common_rhs.description},
            {"position": 1, "value": rhs_output.description},
        ],
    }

    spec, mapping = _pair_spec(
        case_dir=case_dir,
        lhs_path=lhs_path,
        rhs_path=rhs_path,
        lhs_module=lhs_module,
        rhs_module=rhs_module,
        lhs_launch=lhs_launch,
        rhs_launch=rhs_launch,
        lhs_store=StoreInfo(
            0,
            lhs_output,
            (common_lhs, PointerInfo(1, unused_lhs)),
        ),
        rhs_store=StoreInfo(0, rhs_output, (common_rhs,)),
        leaf={
            "leaf_index": 0,
            "path": "compiled_output[0]",
            "value": rhs_output.description,
        },
        nodeid="tests/test_op.py::test_op[x]",
        pair_id="tests/test_op.py::test_op[x]::r000-e000-o000",
    )

    assert set(spec["predicates"]["abi"]) == {"Input0", "Output"}
    assert spec["predicates"]["abi"]["Input0"]["lhs"]["offset"] == 4
    assert "offset" not in spec["predicates"]["abi"]["Input0"]["rhs"]
    assert mapping["roles"]["Input0"]["lhs_tensor_id"] == "tensor-input"
    assert mapping["unmapped_selected_dependencies"] == {"lhs": [1], "rhs": []}
    assert spec["metadata"]["frontends"]["lhs"]["store_index"] == 0


def test_runner_expands_multiple_observation_specs(tmp_path):
    from tools.run_benchmark_verification import _tasks

    case_dir = tmp_path / "op" / "case"
    status_path = case_dir / "pairspec" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "format": "etv-benchmark-pairspec-status-v2",
                "nodeid": "tests/test_op.py::test_op[x]",
                "case_dir": str(case_dir),
                "status": "generated",
                "pairspecs": [
                    {"pair_id": "pair-0", "pairspec": str(case_dir / "pair-0.json")},
                    {"pair_id": "pair-1", "pairspec": str(case_dir / "pair-1.json")},
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = _tasks(status_path)

    assert [task["pair_id"] for task in tasks] == ["pair-0", "pair-1"]
    assert all(task["kind"] == "pairspec" for task in tasks)


def test_multilaunch_generator_emits_prepartitioned_rhs_sequence(tmp_path):
    from tools.generate_benchmark_pairspecs import (
        PointerInfo,
        StoreInfo,
        _multilaunch_pair_spec,
    )

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    lhs_path = case_dir / "lhs.ttir"
    rhs0_path = case_dir / "rhs0.ttir"
    rhs1_path = case_dir / "rhs1.ttir"
    for path in (lhs_path, rhs0_path, rhs1_path):
        path.write_text(path.stem, encoding="utf-8")

    bias = _tensor_description(tensor_id="bias", storage_id="bias")
    x = _tensor_description(tensor_id="x", storage_id="x")
    y = _tensor_description(tensor_id="y", storage_id="y")
    lhs_out = _tensor_description(
        tensor_id="lhs-out", storage_id="lhs-out", role="output"
    )
    tmp = _tensor_description(tensor_id="tmp", storage_id="tmp", role="scratch")
    rhs_out = _tensor_description(
        tensor_id="rhs-out", storage_id="rhs-out", role="output"
    )
    lhs_module = _module("lhs", ("!tt.ptr<f32>",) * 4)
    rhs0_module = _module("rhs0", ("!tt.ptr<f32>",) * 3)
    rhs1_module = _module("rhs1", ("!tt.ptr<f32>",) * 3)
    for module in (lhs_module, rhs0_module, rhs1_module):
        module.operations = ()
    lhs_launch = {
        "programs": 1,
        "runtime_arguments": [
            {"value": bias},
            {"value": x},
            {"value": y},
            {"value": lhs_out},
        ],
    }
    rhs0_launch = {
        "programs": 1,
        "runtime_arguments": [{"value": x}, {"value": y}, {"value": tmp}],
    }
    rhs1_launch = {
        "programs": 1,
        "runtime_arguments": [
            {"value": bias},
            {"value": tmp},
            {"value": rhs_out},
        ],
    }
    lhs_store = StoreInfo(
        0,
        PointerInfo(3, lhs_out),
        (PointerInfo(0, bias), PointerInfo(1, x), PointerInfo(2, y)),
    )
    rhs0_store = StoreInfo(
        0,
        PointerInfo(2, tmp),
        (PointerInfo(0, x), PointerInfo(1, y)),
    )
    rhs1_store = StoreInfo(
        0,
        PointerInfo(2, rhs_out),
        (PointerInfo(0, bias), PointerInfo(1, tmp)),
    )
    rhs_nodes = [
        (0, rhs0_path, rhs0_module, rhs0_launch, rhs0_store),
        (1, rhs1_path, rhs1_module, rhs1_launch, rhs1_store),
    ]

    spec, mapping = _multilaunch_pair_spec(
        case_dir=case_dir,
        lhs_path=lhs_path,
        lhs_module=lhs_module,
        lhs_launch=lhs_launch,
        lhs_store=lhs_store,
        rhs_nodes=rhs_nodes,
        final_node=rhs_nodes[1],
        leaf={"leaf_index": 0, "path": "output", "value": rhs_out},
        nodeid="tests/test_op.py::test_op[x]",
        pair_id="multi",
    )

    launches = spec["metadata"]["launches"]["rhs"]
    assert [item["id"] for item in launches] == [
        "launch000.store000",
        "launch001.store000",
    ]
    assert "Internal0" in launches[0]["abi"]
    assert "Internal0" in launches[1]["abi"]
    assert spec["metadata"]["partition"]["enabled"] is True
    assert spec["metadata"]["llm"]["enabled"] is True
    assert mapping["mode"] == "prepartitioned_rhs_launch_sequence"
