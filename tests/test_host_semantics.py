"""Host semantics regressions use symbolic tensor values and no Torch/Triton."""

import hashlib
import json
from fractions import Fraction
from math import prod

import pytest

from etv.host import lift_host
from etv.memory import build_obligations
from etv.pairspec import parse_pair_spec
from etv.semantics import evaluate
from etv.verify import _programs, verify


def buffer(storage, shape=(4,), stride=None, offset=0):
    if stride is None:
        stride = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    return {
        "kind": "buffer",
        "storage_id": storage,
        "offset": offset,
        "element_size": 4,
        "dtype": "torch.float32",
        "shape": list(shape),
        "stride": list(stride),
        "device": "cuda:0",
        "storage_nbytes": 4096,
    }


def scalar(value):
    return {
        "kind": "float",
        "dtype": "torch.float32",
        "value": str(Fraction(value)),
        "encoding": "rational",
        "source": {"kind": "integer", "value": value},
    }


def event(side, step, kind, inputs, output, attributes=None):
    return {
        "side": side,
        "step": step,
        "kind": kind,
        "operator": {
            "copy": "aten.clone.default",
            "constant": "aten.full.default",
            "div_scalar": "aten.div.Tensor",
            "pad": "aten.constant_pad_nd.default",
        }[kind],
        "inputs": inputs,
        "output": output,
        "attributes": attributes or {},
    }


def write_case(tmp_path, events, roles, numel=4):
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "format": "etv-runtime-capture-v1",
                "host_operations": events,
            }
        )
    )
    capture_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
    programs = {"lhs": [], "rhs": []}
    all_roles = set()
    for i, (operation, event_roles) in enumerate(zip(events, roles, strict=True)):
        buffers = operation["inputs"] + [operation["output"]]
        all_roles.update(event_roles)
        programs[operation["side"]].append(
            {
                "kind": "host_op",
                "id": f"host.{i}",
                "file": "capture.json",
                "expected_sha256": capture_hash,
                "event": i,
                "step": operation["step"],
                "abi": {
                    role: {"kind": "block", "name": f"arg{index}", "offset": view["offset"]}
                    for index, (role, view) in enumerate(zip(event_roles, buffers, strict=True))
                },
                "stores": [{"index": 0, "role": event_roles[-1]}],
            }
        )
    data = {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": "host-regression",
            "execution_scope": "ttir-with-host-ops",
            "semantic_mode": "abstract_float",
            "target": {"index_bits": 64},
            "limits": {
                "smt_timeout_ms": 1000,
                "egraph_timeout_ms": 3000,
                "max_iterations": 3,
                "max_enodes": 10000,
            },
        },
        "programs": programs,
        "assumptions": {"for_llm": []},
        "predicates": {
            "roles": {r: {"kind": "buffer", "element": "abstract_float"} for r in all_roles},
            "bindings": {},
            "parameters": {},
            "constraints": [],
            "disjoint": [sorted(all_roles)] if len(all_roles) > 1 else [],
            "custom": [],
        },
        "observation": {
            "role": "Output",
            "numel": numel,
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "rewrites": [],
    }
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(data))
    return path


def division_case(tmp_path, rhs_divisor=4):
    events = [
        event("lhs", 0, "copy", [buffer("x")], buffer("temp_l")),
        event(
            "lhs",
            1,
            "div_scalar",
            [buffer("temp_l")],
            buffer("out_l"),
            {"divisor": scalar(4), "rounding_mode": None},
        ),
        event(
            "rhs",
            0,
            "div_scalar",
            [buffer("x")],
            buffer("temp_r"),
            {"divisor": scalar(rhs_divisor), "rounding_mode": None},
        ),
        event("rhs", 1, "copy", [buffer("temp_r")], buffer("out_r")),
    ]
    roles = [("Input", "TempL"), ("TempL", "Output"), ("Input", "TempR"), ("TempR", "Output")]
    return write_case(tmp_path, events, roles)


def test_both_sides_ordered_copy_division_proves_for_arbitrary_values(tmp_path):
    report = verify(division_case(tmp_path))
    assert report.status == "PROVED", (report.reason, report.diagnostic)
    assert all(x.kind == "host_capture" for x in report.inputs[1:])
    assert all(x.kind == "host_op" for x in report.launches)
    assert all("capture event" in x.detail for x in report.obligations[:4])
    assert not report.trusted_axioms
    assert any("simultaneous writes" in x.detail for x in report.obligations)


def test_wrong_post_divisor_has_counterexample(tmp_path):
    report = verify(division_case(tmp_path, 5))
    assert (report.status, report.reason) == ("DISPROVED", "COMPUTE_MISMATCH")
    assert report.counterexample is not None


def test_copy_preserves_broadcast_stride_and_storage_offset(tmp_path):
    source = buffer("x", (2, 1), (2, 0), offset=3)
    expanded = buffer("x", (2, 3), (2, 0), offset=3)
    events = [
        event("lhs", 0, "copy", [source], buffer("out_l", (2, 3))),
        event("rhs", 0, "copy", [expanded], buffer("out_r", (2, 3))),
    ]
    path = write_case(tmp_path, events, [("Input", "Output")] * 2, 6)
    spec = parse_pair_spec(path)
    memory = build_obligations(_programs(spec, "test"), spec)
    assert memory.status == "closed"
    assert [evaluate(r.lhs.args[0]) for r in memory.roots] == [3, 3, 3, 5, 5, 5]
    assert verify(path).status == "PROVED"


def test_quantile_literal_materialization_is_a_prior_store(tmp_path):
    rounded = {
        "kind": "float",
        "dtype": "torch.float32",
        "value": "1/2",
        "encoding": "rational",
        "source": {"kind": "float", "value": "1/2"},
    }
    events = [
        event("lhs", 0, "constant", [], buffer("q", ()), {"value": rounded}),
        event("lhs", 1, "copy", [buffer("q", ())], buffer("out_l")),
        event("rhs", 0, "constant", [], buffer("out_r"), {"value": rounded}),
    ]
    path = write_case(tmp_path, events, [("Q",), ("Q", "Output"), ("Output",)])
    spec = parse_pair_spec(path)
    memory = build_obligations(_programs(spec, "test"), spec)
    assert memory.status == "closed"
    assert [evaluate(r.lhs) for r in memory.roots] == [Fraction(1, 2)] * 4
    assert verify(path).status == "PROVED"


def test_same_step_cannot_read_an_uncommitted_copy(tmp_path):
    path = division_case(tmp_path)
    data = json.loads(path.read_text())
    capture_path = tmp_path / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["host_operations"][1]["step"] = 0
    capture_path.write_text(json.dumps(capture))
    digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    for launch in data["programs"]["lhs"] + data["programs"]["rhs"]:
        launch["expected_sha256"] = digest
    data["programs"]["lhs"][1]["step"] = 0
    path.write_text(json.dumps(data))
    report = verify(path)
    # The core cannot assume simultaneous snapshot reads establish dependency safety.
    assert (report.status, report.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED")


def test_capture_hash_is_checked(tmp_path):
    path = division_case(tmp_path)
    (tmp_path / "capture.json").write_text("{}")
    assert verify(path).reason == "FILE_HASH_MISMATCH"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda e: e[0]["output"].update(storage_id="x"), "HOST_ALIAS_UNSUPPORTED"),
        (lambda e: e[0]["output"].update(stride=[0]), "HOST_OVERLAPPING_OUTPUT_UNSUPPORTED"),
        (lambda e: e[0]["inputs"][0].update(storage_nbytes=2), "INVALID_PAIRSPEC"),
        (
            lambda e: e[1]["attributes"]["divisor"].update(value="inf", encoding="nonfinite"),
            "HOST_NONFINITE_UNSUPPORTED",
        ),
        (lambda e: e[1]["attributes"].update(rounding_mode="floor"), "HOST_DIVISION_UNSUPPORTED"),
        (
            lambda e: e[1]["attributes"]["divisor"]["source"].update(kind="tensor"),
            "INVALID_PAIRSPEC",
        ),
    ],
)
def test_invalid_host_semantics_are_not_proved(tmp_path, mutation, reason):
    path = division_case(tmp_path)
    capture_path = tmp_path / "capture.json"
    capture = json.loads(capture_path.read_text())
    mutation(capture["host_operations"])
    capture_path.write_text(json.dumps(capture))
    digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    for launch in data["programs"]["lhs"] + data["programs"]["rhs"]:
        launch["expected_sha256"] = digest
    path.write_text(json.dumps(data))
    result = verify(path)
    assert (result.status, result.reason) == ("UNKNOWN", reason), result.diagnostic


def test_finite_padding_masks_the_source_address(tmp_path):
    events = [
        event(
            side,
            0,
            "pad",
            [buffer("x", (2,))],
            buffer("out_" + side, (4,)),
            {"padding": [1, 1], "value": scalar(0)},
        )
        for side in ("lhs", "rhs")
    ]
    path = write_case(tmp_path, events, [("Input", "Output")] * 2)
    spec = parse_pair_spec(path)
    memory = build_obligations(_programs(spec, "test"), spec)
    assert memory.status == "closed"
    assert evaluate(memory.roots[0].lhs) == evaluate(memory.roots[-1].lhs) == 0
    assert [evaluate(r.lhs.args[0]) for r in memory.roots[1:3]] == [0, 1]
    assert verify(path).status == "PROVED"


def test_host_endpoint_cannot_change_captured_offset(tmp_path):
    path = division_case(tmp_path)
    data = json.loads(path.read_text())
    data["programs"]["lhs"][0]["abi"]["Input"]["offset"] = 1
    path.write_text(json.dumps(data))
    report = verify(path)
    assert (report.status, report.reason) == ("UNKNOWN", "INVALID_PAIRSPEC")


def test_host_kernel_has_no_runtime_input_value_constants(tmp_path):
    spec = parse_pair_spec(division_case(tmp_path))
    kernel = lift_host(spec.lhs[0])
    assert kernel.stores[0].value.op == "load"
    assert kernel.stores[0].value.attr("storage").logical_role == "Input"


@pytest.mark.parametrize("lhs_stride,offset", [((1, 2), 0), ((6, 2), 3)])
def test_observation_projects_transposed_and_strided_storage(tmp_path, lhs_stride, offset):
    events = [
        event("lhs", 0, "copy", [buffer("x", (2, 3))], buffer("out_l", (2, 3), lhs_stride, offset)),
        event("rhs", 0, "copy", [buffer("x", (2, 3))], buffer("out_r", (2, 3))),
    ]
    path = write_case(tmp_path, events, [("Input", "Output")] * 2, 6)
    data = json.loads(path.read_text())
    data["observation"]["layouts"] = {
        "lhs": {"shape": [2, 3], "stride": list(lhs_stride), "offset": offset},
        "rhs": {"shape": [2, 3], "stride": [3, 1], "offset": 0},
    }
    path.write_text(json.dumps(data))
    result = verify(path)
    assert result.status == "PROVED", (result.reason, result.diagnostic)
    assert len(result.equalities) == 6
    assert {p.id for p in result.predicates if p.kind == "observation_layout"} == {
        "observation.layout.lhs",
        "observation.layout.rhs",
    }


def test_observation_layout_does_not_hide_permuted_values(tmp_path):
    events = [
        event("lhs", 0, "copy", [buffer("x", (2, 2))], buffer("out_l", (2, 2))),
        event("rhs", 0, "copy", [buffer("x", (2, 2))], buffer("out_r", (2, 2))),
    ]
    path = write_case(tmp_path, events, [("Input", "Output")] * 2)
    data = json.loads(path.read_text())
    data["observation"]["layouts"] = {
        "lhs": {"shape": [2, 2], "stride": [1, 2], "offset": 0},
        "rhs": {"shape": [2, 2], "stride": [2, 1], "offset": 0},
    }
    path.write_text(json.dumps(data))
    result = verify(path)
    assert (result.status, result.reason) == ("DISPROVED", "COMPUTE_MISMATCH")


def test_observation_layout_size_is_checked(tmp_path):
    path = division_case(tmp_path)
    data = json.loads(path.read_text())
    data["observation"]["layouts"] = {
        side: {"shape": [2], "stride": [1], "offset": 0} for side in ("lhs", "rhs")
    }
    path.write_text(json.dumps(data))
    result = verify(path)
    assert (result.status, result.reason) == ("UNKNOWN", "OBSERVATION_LAYOUT_SIZE_MISMATCH")


def test_half_divisor_uses_float32_opmath_scalar(tmp_path):
    events = []
    for side in ("lhs", "rhs"):
        source, output = buffer("x"), buffer("out_" + side)
        for view in (source, output):
            view.update(dtype="torch.float16", element_size=2)
        events.append(
            event(
                side,
                0,
                "div_scalar",
                [source],
                output,
                {"divisor": scalar(2049), "rounding_mode": None},
            )
        )
    path = write_case(tmp_path, events, [("Input", "Output")] * 2)
    spec = parse_pair_spec(path)
    memory = build_obligations(_programs(spec, "test"), spec)
    assert memory.status == "closed"
    assert all(evaluate(root.lhs.args[1]) == 2049 for root in memory.roots)
    assert verify(path).status == "PROVED"


def test_half_literal_checks_float32_intermediate_rounding(tmp_path):
    original = 1 + 2**-11 + 2**-25
    literal = {
        "kind": "float",
        "dtype": "torch.float16",
        "value": "1",
        "encoding": "rational",
        "source": {"kind": "float", "value": str(Fraction(original))},
    }
    events = []
    for side in ("lhs", "rhs"):
        output = buffer("out_" + side)
        output.update(dtype="torch.float16", element_size=2)
        events.append(event(side, 0, "constant", [], output, {"value": literal}))
    path = write_case(tmp_path, events, [("Output",)] * 2)
    assert verify(path).status == "PROVED"
