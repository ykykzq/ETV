"""A nodeid alone cannot freeze randomly generated q/p or runtime tensor views."""

import copy
import importlib.util
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).with_name("test_host_recapture.py")
SPEC = importlib.util.spec_from_file_location("host_recapture_argument_fixtures", FIXTURE_PATH)
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
workflow = fixtures.workflow


ARGUMENTS = {
    "q": 0.723417723970258,
    "p": 0.9442055242712242,
    "input": {"shape": [451], "dtype": "torch.float32", "stride": [1], "device": "cuda:0"},
    "out": None,
    "keepdim": False,
}


def prepare_recordings(tmp_path, *, original_arguments=True):
    frozen, baseline = fixtures.make_frozen(tmp_path)
    for path in frozen.glob("*/case_status.json"):
        status = workflow.read(path)
        if original_arguments:
            status["arguments"] = copy.deepcopy(ARGUMENTS)
        else:
            del status["arguments"]
        workflow.write(path, status)
    lock_path = frozen / "corpus.lock.json"
    lock = workflow.read(lock_path)
    lock["files"] = {
        str(path.relative_to(frozen)): workflow.digest(path)
        for path in frozen.rglob("*.json")
        if path != lock_path
    }
    workflow.write(lock_path, lock)
    plan_dir = tmp_path / "plan"
    plan = workflow.prepare(frozen, baseline, plan_dir)
    for target in plan["cases"]:
        fixtures.record(plan_dir, target)
        path = plan_dir / "recaptured" / target["case_id"] / "case_status.json"
        status = workflow.read(path)
        status["arguments"] = copy.deepcopy(ARGUMENTS)
        workflow.write(path, status)
    return frozen, plan_dir, plan


@pytest.mark.parametrize(
    "key,value,path",
    [
        ("q", 0.25, "arguments.q"),
        ("p", 0.5, "arguments.p"),
        ("keepdim", 0, "arguments.keepdim"),
        (
            "out",
            {"shape": [1], "dtype": "torch.float32", "stride": [1], "device": "cuda:0"},
            "arguments.out",
        ),
    ],
)
def test_changed_runtime_scalar_retains_original_capture(tmp_path, key, value, path):
    frozen, plan_dir, plan = prepare_recordings(tmp_path)
    identity = plan["cases"][0]["case_id"]
    status_path = plan_dir / "recaptured" / identity / "case_status.json"
    status = workflow.read(status_path)
    status["arguments"][key] = value
    workflow.write(status_path, status)
    original = (frozen / identity / "capture.json").read_bytes()
    out = tmp_path / "derived"
    audit = workflow.merge(frozen, plan_dir, out)
    failed = audit["cases"][0]
    assert failed["failure"] == "recapture_arguments_mismatch"
    assert not failed["enriched"]
    assert failed["capture_origin"] == "retained_original_after_recapture_failure"
    assert failed["runtime_arguments_comparison"]["differences"][0]["path"] == path
    assert (out / identity / "capture.json").read_bytes() == original
    assert audit["enriched"] == 1 and audit["failed_recaptures"] == 1
    assert (out / "suite.lock.json").read_bytes() == (frozen / "suite.lock.json").read_bytes()
    workflow.verify_frozen(out)


@pytest.mark.parametrize(
    "key,value",
    [
        ("shape", [450]),
        ("stride", [2]),
        ("dtype", "torch.float16"),
        ("device", "cpu"),
    ],
)
def test_changed_tensor_descriptor_is_not_a_recapture_of_same_case(tmp_path, key, value):
    frozen, plan_dir, plan = prepare_recordings(tmp_path)
    status_path = plan_dir / "recaptured" / plan["cases"][0]["case_id"] / "case_status.json"
    status = workflow.read(status_path)
    status["arguments"]["input"][key] = value
    workflow.write(status_path, status)
    audit = workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert audit["cases"][0]["failure"] == "recapture_arguments_mismatch"


def test_cuda_device_ordinal_is_the_only_normalization(tmp_path):
    frozen, plan_dir, plan = prepare_recordings(tmp_path)
    status_path = plan_dir / "recaptured" / plan["cases"][0]["case_id"] / "case_status.json"
    status = workflow.read(status_path)
    status["arguments"]["input"]["device"] = "cuda:3"
    workflow.write(status_path, status)
    audit = workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert audit["enriched"] == 2
    check = audit["cases"][0]["runtime_arguments_comparison"]
    assert check["differences"] == []
    assert (
        check["original_normalized_arguments_sha256"]
        == check["recaptured_normalized_arguments_sha256"]
    )
    assert workflow.normalized_arguments({"device": "cuda:3"}) == {"device": "cuda:3"}


@pytest.mark.parametrize("side", ["original", "recaptured"])
def test_missing_arguments_are_explicit_failure(tmp_path, side):
    frozen, plan_dir, plan = prepare_recordings(tmp_path, original_arguments=side != "original")
    if side == "recaptured":
        path = plan_dir / "recaptured" / plan["cases"][0]["case_id"] / "case_status.json"
        status = workflow.read(path)
        del status["arguments"]
        workflow.write(path, status)
    audit = workflow.merge(frozen, plan_dir, tmp_path / "derived")
    assert audit["cases"][0]["failure"] == f"recapture_{side}_arguments_missing"
    assert not audit["cases"][0]["enriched"]


def test_scalar_comparison_preserves_signed_zero_and_numeric_types():
    assert workflow.argument_differences(0.0, -0.0)
    assert workflow.argument_differences(1, 1.0)
    assert not workflow.argument_differences({"nested": [True, 0.5]}, {"nested": [True, 0.5]})
