import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from etv.frontend import parse_ttir, operations
from etv.report import report_json
from etv.verify import verify

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "case", ["add", "compute_mismatch", "mask_mismatch", "address_mismatch", "multilaunch"]
)
def test_acceptance(case):
    path = ROOT / "examples" / case
    report = verify(path / "pair.json")
    expected = json.loads((path / "expected.json").read_text())
    assert report.status == expected["status"], (report.reason, report.diagnostic)
    if "reason" in expected:
        assert report.reason == expected["reason"]
        assert report.counterexample is not None and report.counterexample.replay
    if report.status == "PROVED":
        assert report.predicates and report.obligations and report.equalities
        assert all(e.merged and e.dependencies for e in report.equalities)
        assert not report.trusted_axioms
        assert report_json(report)["soundness"]["level"] == "formal_under_declared_predicates"


def test_snapshot_is_structured():
    module = parse_ttir(ROOT / "examples/add/lhs-0.ttir", "ntops_add_kernel")
    ops = operations(module.module)
    assert len({op.ordinal for op in ops}) == len(ops)
    assert all(op.assembly and op.location for op in ops)
    assert module.function.regions[0][0].arguments[0].type.kind == "pointer"
    assert any(op.attribute("value").kind == "dense" for op in ops if op.name == "arith.constant")


def test_parameterized_add():
    report = verify(ROOT / "examples/add/pair_parametric.json")
    assert report.status == "PROVED", (report.reason, report.diagnostic, report.obligations[-1:])
    assert any(p.id == "shape.product" for p in report.predicates)


def test_same_step_producer_consumer_needs_explicit_ordering(tmp_path):
    shutil.copytree(ROOT / "examples/multilaunch", tmp_path / "case")
    path = tmp_path / "case/pair.json"
    spec = json.loads(path.read_text())
    spec["programs"]["rhs"][1]["step"] = 0
    path.write_text(json.dumps(spec))
    report = verify(path)
    # Sharing a step does not establish launch ordering or cross-program safety.
    assert (report.status, report.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED")


def test_stdout_json_and_exit_code(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "etv",
            "verify",
            str(ROOT / "examples/compute_mismatch/pair.json"),
            "--out",
            str(tmp_path),
            "--json",
            "--log-format",
            "jsonl",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["status"] == "DISPROVED"
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert events[-1]["event"] == "verification_finished"
    assert all(
        {"time", "level", "event", "run_id", "pair_id", "phase", "fields"} <= e.keys()
        for e in events
    )
    assert json.loads((tmp_path / "report.json").read_text())["reason"] == "COMPUTE_MISMATCH"


@pytest.mark.parametrize("bad_mask", [False, True])
def test_fixed_large_grid_symbolic_proof_and_replay_fallback(tmp_path, bad_mask):
    shutil.copytree(ROOT / "examples/compute_mismatch", tmp_path / "case")
    path = tmp_path / "case/pair.json"
    data = json.loads(path.read_text())
    data["observation"]["numel"] = 512
    text = (path.parent / "lhs.ttir").read_text()
    text = (
        text.replace("end = 4", "end = 512")
        .replace("tensor<4x", "tensor<512x")
        .replace("dense<4>", "dense<512>")
    )
    (path.parent / "lhs.ttir").write_text(text)
    right = text.replace("@lhs", "@rhs")
    if bad_mask:
        right = right.replace("tt.store %outptr, %x, %mask", "tt.store %outptr, %x, %badmask")
    (path.parent / "rhs.ttir").write_text(right)
    path.write_text(json.dumps(data))
    report = verify(path)
    assert report.status == ("DISPROVED" if bad_mask else "PROVED"), (
        report.reason,
        report.diagnostic,
    )
    if bad_mask:
        assert report.reason == "MASK_MISMATCH" and report.counterexample.index == "3"
    else:
        assert len(report.equalities) == 1
        assert any(
            o.id == "memory.lhs.uniqueness" and o.status == "unsat" for o in report.obligations
        )
