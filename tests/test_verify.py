import json
from pathlib import Path

import pytest

from etv.model import Expr, Sort, Status
from etv.reporting import write_report
from etv.verify import _definedness_issue, verify_spec


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("add_proved.json", Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT"),
        ("add_fma_proved.json", Status.PROVED, "OBSERVABLE_MEMORY_EQUIVALENT"),
        ("add_bad_stride.json", Status.DISPROVED, "OUTPUT_ADDRESS_MISMATCH"),
        ("add_bad_mask.json", Status.DISPROVED, "MASK_MISMATCH"),
        ("add_bad_compute.json", Status.DISPROVED, "COMPUTE_MISMATCH"),
        ("add_missing_alias.json", Status.UNKNOWN, "MISSING_ALIAS_FACT"),
    ],
)
def test_end_to_end_verdicts(name, status, reason):
    report = verify_spec(ROOT / "examples/specs" / name)

    assert report["status"] == status.value
    assert report["reason"] == reason


def test_compute_counterexample_is_replayable():
    report = verify_spec(ROOT / "examples/specs/add_bad_compute.json")
    witness = report["counterexample"]

    assert witness["kind"] == "ABSTRACT_VALUE_MODEL"
    assert witness["lhs_value"] == "2"
    assert witness["rhs_value"] == "0"


def test_machine_report_is_deterministic():
    path = ROOT / "examples/specs/add_fma_proved.json"

    assert verify_spec(path) == verify_spec(path)


def test_writes_json_and_markdown_artifacts(tmp_path):
    report = verify_spec(ROOT / "examples/specs/add_proved.json")

    write_report(report, tmp_path)

    assert (tmp_path / "report.json").is_file()
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Status: **PROVED**" in markdown
    assert "It does not prove" in markdown


def test_partial_float_operations_require_proved_domains(tmp_path):
    value = Expr("input", data="X", sort=Sort.FLOAT)
    one = Expr("const_float", data=(1, 1), sort=Sort.FLOAT)

    assert _definedness_issue(Expr("fdiv", args=(value, one), sort=Sort.FLOAT)) is None
    assert _definedness_issue(Expr("fdiv", args=(one, value), sort=Sort.FLOAT))["operation"] == "fdiv"
    assert _definedness_issue(Expr("fsqrt", args=(value,), sort=Sort.FLOAT))["operation"] == "fsqrt"

    spec = json.loads((ROOT / "examples/specs/add_proved.json").read_text(encoding="utf-8"))
    lhs = json.loads((ROOT / "examples/programs/add_ntops_2d.json").read_text(encoding="utf-8"))
    rhs = json.loads((ROOT / "examples/programs/add_inductor_linear.json").read_text(encoding="utf-8"))
    lhs["stores"][0]["value"] = {
        "op": "fdiv",
        "args": [lhs["stores"][0]["value"], {"scalar": "nt_alpha"}],
    }
    rhs["stores"][0]["value"] = {
        "op": "fdiv",
        "args": [
            rhs["stores"][0]["value"],
            {
                "op": "load",
                "block": "in_ptr1",
                "offset": 0,
                "mask": True,
                "default": {"float": "0"},
            },
        ],
    }
    spec["lhs"] = "lhs.json"
    spec["rhs"] = "rhs.json"
    (tmp_path / "lhs.json").write_text(json.dumps(lhs), encoding="utf-8")
    (tmp_path / "rhs.json").write_text(json.dumps(rhs), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    report = verify_spec(spec_path)
    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "FLOAT_DEFINEDNESS_NOT_PROVED"
