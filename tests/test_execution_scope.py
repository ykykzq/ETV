"""Execution scope must not silently admit modeled host operations."""

from dataclasses import replace
import json

import pytest

from etv.errors import InputError
from etv.pairspec import parse_pair_spec
from etv.report import report_json
from etv.verify import _programs, verify
from test_host_semantics import buffer, event, scalar, write_case


def host_pair(tmp_path):
    events = [
        event(side, 0, "constant", [], buffer(side), {"value": scalar(0)})
        for side in ("lhs", "rhs")
    ]
    return write_case(tmp_path, events, [("Output",), ("Output",)])


@pytest.mark.parametrize("scope", [None, "ttir-only"])
def test_strict_scope_rejects_host_step_before_semantic_lifting(tmp_path, scope):
    path = host_pair(tmp_path)
    data = json.loads(path.read_text())
    if scope is None:
        del data["metadata"]["execution_scope"]
    else:
        data["metadata"]["execution_scope"] = scope
    path.write_text(json.dumps(data))
    report = verify(path)
    assert report.execution_scope == "ttir-only"
    assert (report.status, report.reason) == ("UNKNOWN", "HOST_STEP_OUTSIDE_TTIR_ONLY")
    assert not report.launches and not report.equalities


def test_host_composition_requires_explicit_opt_in_and_is_labeled(tmp_path):
    path = host_pair(tmp_path)
    report = report_json(verify(path))
    assert report["status"] == "PROVED"
    assert report["execution_scope"] == "ttir-with-host-ops"
    assert report["physical_launch_count"] == 0
    assert report["execution_step_count"] == 2
    assert len(report["inputs"]["host_operations"]) == 2
    # A programmatic caller also cannot bypass the scope boundary.
    strict = replace(parse_pair_spec(path), execution_scope="ttir-only")
    with pytest.raises(InputError, match="host semantics excluded"):
        _programs(strict, "scope-regression")


def test_unknown_scope_is_not_treated_as_host_opt_in(tmp_path):
    path = host_pair(tmp_path)
    data = json.loads(path.read_text())
    data["metadata"]["execution_scope"] = "all"
    path.write_text(json.dumps(data))
    report = verify(path)
    assert (report.status, report.reason) == ("UNKNOWN", "INVALID_PAIRSPEC")
    assert not report.equalities
