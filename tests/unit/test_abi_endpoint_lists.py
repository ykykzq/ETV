"""One storage role may have several actual pointer arguments with distinct views."""

import pytest

from etv.errors import InputError
from etv.pairspec import parse_pair_spec
from test_contracts import minimal, write


def multiple_endpoints(tmp_path):
    data = minimal(tmp_path)
    data["predicates"]["roles"]["Input"] = {"kind": "buffer", "element": "abstract_float"}
    for launch in (data["programs"]["lhs"][0], data["programs"]["rhs"][0]):
        launch["abi"]["Input"] = [
            {"kind": "block", "name": "arg1", "offset": 0},
            {"kind": "block", "name": "arg2", "offset": 4},
        ]
    return data


def test_same_storage_endpoints_keep_individual_offsets_and_unique_predicates(tmp_path):
    spec = parse_pair_spec(write(tmp_path, multiple_endpoints(tmp_path)))
    endpoints = [ep for ep in spec.lhs[0].abi if ep.role == "Input"]
    assert [(ep.name, ep.offset) for ep in endpoints] == [("arg1", 0), ("arg2", 4)]
    assert {p.id for p in spec.predicates} == {
        "abi.lhs.0.Input.arg1",
        "abi.lhs.0.Input.arg2",
        "abi.lhs.0.Output",
        "abi.rhs.0.Input.arg1",
        "abi.rhs.0.Input.arg2",
        "abi.rhs.0.Output",
    }
    assert spec.disjoint == ()  # Parsing aliases never creates a separation premise.


@pytest.mark.parametrize(
    "endpoints",
    [
        [],
        [None],
        ["arg1"],
        [[{"kind": "block", "name": "arg1"}]],
        [{"kind": "block", "name": "arg1"}, {"kind": "block", "name": "arg1"}],
        [{"kind": "block", "name": "arg0"}],  # Duplicate across roles.
    ],
)
def test_malformed_or_duplicate_physical_endpoints_are_rejected(tmp_path, endpoints):
    data = multiple_endpoints(tmp_path)
    data["programs"]["lhs"][0]["abi"]["Input"] = endpoints
    with pytest.raises(InputError):
        parse_pair_spec(write(tmp_path, data))


def test_single_element_list_retains_legacy_predicate_id(tmp_path):
    data = minimal(tmp_path)
    abi = data["programs"]["lhs"][0]["abi"]
    abi["Output"] = [abi["Output"]]
    spec = parse_pair_spec(write(tmp_path, data))
    assert {p.id for p in spec.predicates} == {"abi.lhs.0.Output", "abi.rhs.0.Output"}


def test_real_ttir_alias_views_read_same_storage_offsets(tmp_path):
    pytest.importorskip("triton._C.libtriton")
    pytest.importorskip("etv._mlir_native")
    from etv.verify import verify

    data = multiple_endpoints(tmp_path)
    data["predicates"]["disjoint"] = [["Input", "Output"]]
    # LHS arg2 is a view beginning at element 4 of the same actual storage.
    # RHS addresses element 4 through the base pointer instead.
    for side in ("lhs", "rhs"):
        view = "%p = tt.addptr %arg1, %four : !tt.ptr<f32>, i32" if side == "rhs" else ""
        other = "%p" if side == "rhs" else "%arg2"
        (tmp_path / f"{side}.ttir").write_text(f"""module {{
          tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {{
            %four = arith.constant 4 : i32
            {view}
            %a = tt.load %arg1 : !tt.ptr<f32>
            %b = tt.load {other} : !tt.ptr<f32>
            %sum = arith.addf %a, %b : f32
            tt.store %arg0, %sum : !tt.ptr<f32>
            tt.return
          }}
        }}""")
        data["programs"][side][0]["file"] = f"{side}.ttir"
    report = verify(write(tmp_path, data))
    assert report.status == "PROVED", (report.reason, report.diagnostic)
    assert len({p.id for p in report.predicates}) == len(report.predicates)
