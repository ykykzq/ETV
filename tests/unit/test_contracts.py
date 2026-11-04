import copy
import json
from pathlib import Path

import pytest

from etv.errors import InputError
from etv.pairspec import parse_pair_spec


def minimal(path: Path) -> dict:
    (path / "kernel.ttir").write_text("module {}")
    launch = {
        "id": "lhs.0",
        "file": "kernel.ttir",
        "function": "kernel",
        "step": 0,
        "grid": {"programs": 1},
        "stores": [{"index": 0, "role": "Output"}],
        "abi": {"Output": {"kind": "block", "name": "arg0"}},
        "bindings": {},
    }
    right = copy.deepcopy(launch)
    right["id"] = "rhs.0"
    return {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": "unit",
            "semantic_mode": "abstract_float",
            "limits": {
                "smt_timeout_ms": 100,
                "egraph_timeout_ms": 1000,
                "max_iterations": 2,
                "max_enodes": 1000,
            },
        },
        "programs": {"lhs": [launch], "rhs": [right]},
        "assumptions": {"for_llm": []},
        "predicates": {
            "roles": {"Output": {"kind": "buffer", "element": "abstract_float"}},
            "bindings": {},
            "parameters": {},
            "constraints": [],
            "disjoint": [],
            "custom": [],
        },
        "observation": {
            "role": "Output",
            "numel": 1,
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "rewrites": [],
    }


def write(path: Path, data: dict) -> Path:
    spec = path / "pair.json"
    spec.write_text(json.dumps(data))
    return spec


def test_valid(tmp_path):
    spec = parse_pair_spec(write(tmp_path, minimal(tmp_path)))
    assert spec.lhs[0].id == "lhs.0"
    assert {p.id for p in spec.predicates} == {"abi.lhs.0.Output", "abi.rhs.0.Output"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(unknown=True),
        lambda d: d["metadata"].update(unknown=True),
        lambda d: d["programs"]["lhs"][0].update(step=-1),
        lambda d: d["programs"]["lhs"][0].update(file="../escape.ttir"),
        lambda d: d["programs"]["lhs"][0]["abi"].update(Missing={"kind": "block", "name": "arg0"}),
        lambda d: d["programs"]["lhs"][0]["grid"].update(programs={"var": "missing"}),
        lambda d: d["predicates"]["parameters"].update(n={"type": "i8", "min": 0, "max": 128}),
        lambda d: d["predicates"]["bindings"].update(n={"var": "n"}),
        lambda d: d["metadata"].update(llm={"enabled": True}),
        lambda d: d["programs"].update(rhs=[]),
        lambda d: d["observation"].update(numel=True),
    ],
)
def test_strict_contract(tmp_path, mutation):
    data = minimal(tmp_path)
    mutation(data)
    with pytest.raises(InputError):
        parse_pair_spec(write(tmp_path, data))


def test_duplicate_json_key(tmp_path):
    path = tmp_path / "pair.json"
    path.write_text('{"format": "a", "format": "b"}')
    with pytest.raises(InputError, match="duplicate"):
        parse_pair_spec(path)


def test_typed_launch_constant_is_exact_and_not_a_grid_expression(tmp_path):
    from fractions import Fraction
    from etv.ir import FLOAT

    data = minimal(tmp_path)
    data["programs"]["lhs"][0]["bindings"]["arg1"] = {
        "constant": "13421773/134217728",
        "type": "abstract_float",
    }
    spec = parse_pair_spec(write(tmp_path, data))
    bound = dict(spec.lhs[0].bindings)["arg1"]
    assert bound.type == FLOAT and bound.attr("value") == Fraction(13421773, 134217728)
    data["programs"]["lhs"][0]["grid"]["programs"] = {"var": "arg1"}
    with pytest.raises(InputError):
        parse_pair_spec(write(tmp_path, data))


def test_boolean_representation_is_an_explicit_predicate(tmp_path):
    data = minimal(tmp_path)
    data["metadata"]["target"] = {"boolean_storage": "canonical_i8"}
    spec = parse_pair_spec(write(tmp_path, data))
    assert all(x.canonical_boolean_storage for x in spec.lhs + spec.rhs)
    assert "target.boolean_storage" in {p.id for p in spec.predicates}
    data["metadata"]["target"]["boolean_storage"] = "arbitrary_bytes"
    with pytest.raises(InputError):
        parse_pair_spec(write(tmp_path, data))
