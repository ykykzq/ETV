import json
from pathlib import Path
import shutil

import pytest

from etv.ir import BOOL, Expr, const
from etv.semantics import simplify
from etv.smt import Encoder, QueryResult
from etv.verify import verify

ROOT = Path(__file__).resolve().parents[2]


def case_copy(tmp_path, name="compute_mismatch"):
    target = tmp_path / "case"
    shutil.copytree(ROOT / "examples" / name, target)
    path = target / "pair.json"
    return path, json.loads(path.read_text())


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("alias", "NO_ALIAS_NOT_DECLARED"),
        ("race", "WRITE_RACE"),
        ("coverage", "COVERAGE_MISMATCH"),
        ("undefined", "UNDEFINED_LOAD_REACHABLE"),
        ("float_domain", "FLOAT_DEFINEDNESS_NOT_PROVED"),
        ("resource", "RESOURCE_LIMIT"),
        ("type", "ABI_ROLE_MISSING"),
    ],
)
def test_mandatory_obligations(tmp_path, mutation, reason):
    path, data = case_copy(tmp_path)
    if mutation == "alias":
        data["predicates"]["disjoint"] = []
    elif mutation == "race":
        data["programs"]["lhs"][0]["grid"]["programs"] = 2
    elif mutation == "coverage":
        data["observation"]["numel"] = 5
    elif mutation == "resource":
        data["metadata"]["limits"]["max_instances"] = 3
    elif mutation == "type":
        data["predicates"]["roles"]["Input"]["element"] = "i32"
    else:
        kernel = path.parent / "lhs.ttir"
        text = kernel.read_text()
        if mutation == "undefined":
            text = text.replace("tt.load %pointer, %mask", "tt.load %pointer, %badmask")
        else:
            text = text.replace(
                "tt.store %outptr, %x",
                "%root = math.log %x : tensor<4xf32>\n    tt.store %outptr, %root",
            )
        kernel.write_text(text)
    path.write_text(json.dumps(data))
    result = verify(path)
    assert result.reason == reason, (result.reason, result.diagnostic)
    assert result.status == ("DISPROVED" if mutation in ("race", "coverage") else "UNKNOWN")
    if result.status == "DISPROVED":
        assert result.counterexample and result.counterexample.replay


@pytest.mark.parametrize(
    "mutation,reason",
    [("output_load", "UNDEFINED_LOAD_REACHABLE"), ("no_writer", "INVALID_PAIRSPEC")],
)
def test_parameterized_state_checks(tmp_path, mutation, reason):
    path, _ = case_copy(tmp_path, "add")
    path = path.with_name("pair_parametric.json")
    data = json.loads(path.read_text())
    launch = data["programs"]["lhs"][0]
    if mutation == "output_load":
        kernel = path.parent / launch["file"]
        text = kernel.read_text()
        source = launch["abi"]["Input"]["name"]
        output = launch["abi"]["Output"]["name"]
        text = text.replace("tt.splat %" + source, "tt.splat %" + output)
        kernel.write_text(text)
    else:
        data["predicates"]["roles"]["Internal"] = {"kind": "buffer", "element": "abstract_float"}
        launch["abi"]["Internal"] = launch["abi"].pop("Output")
        launch["stores"][0]["role"] = "Internal"
        data["predicates"]["disjoint"][0].append("Internal")
    path.write_text(json.dumps(data))
    result = verify(path)
    assert (result.status, result.reason) == ("UNKNOWN", reason), result.diagnostic


def test_boolean_simplification_preserves_definedness():
    expression = Expr("and", BOOL, (const(False, BOOL), Expr("undefined", BOOL)))
    assert str(Encoder().defined(simplify(expression))) != "True"


def test_solver_unknown_cannot_prove(tmp_path, monkeypatch):
    path, _ = case_copy(tmp_path)
    monkeypatch.setattr(
        "etv.memory.query", lambda *args, **kwargs: QueryResult("unknown", detail="timeout")
    )
    result = verify(path)
    assert (result.status, result.reason) == ("UNKNOWN", "SMT_UNKNOWN")


def test_same_step_multiple_stores_are_one_launch(tmp_path):
    path, data = case_copy(tmp_path)
    for side in ("lhs", "rhs"):
        launch = data["programs"][side][0]
        launch["stores"].append({"index": 1, "role": "Internal"})
        launch["abi"]["Internal"] = {"kind": "block", "name": "arg2"}
        kernel = path.parent / launch["file"]
        text = (path.parent / "lhs.ttir").read_text() if side == "rhs" else kernel.read_text()
        if side == "rhs":
            text = text.replace("@lhs", "@rhs")
        else:
            text = text.replace("%arg1: !tt.ptr<f32>)", "%arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>)")
            text = text.replace(
                "    tt.return",
                "    %p = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>\n    %q = tt.addptr %p, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>\n    tt.store %q, %x, %mask : tensor<4x!tt.ptr<f32>>\n    tt.return",
            )
        kernel.write_text(text)
    data["predicates"]["roles"]["Internal"] = {"kind": "buffer", "element": "abstract_float"}
    data["predicates"]["disjoint"][0].append("Internal")
    path.write_text(json.dumps(data))
    result = verify(path)
    assert result.status == "PROVED", (result.reason, result.diagnostic)
    assert len(result.launches) == 2 and sum(launch.stores for launch in result.launches) == 4


def test_predicate_dag_reaches_manifest(tmp_path):
    from etv.report import report_json

    report = report_json(verify(ROOT / "examples/multilaunch/pair.json"))
    assert report["status"] == "PROVED"
    ids = {p["id"] for p in report["predicates"] + report["obligations"]}
    ids.update(p["id"] for p in report["inputs"]["programs"])
    ids.update(p["id"] for p in report["rewrites"]["admissions"])
    for fact in report["obligations"] + report["proof"]["roots"]:
        assert set(fact["dependencies"]) <= ids
    assert report["soundness"]["level"] == "formal_under_declared_predicates"
    assert not report["proof"]["egraph"]["independent_certificate"]
