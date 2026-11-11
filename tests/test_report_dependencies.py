"""Large enumerated proofs retain complete audit references without quadratic JSON."""

import hashlib
import json

from etv.report import report_json
from etv.verify import verify
from test_host_semantics import buffer, division_case, event, scalar, write_case


def assert_complete_dependency_graph(data):
    nodes = {
        item["id"]: item.get("dependencies", [])
        for item in data["obligations"] + data["proof"]["roots"] + data["proof"]["partitions"]
    }
    leaves = {item["id"] for item in data["predicates"]}
    leaves.update(item["id"] for item in data["rewrites"]["admissions"])
    leaves.add(data["inputs"]["spec"]["id"])
    for items in (data["inputs"]["programs"], data["inputs"]["host_operations"]):
        leaves.update(item["id"] for item in items)
    visited = set()

    def visit(name, ancestors):
        assert name not in ancestors, f"dependency cycle at {name}"
        assert name in nodes or name in leaves, f"dangling dependency {name}"
        if name in visited or name in leaves:
            return
        for dependency in nodes[name]:
            visit(dependency, ancestors | {name})
        visited.add(name)

    for name in nodes:
        visit(name, set())


def test_enumerated_report_has_linear_shared_memory_dependencies(tmp_path):
    size = 128
    events = [
        event(side, 0, "constant", [], buffer("out_" + side, (size,)), {"value": scalar(3)})
        for side in ("lhs", "rhs")
    ]
    path = write_case(tmp_path, events, [("Output",), ("Output",)], size)
    report = verify(path)
    assert report.status == "PROVED", (report.reason, report.diagnostic)
    data = report_json(report)
    group = next(item for item in data["obligations"] if item["id"] == "memory.closed")
    assert set(group["dependencies"]) == {
        item["id"] for item in data["obligations"] if item["id"] != "memory.closed"
    }
    roots = data["proof"]["roots"]
    assert len(roots) == size
    assert all(root["dependencies"] == ["memory.closed"] for root in roots)
    assert_complete_dependency_graph(data)
    compact_size = len(json.dumps(data))
    for root in roots:
        root["dependencies"] = group["dependencies"]
    assert len(json.dumps(data)) > 4 * compact_size


def test_shared_dependency_fact_preserves_counterexample_verdict(tmp_path):
    report = verify(division_case(tmp_path, rhs_divisor=5))
    assert (report.status, report.reason) == ("DISPROVED", "COMPUTE_MISMATCH")
    assert report.counterexample is not None
    assert any(fact.id == "memory.closed" for fact in report.obligations)
    data = report_json(report)
    assert_complete_dependency_graph(data)
    counterexample = next(
        item for item in data["obligations"] if item["id"].startswith("value.counterexample.")
    )
    assert "memory.closed" in counterexample["dependencies"]


def test_used_rewrite_remains_a_direct_equality_dependency(tmp_path):
    path = division_case(tmp_path)
    capture_path = tmp_path / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["host_operations"][2]["attributes"]["divisor"] = scalar(2)
    second = capture["host_operations"][3]
    second.update(kind="div_scalar", operator="aten.div.Tensor")
    second["attributes"] = {"divisor": scalar(2), "rounding_mode": None}
    capture_path.write_text(json.dumps(capture))
    digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    for launch in data["programs"]["lhs"] + data["programs"]["rhs"]:
        launch["expected_sha256"] = digest
    data["rewrites"] = [{"file": "rules.json"}]
    path.write_text(json.dumps(data))
    variable = {"pvar": "x", "type": "abstract_float"}

    def divided(value, divisor):
        return {
            "op": "fdiv",
            "type": "abstract_float",
            "args": [value, {"const": str(divisor), "type": "abstract_float"}],
        }

    (tmp_path / "rules.json").write_text(
        json.dumps(
            {
                "format": "etv-rewrite-v1",
                "rules": [
                    {
                        "id": "user.divide_twice",
                        "kind": "algebraic",
                        "lhs": divided(divided(variable, 2), 2),
                        "rhs": divided(variable, 4),
                        "requires": [],
                        "validation": {"method": "z3"},
                    }
                ],
            }
        )
    )
    report = verify(path)
    assert report.status == "PROVED", (report.reason, report.diagnostic)
    rule = next(fact for fact in report.rewrites if fact.id == "user.divide_twice")
    assert rule.used and rule.matches > 0
    data = report_json(report)
    assert all(
        "memory.closed" in root["dependencies"] and rule.id in root["dependencies"]
        for root in data["proof"]["roots"]
    )
    assert_complete_dependency_graph(data)
