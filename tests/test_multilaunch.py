from dataclasses import replace
import json
from pathlib import Path

import pytest

from etv.model import Status
from etv.multilaunch import (
    build_launch_tree,
    compose_launch_value,
    evaluate_launch_sequence,
    flatten_launch_tree,
)
from etv.partition import PartitionError, propose_launch_partition_plan
from etv.schema import load_internal_pair_spec, load_program
from etv.verify import verify_internal_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"
SPEC_PATH = FIXTURES / "specs/multilaunch_proved.json"


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, purpose, system, payload):
        self.calls.append((purpose, system, payload))
        return self.response, {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-launch-partitioner",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "usage": {},
        }


def _response(child_path="root.args[1]"):
    return {
        "partitions": [
            {
                "id": "multiply",
                "family": "family_0",
                "anchor": "root.dep[0]",
                "semantic": "elementwise multiply",
                "counterpart_path": child_path,
            },
            {
                "id": "output",
                "family": "family_0",
                "anchor": "root",
                "semantic": "elementwise add and final store",
                "counterpart_path": "root",
            },
        ]
    }


def _evaluated():
    spec = load_internal_pair_spec(SPEC_PATH)
    programs = tuple(load_program(item.path) for item in spec.launches("lhs"))
    counterpart = load_program(spec.rhs_path)
    sequence = evaluate_launch_sequence(spec, "lhs", spec.launches("lhs"), programs)
    roots = []
    from etv.evaluator import evaluate_program

    other = evaluate_program(counterpart, spec, "rhs")
    other_by_index = {item.logical_index: item for item in other.active}
    semantics = {item.launch_id: item.semantic for item in spec.launches("lhs")}
    for index, produced in sorted(sequence.final.items()):
        roots.append(
            (index, build_launch_tree(produced, semantics), other_by_index[index].value)
        )
    return spec, sequence, counterpart, roots


def test_sequence_evaluation_reconstructs_intermediate_memory_flow():
    spec, sequence, _, roots = _evaluated()

    assert [item.launch_id for item in spec.launches("lhs")] == ["multiply", "add"]
    assert set(sequence.final) == {0, 1, 2, 3}
    nodes = flatten_launch_tree(roots[0][1])
    assert [item.launch_id for item in nodes] == ["add", "multiply"]
    assert roots[0][1].dependencies[0].expression_path == "root.args[1]"
    assert compose_launch_value(sequence.final[0]) == roots[0][2]


def test_same_step_store_is_not_visible_inside_the_kernel():
    spec = load_internal_pair_spec(SPEC_PATH)
    launches = (
        spec.launches("lhs")[0],
        replace(spec.launches("lhs")[1], step=0),
    )
    programs = tuple(load_program(item.path) for item in launches)

    sequence = evaluate_launch_sequence(spec, "lhs", launches, programs)
    root = build_launch_tree(
        sequence.final[0], {item.launch_id: item.semantic for item in launches}
    )

    assert root.launch_id == "add"
    assert root.dependencies == ()
    assert compose_launch_value(sequence.final[0]).args[1].op == "read"


def test_launch_partition_matches_only_counterpart_paths():
    spec, sequence, counterpart, roots = _evaluated()
    client = FakeClient(_response())

    plan = propose_launch_partition_plan(
        spec, sequence, counterpart, roots, client=client
    )

    assert [item.partition_id for item in plan.batches] == ["multiply", "output"]
    assert plan.batches[1].dependencies == ("multiply",)
    assert plan.batches[1].lhs_path.startswith("launch:add:")
    assert plan.batches[1].rhs_path == "root"
    assert plan.batches[1].root_pairs[0][1].args[1].op == "input"
    assert plan.batches[1].root_pairs[0][1] == plan.batches[1].root_pairs[0][2]
    assert plan.audit["machine_checks"]["fixed_launch_boundaries"] is True
    assert client.calls[0][0] == "launch_partition_matching"
    payload = client.calls[0][2]
    assert payload["prepartitioned_side"] == "lhs"
    assert [item["id"] for item in payload["launches"]] == ["multiply", "add"]


def test_launch_partition_rejects_reused_counterpart_boundary():
    spec, sequence, counterpart, roots = _evaluated()
    client = FakeClient(_response(child_path="root"))

    with pytest.raises(PartitionError, match="selected twice"):
        propose_launch_partition_plan(spec, sequence, counterpart, roots, client=client)


def test_launch_partition_rejects_cross_sort_boundary():
    spec, sequence, counterpart, roots = _evaluated()
    client = FakeClient(_response(child_path="root.args[1].args[0].args[0]"))

    with pytest.raises(PartitionError, match="crosses a typed boundary"):
        propose_launch_partition_plan(spec, sequence, counterpart, roots, client=client)


def test_multilaunch_verification_proves_and_composes(monkeypatch):
    calls = []

    def fake_complete(self, purpose, system, payload):
        calls.append((purpose, payload))
        return _response(), {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-launch-partitioner",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "usage": {},
        }

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_internal_spec(SPEC_PATH)

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["partitioning"]["mode"] == ("prepartitioned_launch_sequence")
    assert report["proof"]["partitioning"]["proof_order"] == [
        "multiply",
        "output",
    ]
    assert report["proof"]["launch_sequence"]["ordered_launches"] == [
        "multiply",
        "add",
    ]
    assert len(calls) == 1
    launch_block = next(
        item for item in report["blocks"] if item["kind"] == "LAUNCH_SEQUENCE"
    )
    assert launch_block["status"] == Status.PROVED.value
    store_block = next(item for item in report["blocks"] if item["kind"] == "STORE")
    assert "sequential launch composition" in store_block["summary"]


def test_rhs_multilaunch_verification_uses_lhs_counterpart(tmp_path, monkeypatch):
    source = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    metadata = source["metadata"]
    metadata["lhs"] = str(FIXTURES / "programs/multilaunch_fused.json")
    metadata.pop("rhs", None)
    metadata["frontends"] = {"lhs": {"kind": "semantic_json"}}
    metadata["launches"] = {"rhs": metadata["launches"]["lhs"]}
    for launch in metadata["launches"]["rhs"]:
        launch["file"] = str(FIXTURES / "programs" / Path(launch["file"]).name)
    for value in source["predicates"]["abi"].values():
        value["lhs"], value["rhs"] = value["rhs"], value["lhs"]
    path = tmp_path / "rhs-multilaunch.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    def fake_complete(self, purpose, system, payload):
        assert payload["prepartitioned_side"] == "rhs"
        return _response(), {"purpose": purpose, "provider": "deepseek"}

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_internal_spec(path)

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["launch_sequence"]["prepartitioned_side"] == "rhs"
    subgraphs = report["proof"]["egraph"]["subgraphs"]
    assert subgraphs[0]["egraph"]["root_pairs"] == 4


def test_wrong_llm_match_is_not_used_as_equivalence(monkeypatch):
    def fake_complete(self, purpose, system, payload):
        return _response(child_path="root.args[0]"), {
            "purpose": purpose,
            "provider": "deepseek",
        }

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_internal_spec(SPEC_PATH)

    # The proposed multiply-vs-bias subgraph is rejected by the proof core.
    # Exact sequential composition can still prove the whole fixed program.
    assert report["status"] == Status.PROVED.value
    assert report["proof"]["partitioning"]["status"] == "verification_failed"
    assert report["proof"]["partitioning"]["fallback"] == ("composed_whole_program")
    partition_block = next(
        item for item in report["blocks"] if item["kind"] == "SUBGRAPH_PARTITION"
    )
    assert partition_block["status"] == Status.UNKNOWN.value
    assert partition_block["required_for_final"] is False
