import json
from dataclasses import replace
from pathlib import Path

import pytest

from etv.ir import Expr, Sort
from etv.model import LLMConfig, PartitionConfig, Status
from etv.partition import PartitionError, propose_partition_plan
from etv.reporting import render_markdown
from etv.schema import load_internal_pair_spec, load_program
from etv.verify import verify_internal_spec as verify_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, purpose, system, payload):
        self.calls.append((purpose, system, payload))
        return self.response, {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-partitioner",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "usage": {},
        }


def _input(name):
    return Expr("input", data=name, sort=Sort.FLOAT)


def _op(name, *args):
    return Expr(name, args=tuple(args), sort=Sort.FLOAT)


def _partition_spec():
    spec = load_internal_pair_spec(FIXTURES / "specs/add_proved.json")
    return replace(
        spec,
        llm=LLMConfig(enabled=True, generate_rules=False),
        partition=PartitionConfig(enabled=True, min_partitions=2, max_partitions=4),
    )


def test_partition_plan_is_dependency_ordered_and_uses_typed_boundaries():
    spec = _partition_spec()
    lhs_program = load_program(FIXTURES / "programs/add_ntops_2d.json")
    rhs_program = load_program(FIXTURES / "programs/add_inductor_linear.json")
    lhs = _op("fadd", _input("x"), _op("fmul", _input("a"), _input("y")))
    rhs = _op("fadd", _input("x"), _op("fmul", _input("y"), _input("a")))
    client = FakeClient(
        {
            "partitions": [
                {
                    "id": "output",
                    "family": "family_0",
                    "semantic": "scaled add",
                    "lhs_path": "root",
                    "rhs_path": "root",
                },
                {
                    "id": "multiply",
                    "family": "family_0",
                    "semantic": "scale Other",
                    "lhs_path": "root.args[1]",
                    "rhs_path": "root.args[1]",
                },
            ]
        }
    )

    plan = propose_partition_plan(
        spec, lhs_program, rhs_program, [(0, lhs, rhs)], client=client
    )

    assert [item.partition_id for item in plan.batches] == ["multiply", "output"]
    assert plan.batches[1].dependencies == ("multiply",)
    output_lhs = plan.batches[1].root_pairs[0][1]
    output_rhs = plan.batches[1].root_pairs[0][2]
    assert output_lhs.args[1].op == "input"
    assert output_lhs.args[1] == output_rhs.args[1]
    assert output_lhs.args[1].data.startswith("partition:family_0:multiply:")
    assert plan.audit["machine_checks"]["paired_root_coverage"] is True
    assert len(client.calls) == 1
    assert client.calls[0][0] == "program_partitioning"
    assert client.calls[0][2]["programs"]["lhs"]["stores"]
    assert client.calls[0][2]["programs"]["rhs"]["stores"]


def test_partition_plan_rejects_incomplete_root_coverage():
    spec = _partition_spec()
    lhs_program = load_program(FIXTURES / "programs/add_ntops_2d.json")
    rhs_program = load_program(FIXTURES / "programs/add_inductor_linear.json")
    lhs = _op(
        "fadd",
        _op("fmul", _input("a"), _input("x")),
        _op("fmul", _input("b"), _input("y")),
    )
    rhs = lhs
    client = FakeClient(
        {
            "partitions": [
                {
                    "id": "left",
                    "family": "family_0",
                    "semantic": "left product",
                    "lhs_path": "root.args[0]",
                    "rhs_path": "root.args[0]",
                },
                {
                    "id": "right",
                    "family": "family_0",
                    "semantic": "right product",
                    "lhs_path": "root.args[1]",
                    "rhs_path": "root.args[1]",
                },
            ]
        }
    )

    with pytest.raises(PartitionError, match="paired root partition"):
        propose_partition_plan(
            spec, lhs_program, rhs_program, [(0, lhs, rhs)], client=client
        )


def test_partition_plan_rejects_different_dependency_topologies():
    spec = replace(
        _partition_spec(),
        partition=PartitionConfig(enabled=True, min_partitions=3, max_partitions=4),
    )
    lhs_program = load_program(FIXTURES / "programs/add_ntops_2d.json")
    rhs_program = load_program(FIXTURES / "programs/add_inductor_linear.json")
    lhs = _op(
        "fadd",
        _op("fmul", _input("a"), _input("x")),
        _op("fmul", _input("b"), _op("fmul", _input("c"), _input("y"))),
    )
    rhs = lhs
    client = FakeClient(
        {
            "partitions": [
                {
                    "id": "root_graph",
                    "family": "family_0",
                    "semantic": "output",
                    "lhs_path": "root",
                    "rhs_path": "root",
                },
                {
                    "id": "parent",
                    "family": "family_0",
                    "semantic": "parent multiply",
                    "lhs_path": "root.args[1]",
                    "rhs_path": "root.args[0]",
                },
                {
                    "id": "child",
                    "family": "family_0",
                    "semantic": "child multiply",
                    "lhs_path": "root.args[1].args[1]",
                    "rhs_path": "root.args[1]",
                },
            ]
        }
    )

    with pytest.raises(PartitionError, match="inconsistent lhs/rhs dependency"):
        propose_partition_plan(
            spec, lhs_program, rhs_program, [(0, lhs, rhs)], client=client
        )


def _write_partition_spec(tmp_path):
    source = json.loads(
        (FIXTURES / "specs/add_proved.json").read_text(encoding="utf-8")
    )
    source["lhs"] = str(FIXTURES / "programs/add_ntops_2d.json")
    source["rhs"] = str(FIXTURES / "programs/add_inductor_linear.json")
    source["llm"] = {"enabled": True, "generate_rules": False}
    source["partition"] = {
        "enabled": True,
        "min_partitions": 2,
        "max_partitions": 4,
    }
    path = tmp_path / "partition.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _valid_response():
    return {
        "partitions": [
            {
                "id": "multiply",
                "family": "family_0",
                "semantic": "scale Other",
                "lhs_path": "root.args[1]",
                "rhs_path": "root.args[1]",
            },
            {
                "id": "output",
                "family": "family_0",
                "semantic": "add Input",
                "lhs_path": "root",
                "rhs_path": "root",
            },
        ]
    }


def test_partitioned_verification_proves_subgraphs_and_composes(tmp_path, monkeypatch):
    calls = []

    def fake_complete(self, purpose, system, payload):
        calls.append((purpose, payload))
        return _valid_response(), {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-partitioner",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "usage": {},
        }

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_spec(_write_partition_spec(tmp_path))

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["partitioning"]["status"] == "proved"
    assert report["proof"]["partitioning"]["proof_order"] == [
        "multiply",
        "output",
    ]
    assert report["proof"]["egraph"]["stats"]["phase"] == (
        "PAIRED_SUBGRAPH_DECOMPOSITION"
    )
    assert len(report["proof"]["egraph"]["subgraphs"]) == 2
    assert len(calls) == 1
    assert calls[0][0] == "program_partitioning"
    partition_block = next(
        item for item in report["blocks"] if item["kind"] == "SUBGRAPH_PARTITION"
    )
    assert partition_block["status"] == Status.PROVED.value
    assert "DECOMPOSITION" in partition_block["proof_levels"]
    markdown = render_markdown(report)
    assert "### Subgraph Partitioning" in markdown
    assert "`multiply`" in markdown
    assert "`output`" in markdown


def test_rejected_partition_falls_back_to_whole_program(tmp_path, monkeypatch):
    def fake_complete(self, purpose, system, payload):
        return {"partitions": [_valid_response()["partitions"][1]]}, {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-partitioner",
        }

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_spec(_write_partition_spec(tmp_path))

    assert report["status"] == Status.PROVED.value
    assert report["proof"]["partitioning"]["status"] == "proposal_rejected"
    assert report["proof"]["partitioning"]["fallback"] == "whole_program"
    assert report["proof"]["partitioning"]["whole_program_calls"] == 1
    assert report["proof"]["partitioning"]["call"]["purpose"] == (
        "program_partitioning"
    )
    partition_block = next(
        item for item in report["blocks"] if item["kind"] == "SUBGRAPH_PARTITION"
    )
    assert partition_block["required_for_final"] is False
    assert partition_block["status"] == Status.UNKNOWN.value
