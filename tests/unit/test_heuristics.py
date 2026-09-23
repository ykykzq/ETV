from dataclasses import replace

import pytest

from etv.heuristics import propose
from etv.ir import FLOAT, RootPair, const, var
from etv.pairspec import LLMConfig, PartitionConfig, parse_pair_spec
from test_contracts import minimal, write


@pytest.mark.parametrize(
    "response",
    [None, {"path": "../../rules.json"}, {"candidates": [{"lhs": "../../x", "rhs": "rhs.0"}]}],
)
def test_invalid_llm_response_falls_back(tmp_path, monkeypatch, response):
    spec = replace(parse_pair_spec(write(tmp_path, minimal(tmp_path))), llm=LLMConfig(enabled=True))
    monkeypatch.setattr("etv.heuristics._request", lambda *a, **k: response)
    result = propose(spec, (RootPair(var("x", FLOAT), var("x", FLOAT), const(0)),))
    assert result.diagnostic and not result.rules and result.plan is None


def test_invalid_topology_and_gate_fall_back(tmp_path, monkeypatch):
    spec = replace(
        parse_pair_spec(write(tmp_path, minimal(tmp_path))),
        llm=LLMConfig(enabled=True),
        partition=PartitionConfig(True, "llm", 2),
    )
    responses = iter(
        [
            {"candidates": [{"lhs": "lhs.0", "rhs": "rhs.0"}]},
            {
                "rewrites": {"format": "etv-rewrite-v1", "rules": []},
                "partition": {
                    "parts": [
                        {"id": "p", "lhs": "lhs.0", "rhs": "rhs.0", "dependencies": ["future"]}
                    ]
                },
            },
        ]
    )
    monkeypatch.setattr("etv.heuristics._request", lambda *a, **k: next(responses))
    result = propose(spec, (RootPair(var("x", FLOAT), var("x", FLOAT), const(0)),))
    assert result.diagnostic and result.plan is None
