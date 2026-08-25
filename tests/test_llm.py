from etv.llm import DeepSeekClient, propose_rules
from etv.model import Expr, Sort
from etv.schema import load_pair_spec


class FakeDeepSeekClient(DeepSeekClient):
    def __init__(self, spec):
        super().__init__(spec)
        self.calls = 0

    def complete_json(self, purpose, system, payload):
        self.calls += 1
        audit = {
            "purpose": purpose,
            "provider": "deepseek",
            "model": "fake-deepseek",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "usage": {},
        }
        if purpose == "node_selection":
            return {
                "pairs": [{"label": "k", "lhs_path": "root", "rhs_path": "root"}]
            }, audit
        return {
            "rules": [
                {
                    "id": "llm_conditional_commute",
                    "kind": "algebraic",
                    "lhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
                    "rhs": {"op": "fadd", "args": [{"match": "b"}, {"match": "a"}]},
                    "requires": [payload["available_fact_requirements"][-1]],
                }
            ]
        }, audit


def test_deepseek_assistance_selects_nodes_and_parses_conditional_rules(tmp_path):
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    source_path = root / "tests/fixtures/semantic/specs/add_parametric_shapes.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["lhs"] = str(root / "tests/fixtures/semantic/programs/add_symbolic_2d.json")
    source["rhs"] = str(root / "tests/fixtures/semantic/programs/add_symbolic_1d.json")
    source["llm"] = {"enabled": True}
    path = tmp_path / "llm.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    spec = load_pair_spec(path)
    lhs = Expr(
        "fadd",
        args=(
            Expr("input", data="A", sort=Sort.FLOAT),
            Expr("input", data="B", sort=Sort.FLOAT),
        ),
        sort=Sort.FLOAT,
    )
    rhs = Expr(
        "fadd",
        args=(
            Expr("input", data="B", sort=Sort.FLOAT),
            Expr("input", data="A", sort=Sort.FLOAT),
        ),
        sort=Sort.FLOAT,
    )
    client = FakeDeepSeekClient(spec)

    assistance = propose_rules(spec, (("k", lhs, rhs),), client=client)

    assert client.calls == 2
    assert assistance.rules[0].rule_id == "llm_conditional_commute"
    assert assistance.rules[0].generated_by == "llm"
    assert assistance.rules[0].fact_requirements[0].kind == "constraint"
    assert assistance.audit["selected_nodes"][0]["label"] == "k"
