import json
from pathlib import Path

import yaml

from tools.materialize_ntops_operator_specs import materialize


ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "specs" / "ntops"
EVIDENCE_PATH = ROOT / "specs" / "ntops_test_evidence.json"


def _predicate_groups(spec: dict) -> list[list[dict]]:
    groups = [spec["requires"]["relations"]]
    groups.extend(
        argument["constraints"]
        for argument in spec["requires"]["arguments"].values()
    )
    return groups


def _predicate_ops(value: object) -> set[str]:
    if isinstance(value, dict):
        result = {value["op"]} if "op" in value else set()
        for child in value.values():
            result.update(_predicate_ops(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_predicate_ops(child))
        return result
    return set()


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        result = set()
        for child in value.values():
            result.update(_strings(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_strings(child))
        return result
    return set()


def test_materialized_ntops_specs_are_current_and_complete() -> None:
    materialize(SPEC_ROOT, check=True)

    index = yaml.safe_load((SPEC_ROOT / "index.yaml").read_text(encoding="utf-8"))
    schema = json.loads(
        (SPEC_ROOT / "operator_spec.schema.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    entries = index["operators"]
    assert index["operator_count"] == len(entries) == 75
    assert "common" not in index
    assert not (SPEC_ROOT / "common.yaml").exists()
    assert schema["properties"]["schema"]["const"] == "operator-requires/v1"
    allowed_predicates = set(
        schema["properties"]["requires"]["properties"]["relations"]["items"]
        ["properties"]["op"]["enum"]
    )

    evidence_names = {item["operator"] for item in evidence["operators"]}
    indexed_names = {item["id"].removeprefix("ntops.torch.") for item in entries}
    assert indexed_names == evidence_names

    for entry in entries:
        path = SPEC_ROOT / entry["path"]
        text = path.read_text(encoding="utf-8")
        spec = yaml.safe_load(text)
        name = spec["operator"]["name"]

        assert "&id" not in text and "*id" not in text
        assert set(spec) == {"schema", "operator", "requires", "evidence"}
        assert spec["schema"] == "operator-requires/v1"
        assert spec["operator"]["id"] == entry["id"]
        assert spec["evidence"]["status"] == "inferred_from_tests"
        assert spec["evidence"]["reviewed"] is True
        assert "../" not in text and "#/" not in text
        assert "observed_case_union" not in text

        evidence_module = next(
            item for item in evidence["operators"] if item["operator"] == name
        )
        assert evidence_module["operator"] == name
        assert spec["evidence"]["tests"] == [
            {
                "path": evidence_module["source"],
                "name": test["name"],
                "line": test["line"],
            }
            for test in evidence_module["tests"]
        ]
        assert set(spec["requires"]) == {
            "observed_domain_zh",
            "arguments",
            "relations",
        }
        assert spec["requires"]["observed_domain_zh"]
        assert spec["requires"]["arguments"]
        relation_arguments = _strings(spec["requires"]["relations"])
        for argument_name, argument in spec["requires"]["arguments"].items():
            assert argument["kind"]
            assert isinstance(argument["constraints"], list)
            assert argument["constraints"] or argument_name in relation_arguments
        for group in _predicate_groups(spec):
            assert all(
                isinstance(predicate, dict) and predicate.get("op")
                for predicate in group
            )
            assert _predicate_ops(group) <= allowed_predicates


def test_ntops_spec_index_matches_independent_kernel_modules() -> None:
    index = yaml.safe_load((SPEC_ROOT / "index.yaml").read_text(encoding="utf-8"))
    indexed_names = {item["id"].removeprefix("ntops.torch.") for item in index["operators"]}
    helper_modules = {"__init__", "element_wise", "pooling", "reduction"}
    kernels = {
        path.stem
        for path in (ROOT / "build/upstream/ntops/src/ntops/kernels").glob("*.py")
        if path.stem not in helper_modules
    }

    assert indexed_names == kernels
