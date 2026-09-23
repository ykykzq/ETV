import json
from pathlib import Path
import shutil

import pytest

from etv.report import report_json
from etv.verify import verify

ROOT = Path(__file__).resolve().parents[2]


def candidate(kind):
    variable = {"pvar": "x", "type": "abstract_float"}
    return {
        "format": "etv-rewrite-v1",
        "rules": [
            {
                "id": "user.drop_increment",
                "kind": kind,
                "lhs": {
                    "op": "fadd",
                    "type": "abstract_float",
                    "args": [variable, {"const": "1", "type": "abstract_float"}],
                },
                "rhs": variable,
                "requires": ["disjoint.Input.Output"],
                "validation": {"method": "z3"},
            }
        ],
    }


@pytest.mark.parametrize("kind,expected", [("algebraic", "DISPROVED"), ("layout", "PROVED")])
def test_rule_claim_does_not_determine_formality(tmp_path, kind, expected):
    shutil.copytree(ROOT / "examples/compute_mismatch", tmp_path / "case")
    case = tmp_path / "case"
    path = case / "pair.json"
    data = json.loads(path.read_text())
    data["rewrites"] = [{"file": "rules.json"}]
    path.write_text(json.dumps(data))
    (case / "rules.json").write_text(json.dumps(candidate(kind)))
    report = verify(path)
    assert report.status == expected, (report.reason, report.diagnostic)
    fact = next(f for f in report.rewrites if f.id == "user.drop_increment")
    assert fact.sha256
    if kind == "layout":
        assert fact.used and fact.matches
        assert fact.admission == "admitted_unverified"
        assert report_json(report)["soundness"]["level"] == "conditional_on_unverified_rewrites"
        assert "user.drop_increment" in report.trusted_axioms
    else:
        assert fact.admission == "rejected" and not fact.used


def test_llm_failure_falls_back_to_whole_goal(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "examples/multilaunch", tmp_path / "case")
    path = tmp_path / "case/pair.json"
    data = json.loads(path.read_text())
    data["metadata"]["llm"] = {
        "enabled": True,
        "provider": "openai_compatible",
        "model": "unit",
        "base_url_env": "ETV_TEST_URL",
        "api_key_env": "ETV_TEST_KEY",
        "timeout_ms": 100,
    }
    data["metadata"]["partition"] = {"enabled": True, "provider": "llm", "max_parts": 2}
    path.write_text(json.dumps(data))
    monkeypatch.delenv("ETV_TEST_KEY", raising=False)
    report = verify(path)
    assert report.status == "PROVED", (report.reason, report.diagnostic)
    assert any(p.status == "fallback" for p in report.partitions)
    assert not report.trusted_axioms


@pytest.mark.parametrize(
    "case,components", [("argsort", 3), ("rms_norm", 3), ("batch_norm", 4), ("quantile", 4)]
)
def test_real_multilaunch_regions_and_counts(case, components):
    report = verify(ROOT / "examples/real" / case / "pair.json")
    assert (report.status, report.reason) == ("UNKNOWN", "TTIR_REGION_UNSUPPORTED")
    assert len(report.launches) == 3
    assert sum(launch.stores for launch in report.launches) == components
    first, second = report.launches[1:]
    assert set(first.writes) <= set(second.reads)
