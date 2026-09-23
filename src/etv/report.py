"""JSON is the fact source; Markdown renders exactly those facts."""

from dataclasses import fields, is_dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from . import __version__
from .result import VerificationReport


def serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: serialize(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    return value


def report_json(report: VerificationReport) -> dict[str, Any]:
    inputs = [serialize(i) for i in report.inputs]
    rules = [serialize(rule) for rule in report.rewrites]
    conditional = any(rule.used and rule.admission == "admitted_unverified" for rule in report.rewrites)
    return {
        "schema_version": 1,
        "tool": {"name": "ETV", "version": __version__, "dependencies": dict(report.versions)},
        "run_id": report.run_id, "pair_id": report.pair_id,
        "status": report.status, "reason": report.reason,
        "inputs": {"spec": next((i for i in inputs if i["kind"] == "spec"), None),
                   "programs": [i for i in inputs if i["kind"] == "ttir"],
                   "rewrites": [i for i in inputs if i["kind"] == "rewrite"]},
        "semantics": {"mode": "abstract_float", "target": {"index_bits": report.index_bits}},
        "predicates": serialize(report.predicates), "obligations": serialize(report.obligations),
        "rewrites": {"sources": sorted({r.source for r in report.rewrites}), "admissions": rules,
                     "applications": [r for r in rules if r["matches"]]},
        "proof": {"roots": serialize(report.equalities),
                  "egraph": {"engine": "egglog", "independent_certificate": False,
                             "iterations": sum(e.iterations for e in report.equalities),
                             "max_enodes": max((e.enodes for e in report.equalities), default=0)},
                  "partitions": serialize(report.partitions)},
        "launches": serialize(report.launches),
        "physical_launch_count": len(report.launches),
        "component_count": sum(l.stores for l in report.launches),
        "counterexample": serialize(report.counterexample),
        "unsupported": list(report.unsupported), "diagnostic": report.diagnostic,
        "soundness": {"level": "conditional_on_unverified_rewrites" if conditional else
                      "formal_under_declared_predicates",
                      "trusted_axioms": list(report.trusted_axioms),
                      "does_not_prove": ["IEEE-754 rounding, NaN, infinity or signed zero",
                                         "numerical tolerance equivalence", "host launch correctness",
                                         "independent proof certificate"]},
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [f"# {data['pair_id']}", "", f"**{data['status']}**: `{data['reason']}`", ""]
    for key in ("semantics", "soundness", "inputs", "predicates", "obligations", "rewrites",
                "proof", "launches", "counterexample", "unsupported", "diagnostic", "tool"):
        lines.extend((f"## {key.replace('_', ' ').title()}", "", "```json",
                      json.dumps(data[key], indent=2, sort_keys=True), "```", ""))
    return "\n".join(lines)


def write_report(report: VerificationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = report_json(report)
    (output_dir / "report.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(render_markdown(data))
