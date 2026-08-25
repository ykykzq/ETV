"""Deterministic JSON and Markdown proof artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def render_markdown(report: Mapping[str, Any]) -> str:
    status = report.get("status", "UNKNOWN")
    lines = [
        f"# ETV verification report: {report.get('pair_id', '<unknown>')}",
        "",
        f"- Status: **{status}**",
        f"- Reason: `{report.get('reason', 'UNKNOWN')}`",
        f"- Semantic mode: `{report.get('semantic_mode')}`",
        f"- Scope: {report.get('scope', '')}",
        "",
        "## Assumptions",
        "",
    ]
    assumptions = report.get("assumptions", [])
    if assumptions:
        lines.extend(
            f"- `{item.get('evidence')}`: {item.get('text')} ({item.get('source')})"
            for item in assumptions
        )
    else:
        lines.append("- None loaded.")

    lines.extend(
        [
            "",
            "## Proof blocks",
            "",
            "| Block | Status | Evidence | Summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for block in report.get("blocks", []):
        evidence = ", ".join(block.get("proof_levels", [])) or "-"
        summary = str(block.get("summary", "")).replace("|", "\\|")
        lines.append(f"| `{block.get('kind')}` | **{block.get('status')}** | {evidence} | {summary} |")

    counterexample = report.get("counterexample")
    if counterexample:
        lines.extend(
            [
                "",
                "## Counterexample",
                "",
                "```json",
                json.dumps(counterexample, indent=2, sort_keys=True, ensure_ascii=False),
                "```",
            ]
        )

    proof = report.get("proof", {})
    if proof:
        lines.extend(["", "## Proof summary", ""])
        finite = proof.get("finite_domain")
        if finite:
            lines.append(
                f"Finite domain: {finite.get('active_output_elements')} active outputs, "
                f"{finite.get('lhs_lanes')} lhs lanes, {finite.get('rhs_lanes')} rhs lanes."
            )
        egraph = proof.get("egraph")
        if egraph:
            stats = egraph.get("stats", {})
            backend = stats.get("backend", {})
            lines.append(
                f"E-graph: `{backend.get('name')} {backend.get('version')}`, "
                f"{stats.get('enodes')} e-nodes, {stats.get('eclasses')} e-classes, "
                f"{stats.get('iterations')} iterations, stop reason `{stats.get('stop_reason')}`."
            )
            rules = stats.get("rule_matches", {})
            lines.append("Matched rules: " + (", ".join(f"`{key}` x{value}" for key, value in rules.items()) or "none"))
            trusted = egraph.get("trusted_rule_uses", [])
            if trusted:
                lines.extend(["", "### Unverified trusted rewrites", ""])
                lines.extend(
                    f"- `{item.get('id')}` matched {item.get('matches')} time(s): "
                    f"{item.get('validation', {}).get('warning')}"
                    for item in trusted
                )

    lines.extend(["", "## Trust boundary", ""])
    lines.extend(f"- {item}" for item in report.get("trusted_axioms", []))
    guarantees = report.get("guarantees", {})
    if guarantees:
        lines.extend(["", f"This result establishes: {guarantees.get('establishes', '')}", "", "It does not prove:", ""])
        lines.extend(f"- {item}" for item in guarantees.get("does_not_prove", []))
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
