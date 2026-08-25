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
        f"- Soundness: `{report.get('soundness', {}).get('level', 'not_established')}`",
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

    llm_context = report.get("llm_context", {})
    lines.extend(["", "## LLM-only context", ""])
    llm_assumptions = llm_context.get("assumptions", [])
    if llm_assumptions:
        lines.extend(f"- {item}" for item in llm_assumptions)
    else:
        lines.append("- None loaded.")
    lines.append(
        f"- Proof relevance: `{llm_context.get('proof_relevance', 'informational_only')}`."
    )

    predicates = report.get("formal_predicates", [])
    lines.extend(["", "## Declared formal predicates", ""])
    if predicates:
        lines.extend(
            f"- `{item.get('id')}`: {item.get('kind')} / {item.get('status')} "
            f"(`{item.get('evidence')}`)"
            for item in predicates
        )
    else:
        lines.append("- None loaded.")

    registry = report.get("rewrite_registry", {})
    lines.extend(["", "## Rewrite registry", ""])
    lines.append(f"- Builtin: `{registry.get('builtin', 'none')}`")
    for source in registry.get("user_sources", []):
        lines.append(
            f"- User file: `{source.get('path')}`; SHA-256 `{source.get('sha256')}`"
        )

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
        lines.append(
            f"| `{block.get('kind')}` | **{block.get('status')}** | {evidence} | {summary} |"
        )

    counterexample = report.get("counterexample")
    if counterexample:
        lines.extend(
            [
                "",
                "## Counterexample",
                "",
                "```json",
                json.dumps(
                    counterexample, indent=2, sort_keys=True, ensure_ascii=False
                ),
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
        parametric = proof.get("parametric_domain")
        if parametric:
            parameter_names = ", ".join(sorted(parametric.get("parameters", {})))
            checks = parametric.get("checks", [])
            proved_checks = sum(item.get("result") == "unsat" for item in checks)
            lines.append(
                f"Parametric domain: `{parameter_names}`; {proved_checks} SMT obligations proved UNSAT, "
                f"with one satisfiable domain check."
            )
        partitioning = proof.get("partitioning")
        if partitioning:
            lines.extend(["", "### Subgraph Partitioning", ""])
            lines.append(
                f"Provider/model: `{partitioning.get('provider')}/"
                f"{partitioning.get('configured_model')}`; status: "
                f"`{partitioning.get('status')}`; whole-program calls: "
                f"{partitioning.get('whole_program_calls', 0)}."
            )
            partitions = partitioning.get("partitions", [])
            if partitions:
                lines.extend(
                    [
                        "",
                        "| Subgraph | Family | Semantic | Dependencies | Instances |",
                        "| --- | --- | --- | --- | --- |",
                    ]
                )
                for item in partitions:
                    dependencies = ", ".join(item.get("dependencies", [])) or "-"
                    semantic = str(item.get("semantic", "")).replace("|", "\\|")
                    lines.append(
                        f"| `{item.get('id')}` | `{item.get('family')}` | {semantic} | "
                        f"{dependencies} | {item.get('instances')} |"
                    )
            if partitioning.get("fallback"):
                lines.append(
                    f"Fallback: `{partitioning.get('fallback')}`; "
                    f"error: {partitioning.get('error', partitioning.get('subgraph_reason', 'unknown'))}."
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
            initial = egraph.get("initial_state", {})
            if initial:
                lines.append(
                    f"Initial unmatched roots: {initial.get('unmatched_root_pairs')}; "
                    f"rewrite phase: `{stats.get('phase', 'UNSPECIFIED')}`; "
                    f"unmatched after predicate-derived rewrites: "
                    f"{egraph.get('after_predicate_rewrites', {}).get('unmatched_root_pairs')}."
                )
            rules = stats.get("rule_matches", {})
            lines.append(
                "Matched rules: "
                + (
                    ", ".join(f"`{key}` x{value}" for key, value in rules.items())
                    or "none"
                )
            )
            relational = [
                item
                for item in egraph.get("rule_application", [])
                if item.get("used") and str(item.get("id", "")).startswith("parametric_")
            ]
            if relational:
                lines.extend(["", "### Applied Relational Rewrites", ""])
                lines.extend(
                    f"- `{item.get('id')}` matched {item.get('matches')} time(s); "
                    f"admission `{item.get('admission_status')}`."
                    for item in relational
                )
            trace = stats.get("iteration_trace", [])
            if trace:
                lines.extend(
                    [
                        "",
                        "### E-graph Iterations",
                        "",
                        "| Phase/iteration | Before | After | Updated | Matched rules |",
                        "| --- | --- | --- | --- | --- |",
                    ]
                )
                for item in trace:
                    before = item.get("before", {})
                    after = item.get("after", {})
                    matched = ", ".join(
                        f"{name} x{count}"
                        for name, count in item.get("rule_matches", {}).items()
                    ) or "none"
                    lines.append(
                        f"| {item.get('phase', stats.get('phase', 'UNSPECIFIED'))} "
                        f"#{item.get('iteration')} | "
                        f"{before.get('enodes')}/{before.get('eclasses')} | "
                        f"{after.get('enodes')}/{after.get('eclasses')} | "
                        f"{item.get('updated')} | {matched} |"
                    )
            unverified = egraph.get(
                "unverified_rule_uses", egraph.get("trusted_rule_uses", [])
            )
            if unverified:
                lines.extend(["", "### Applied Unverified Rewrites", ""])
                lines.extend(
                    f"- `{item.get('id')}` matched {item.get('matches')} time(s): "
                    f"{item.get('validation', {}).get('warning')}"
                    for item in unverified
                )
        llm = proof.get("llm_assistance")
        if llm:
            lines.extend(["", "### LLM Assistance", ""])
            lines.append(
                f"Provider/model: `{llm.get('provider')}/{llm.get('configured_model', llm.get('model'))}`; "
                f"status: `{llm.get('status', 'completed')}`."
            )
            generated = llm.get("generated_rule_ids", [])
            lines.append(
                "Generated candidates: "
                + (", ".join(f"`{item}`" for item in generated) or "none")
            )

    lines.extend(["", "## Trust boundary", ""])
    soundness = report.get("soundness", {})
    conditions = soundness.get("conditional_on", [])
    lines.append(f"- Soundness level: `{soundness.get('level', 'not_established')}`")
    if conditions:
        lines.extend(f"- Conditional on: {item}" for item in conditions)
    lines.extend(f"- {item}" for item in report.get("trusted_axioms", []))
    guarantees = report.get("guarantees", {})
    if guarantees:
        lines.extend(
            [
                "",
                f"This result establishes: {guarantees.get('establishes', '')}",
                "",
                "It does not prove:",
                "",
            ]
        )
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
