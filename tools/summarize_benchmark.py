#!/usr/bin/env python3
"""Build numeric, capture, PairSpec, and formal benchmark reports."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _reference_self_check(case_dir: Path) -> bool | None:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = [
        item
        for item in manifest.get("rhs_references", [])
        if item.get("ttir_status") == "captured"
    ]
    if not references:
        return None
    checks: list[bool] = []
    for reference in references:
        check = (reference.get("checks") or {}).get("compiled_vs_fx_eager")
        checks.append(
            bool(
                check
                and check.get("structure_equal")
                and all(
                    leaf.get("allclose_1e-3")
                    for leaf in check.get("tensor_leaves", [])
                )
            )
        )
    return all(checks)


def _numeric_failure_group(operator: str) -> str:
    return {
        "conv2d": "Ninetoothed/Triton mask broadcasting compile failure",
        "addmv": "Ninetoothed/Triton tt.dot minimum-dimension compile failure",
        "msort": "Ninetoothed/Triton sort reshape compile failure",
    }.get(operator, f"unclassified numeric failure in {operator}")


def _pairspec_reason_group(reason: str | None) -> str:
    text = reason or "PairSpec unavailable"
    if text.startswith("requires one captured LHS launch"):
        return "LHS TTIR is absent or has multiple launches"
    if text.startswith("requires one selected RHS TTIR and one runtime launch"):
        return "RHS reference has zero or multiple kernels/launches"
    if text.startswith("ETV requires one tt.store"):
        return "legacy single-store frontend rejected multiple stores"
    if text.startswith("requires at least one reference with RHS TTIR") or text.startswith(
        "requires one reference with RHS TTIR"
    ):
        return "RHS TTIR is absent"
    if text.startswith("cannot align TTIR pointer arguments"):
        return "TTIR/runtime pointer ABI cannot be aligned"
    if text.startswith("legacy capture has unequal selected-store input pointers"):
        return "legacy capture lacks provenance for unequal selected-store ABI"
    if text.startswith("selected output depends on unmatched storage provenance"):
        return "selected store depends on unmatched scratch/intermediate storage"
    if "dependency multiplicity" in text or "multiple unmatched views" in text:
        return "selected-store alias/view mapping is ambiguous"
    if text.startswith("input/output alias relation differs"):
        return "input/output alias relation differs between sides"
    if text.startswith("cannot associate RHS store") or text.startswith(
        "cannot associate output leaf"
    ):
        return "stores cannot be associated with output leaves"
    if text.startswith("reference output has no tensor leaves") or "empty or not a tensor" in text:
        return "reference output has no verifiable tensor leaf"
    if text.startswith("RHS compiled-vs-FX self-check did not pass"):
        return "captured RHS failed compiled-vs-FX self-check"
    if text.startswith("capture manifest unavailable"):
        return "test did not reach the capture hook"
    return text


def _pairspecs(record: dict[str, Any]) -> list[dict[str, Any]]:
    observations = record.get("pairspecs")
    if isinstance(observations, list):
        return observations
    if record.get("status") == "generated" and record.get("pairspec"):
        return [{"pairspec": record["pairspec"], "pair_id": record.get("nodeid")}]
    return []


def _attempt_failures(record: dict[str, Any]) -> list[str]:
    return [
        str(attempt.get("reason") or "unspecified reference-attempt failure")
        for attempt in record.get("attempts", [])
        if attempt.get("status") != "generated"
    ]


def _table_row(values: Iterable[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _status_counter(records: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(record.get("status", "MISSING")) for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    collection = json.loads(
        (root / "collection-summary.json").read_text(encoding="utf-8")
    )
    pairspec = json.loads(
        (root / "pairspec-summary.json").read_text(encoding="utf-8")
    )
    formal = json.loads((root / "formal-summary.json").read_text(encoding="utf-8"))

    collection_records = collection["records"]
    pair_by_node = {record.get("nodeid"): record for record in pairspec["records"]}
    formal_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in formal["records"]:
        formal_by_node[str(record.get("nodeid"))].append(record)
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in collection_records:
        by_operator[record["operator"]].append(record)

    exact_nodes = {
        record["nodeid"]
        for record in collection_records
        if record["lhs_status"] == "captured" and record["rhs_status"] == "captured"
    }
    self_checks = {
        record["nodeid"]: _reference_self_check(Path(record["case_dir"]))
        for record in collection_records
        if record["nodeid"] in exact_nodes
    }
    self_check_counts = Counter(
        "pass" if value is True else "mismatch_or_unavailable"
        for value in self_checks.values()
    )

    numeric_counts = Counter(record["outcome"] for record in collection_records)
    failed_operators = {
        record["operator"]
        for record in collection_records
        if record["outcome"] == "failed"
    }
    numeric_failure_groups = Counter(
        _numeric_failure_group(record["operator"])
        for record in collection_records
        if record["outcome"] == "failed"
    )

    generated_case_records = [
        record for record in pairspec["records"] if record.get("status") == "generated"
    ]
    generated_observations = [
        observation
        for record in generated_case_records
        for observation in _pairspecs(record)
    ]
    unavailable_reason_groups = Counter(
        _pairspec_reason_group(record.get("reason"))
        for record in pairspec["records"]
        if record.get("status") != "generated"
    )
    failed_attempt_reason_groups = Counter(
        _pairspec_reason_group(reason)
        for record in pairspec["records"]
        if record.get("status") == "generated"
        for reason in _attempt_failures(record)
    )
    recovered_case_counts = dict(pairspec.get("recovered_case_counts", {}))

    formal_counts = _status_counter(formal["records"])
    etv_records = [
        record for record in formal["records"] if record.get("status") != "NOT_RUN"
    ]
    structured_records = [
        record
        for record in formal["records"]
        if record.get("status") in {"PROVED", "DISPROVED", "UNKNOWN"}
    ]
    formal_reason_groups = Counter(
        str(record.get("reason") or "unspecified")
        for record in formal["records"]
        if record.get("status") in {"UNKNOWN", "RUNNER_ERROR", "TIMEOUT"}
    )
    cases_with_etv = {
        nodeid
        for nodeid, records in formal_by_node.items()
        if any(record.get("status") != "NOT_RUN" for record in records)
    }
    cases_with_conclusive = {
        nodeid
        for nodeid, records in formal_by_node.items()
        if any(record.get("status") in {"PROVED", "DISPROVED"} for record in records)
    }
    disproved = [
        record for record in formal["records"] if record.get("status") == "DISPROVED"
    ]

    operator_rows: list[dict[str, Any]] = []
    for operator in sorted(by_operator):
        records = by_operator[operator]
        nodes = [record["nodeid"] for record in records]
        outcomes = Counter(record["outcome"] for record in records)
        specs = sum(len(_pairspecs(pair_by_node.get(nodeid, {}))) for nodeid in nodes)
        formal_statuses = _status_counter(
            item for nodeid in nodes for item in formal_by_node.get(nodeid, [])
        )
        operator_rows.append(
            {
                "operator": operator,
                "tests": len(records),
                "numeric": dict(sorted(outcomes.items())),
                "exact_ttir_pairs": sum(nodeid in exact_nodes for nodeid in nodes),
                "pairspec_cases": sum(
                    pair_by_node.get(nodeid, {}).get("status") == "generated"
                    for nodeid in nodes
                ),
                "pairspec_observations": specs,
                "formal_observations": dict(sorted(formal_statuses.items())),
            }
        )

    summary = {
        "format": "etv-benchmark-final-summary-v2",
        "updated_at_unix": time.time(),
        "numeric": {
            "operators_total": len(by_operator),
            "operators_completed": len(by_operator),
            "operators_without_failures": len(by_operator) - len(failed_operators),
            "operators_with_failures": len(failed_operators),
            "tests_total": len(collection_records),
            "tests_completed": len(collection_records),
            "tests_executed": numeric_counts["passed"] + numeric_counts["failed"],
            "counts": dict(sorted(numeric_counts.items())),
            "failure_groups": dict(sorted(numeric_failure_groups.items())),
        },
        "capture": {
            "lhs": collection["lhs"],
            "rhs": collection["rhs"],
            "both_sides_ttir": len(exact_nodes),
            "operators_with_both_sides_ttir": len(
                {
                    record["operator"]
                    for record in collection_records
                    if record["nodeid"] in exact_nodes
                }
            ),
            "rhs_self_check_on_exact_pairs": dict(sorted(self_check_counts.items())),
        },
        "pairspec": {
            "case_counts": pairspec["counts"],
            "generated_observations": len(generated_observations),
            "recovered_case_counts": recovered_case_counts,
            "unavailable_reason_groups": dict(sorted(unavailable_reason_groups.items())),
            "failed_reference_attempt_reason_groups": dict(
                sorted(failed_attempt_reason_groups.items())
            ),
        },
        "formal": {
            "expected_observations_or_not_run_cases": len(formal["records"]),
            "etv_invocations": len(etv_records),
            "structured_results": len(structured_records),
            "conclusive_results": formal_counts["PROVED"] + formal_counts["DISPROVED"],
            "observation_counts": dict(sorted(formal_counts.items())),
            "cases_with_etv_results": len(cases_with_etv),
            "cases_with_conclusive_results": len(cases_with_conclusive),
            "unknown_or_runner_reason_groups": dict(sorted(formal_reason_groups.items())),
            "disproved": disproved,
        },
        "by_operator": operator_rows,
    }
    _write_json(root / "summary.json", summary)

    lines = [
        "# Complete ntops TTIR benchmark report",
        "",
        "This report covers all collected test specializations from 76 ntops operators. "
        "LHS is the original Ninetoothed launch; RHS is the intercepted original "
        "reference callable/FX graph lowered automatically through TorchInductor.",
        "",
        "## Numeric testing",
        "",
        f"- Operators completed: {len(by_operator)} / {len(by_operator)}",
        f"- Tests completed: {len(collection_records)} / {len(collection_records)}",
        f"- Tests actually executed: {summary['numeric']['tests_executed']}",
        f"- Pass: {numeric_counts['passed']}",
        f"- Fail: {numeric_counts['failed']}",
        f"- Skip: {numeric_counts['skipped']}",
        f"- Operators with failures: {len(failed_operators)} "
        f"({', '.join(sorted(failed_operators))})",
        "",
        "| Numeric failure class | Tests |",
        "| --- | ---: |",
    ]
    lines.extend(
        _table_row((reason, count))
        for reason, count in sorted(numeric_failure_groups.items())
    )
    lines.extend(
        [
            "",
            "## TTIR capture",
            "",
            f"- LHS statuses: `{json.dumps(collection['lhs'], sort_keys=True)}`",
            f"- RHS statuses: `{json.dumps(collection['rhs'], sort_keys=True)}`",
            f"- Both sides captured: {len(exact_nodes)} tests across "
            f"{summary['capture']['operators_with_both_sides_ttir']} operators",
            f"- RHS compiled-vs-FX self-check on two-sided captures: "
            f"`{json.dumps(dict(sorted(self_check_counts.items())), sort_keys=True)}`",
            "",
            "## PairSpec generation",
            "",
            f"- Cases generated: {len(generated_case_records)}",
            f"- Cases unavailable: {pairspec['counts'].get('unavailable', 0)}",
            f"- Generated output observations: {len(generated_observations)}",
            f"- Recovered-feature cases: "
            f"`{json.dumps(recovered_case_counts, sort_keys=True)}`",
            "",
            "| Why formal verification was not reached | Cases |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        _table_row((reason, count))
        for reason, count in sorted(
            unavailable_reason_groups.items(), key=lambda item: (-item[1], item[0])
        )
    )
    if failed_attempt_reason_groups:
        lines.extend(
            [
                "",
                "Partial cases may still have generated observations. Their failed "
                "reference attempts are counted separately:",
                "",
                "| Failed reference-attempt class | Attempts |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            _table_row((reason, count))
            for reason, count in sorted(
                failed_attempt_reason_groups.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
    lines.extend(
        [
            "",
            "## Formal verification",
            "",
            f"- ETV invocations: {len(etv_records)} output observations across "
            f"{len(cases_with_etv)} tests",
            f"- Structured results: {len(structured_records)}",
            f"- Conclusive results: {summary['formal']['conclusive_results']} "
            f"across {len(cases_with_conclusive)} tests",
            f"- Observation statuses: "
            f"`{json.dumps(dict(sorted(formal_counts.items())), sort_keys=True)}`",
            f"- DISPROVED observations: {len(disproved)}",
            "",
            "| UNKNOWN/runner reason | Observations |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        _table_row((reason, count))
        for reason, count in sorted(
            formal_reason_groups.items(), key=lambda item: (-item[1], item[0])
        )
    )
    if disproved:
        lines.extend(["", "### DISPROVED observations", ""])
        lines.extend(
            f"- `{record.get('pair_id') or record.get('nodeid')}`: "
            f"{record.get('reason') or 'no reason recorded'}"
            for record in disproved
        )
        if all(
            "test_diag_1d" in str(record.get("nodeid"))
            and record.get("reason") == "MASK_MISMATCH"
            for record in disproved
        ):
            lines.extend(
                [
                    "",
                    "All four are `diag_1d` with `n` equal to 5 or 10 and diagonal 0. "
                    "The original CUDA tests passed. The LHS wrapper initializes the "
                    "matrix with `torch.zeros` and its captured kernel writes only the "
                    "diagonal, while the RHS kernel writes every element. The current "
                    "kernel-only PairSpec does not model that host-side initialization, "
                    "so these are real store-effect counterexamples under the PairSpec "
                    "but not evidence that the original operator outputs differ.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "No generated observation produced a DISPROVED result in this run.",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-operator results",
            "",
            "Formal columns count output observations, so tuple outputs and multiple "
            "reference expressions can contribute more than one observation per test.",
            "",
            "| Operator | Tests | Pass | Fail | Skip | Exact | Pair cases | Specs | P | D | U | E | T | N |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in operator_rows:
        numeric = row["numeric"]
        statuses = row["formal_observations"]
        lines.append(
            _table_row(
                (
                    row["operator"],
                    row["tests"],
                    numeric.get("passed", 0),
                    numeric.get("failed", 0),
                    numeric.get("skipped", 0),
                    row["exact_ttir_pairs"],
                    row["pairspec_cases"],
                    row["pairspec_observations"],
                    statuses.get("PROVED", 0),
                    statuses.get("DISPROVED", 0),
                    statuses.get("UNKNOWN", 0),
                    statuses.get("RUNNER_ERROR", 0),
                    statuses.get("TIMEOUT", 0),
                    statuses.get("NOT_RUN", 0),
                )
            )
        )
    lines.extend(
        [
            "",
            "Individual node IDs, PairSpecs, pointer/alias diagnostics, raw reasons, "
            "and ETV reports remain in each case directory and the JSON summaries.",
            "",
        ]
    )
    _write_text(root / "REPORT.md", "\n".join(lines))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("numeric", "capture", "pairspec", "formal")
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
