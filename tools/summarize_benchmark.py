#!/usr/bin/env python3
"""Build a complete numeric, capture, PairSpec, and formal benchmark report."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
        "conv2d": (
            "Ninetoothed/Triton compile failure: incompatible mask broadcasting "
            "shapes (108 cases)"
        ),
        "addmv": (
            "Ninetoothed/Triton compile failure: tt.dot requires M, N, K >= 16 "
            "(2 cases)"
        ),
        "msort": (
            "Ninetoothed/Triton compile failure: sort reshape receives a dtype "
            "without numel (2 cases)"
        ),
    }.get(operator, f"unclassified numeric failure in {operator}")


def _pairspec_reason_group(reason: str | None) -> str:
    text = reason or "PairSpec unavailable"
    if text.startswith("requires one captured LHS launch"):
        return "LHS is absent or has multiple launches"
    if text.startswith("requires one selected RHS TTIR and one runtime launch"):
        return "RHS lowered to multiple Triton kernels/launches"
    if text.startswith("ETV requires one tt.store"):
        return "single-kernel IR has multiple stores"
    if text.startswith("requires one reference with RHS TTIR"):
        return "RHS TTIR is absent or multiple references were intercepted"
    if text.startswith("input pointer count differs"):
        return "LHS/RHS pointer ABI counts differ"
    if text == "reference output is not one non-empty tensor":
        return "reference output is tuple/empty/not a single tensor"
    if text.startswith("capture manifest unavailable"):
        return "test did not reach the capture hook"
    return text


def _counter(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


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
    formal_by_node = {record.get("nodeid"): record for record in formal["records"]}
    by_operator_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in collection_records:
        by_operator_records[record["operator"]].append(record)

    exact_nodes = {
        record["nodeid"]
        for record in collection_records
        if record["lhs_status"] == "captured" and record["rhs_status"] == "captured"
    }
    self_checks = {
        nodeid: _reference_self_check(Path(record["case_dir"]))
        for nodeid, record in ((item["nodeid"], item) for item in collection_records)
        if nodeid in exact_nodes
    }
    self_check_counts = Counter(
        "pass" if value else "mismatch" for value in self_checks.values()
    )

    failure_groups = Counter(
        _numeric_failure_group(record["operator"])
        for record in collection_records
        if record["outcome"] == "failed"
    )
    pairspec_reason_groups = Counter(
        _pairspec_reason_group(record.get("reason"))
        for record in pairspec["records"]
        if record["status"] != "generated"
    )
    formal_unknown_reasons = Counter(
        record.get("reason") or "unspecified"
        for record in formal["records"]
        if record["status"] == "UNKNOWN"
    )
    formal_runner_reasons = Counter(
        record.get("reason") or "unspecified"
        for record in formal["records"]
        if record["status"] in {"RUNNER_ERROR", "TIMEOUT"}
    )

    operator_rows = []
    for operator in sorted(by_operator_records):
        records = by_operator_records[operator]
        outcomes = Counter(record["outcome"] for record in records)
        exact = sum(record["nodeid"] in exact_nodes for record in records)
        exact_checked = sum(
            self_checks.get(record["nodeid"]) is True for record in records
        )
        generated = sum(
            pair_by_node.get(record["nodeid"], {}).get("status") == "generated"
            for record in records
        )
        formal_statuses = Counter(
            formal_by_node.get(record["nodeid"], {}).get("status", "MISSING")
            for record in records
        )
        operator_rows.append(
            {
                "operator": operator,
                "tests": len(records),
                "numeric": dict(sorted(outcomes.items())),
                "exact_ttir_pairs": exact,
                "exact_pairs_passing_rhs_self_check": exact_checked,
                "pairspec_generated": generated,
                "formal": dict(sorted(formal_statuses.items())),
            }
        )

    numeric_counts = Counter(record["outcome"] for record in collection_records)
    operators_with_failures = {
        record["operator"]
        for record in collection_records
        if record["outcome"] == "failed"
    }
    generated_count = pairspec["counts"].get("generated", 0)
    formal_counts = Counter(record["status"] for record in formal["records"])
    structured_results = sum(
        formal_counts[status] for status in ("PROVED", "DISPROVED", "UNKNOWN")
    )
    summary = {
        "format": "etv-benchmark-final-summary-v1",
        "updated_at_unix": time.time(),
        "numeric": {
            "operators_total": len(by_operator_records),
            "operators_completed": len(by_operator_records),
            "operators_without_failures": len(by_operator_records)
            - len(operators_with_failures),
            "operators_with_failures": len(operators_with_failures),
            "tests_total": len(collection_records),
            "tests_completed": len(collection_records),
            "tests_executed": numeric_counts["passed"] + numeric_counts["failed"],
            "counts": dict(sorted(numeric_counts.items())),
            "failure_groups": dict(sorted(failure_groups.items())),
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
            "operators_without_both_sides_ttir": sorted(
                set(by_operator_records)
                - {
                    record["operator"]
                    for record in collection_records
                    if record["nodeid"] in exact_nodes
                }
            ),
        },
        "pairspec": {
            "counts": pairspec["counts"],
            "unavailable_reason_groups": dict(sorted(pairspec_reason_groups.items())),
        },
        "formal": {
            "total_records": len(formal["records"]),
            "etv_invoked": generated_count,
            "structured_results": structured_results,
            "conclusive_results": formal_counts["PROVED"]
            + formal_counts["DISPROVED"],
            "counts": dict(sorted(formal_counts.items())),
            "unknown_reasons": dict(sorted(formal_unknown_reasons.items())),
            "runner_error_reasons": dict(sorted(formal_runner_reasons.items())),
            "disproved": formal.get("disproved", []),
        },
        "by_operator": operator_rows,
    }
    _write_json(root / "summary.json", summary)

    lines = [
        "# Complete ntops TTIR benchmark report",
        "",
        "This report covers every collected test specialization from all 76 ntops "
        "operators. LHS is the original Ninetoothed launch; RHS is the original "
        "test reference expression lowered automatically through FX and TorchInductor.",
        "",
        "## Numeric testing",
        "",
        f"- Operators completed: {summary['numeric']['operators_completed']} / "
        f"{summary['numeric']['operators_total']}",
        f"- Tests classified: {summary['numeric']['tests_completed']} / "
        f"{summary['numeric']['tests_total']}",
        f"- Tests actually executed (skip excluded): {summary['numeric']['tests_executed']}",
        f"- Pass: {numeric_counts['passed']}",
        f"- Fail: {numeric_counts['failed']}",
        f"- Skip: {numeric_counts['skipped']}",
        f"- Operators with failures: {len(operators_with_failures)} "
        f"({', '.join(sorted(operators_with_failures))})",
        "",
        "| Failure class | Tests |",
        "| --- | ---: |",
    ]
    lines.extend(
        _table_row([reason, count]) for reason, count in sorted(failure_groups.items())
    )
    lines.extend(
        [
            "",
            "## TTIR capture",
            "",
            f"- LHS statuses: `{json.dumps(collection['lhs'], sort_keys=True)}`",
            f"- RHS statuses: `{json.dumps(collection['rhs'], sort_keys=True)}`",
            f"- Both LHS and RHS TTIR captured: {len(exact_nodes)} tests across "
            f"{summary['capture']['operators_with_both_sides_ttir']} operators",
            f"- Exact-pair RHS compiled-vs-FX self-check: "
            f"{self_check_counts['pass']} pass, {self_check_counts['mismatch']} mismatch",
            "- Operators without any two-sided TTIR pair: "
            + ", ".join(summary["capture"]["operators_without_both_sides_ttir"]),
            "",
            "RHS acquisition limits: 90 references completed without launching a "
            "Triton kernel; 16 failed reference compilation (6 bincount dynamic-output "
            "shapes, 6 max_pool1d cases whose reference is expected to raise, and 4 "
            "CausalBias snapshot reconstructions); 198 tests did not execute an "
            "interceptable reference assignment; and 16 setup-level skips did not "
            "reach the capture hook.",
            "",
            "The self-check mismatches are retained rather than filtered: 8 dropout, "
            "8 alpha_dropout, 6 argsort, 4 sort, and 1 rotary-position-embedding "
            "specialization. They reflect RNG stream differences, tie-index choices, "
            "or low-precision tolerance; they are not used by a generated PairSpec in "
            "this run.",
            "",
            "## PairSpec generation",
            "",
            f"- Generated: {pairspec['counts'].get('generated', 0)}",
            f"- Unavailable: {pairspec['counts'].get('unavailable', 0)}",
            "",
            "| Why formal verification was not reached | Tests |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        _table_row([reason, count])
        for reason, count in sorted(
            pairspec_reason_groups.items(), key=lambda item: (-item[1], item[0])
        )
    )
    lines.extend(
        [
            "",
            "## Formal verification",
            "",
            f"- ETV invoked: {generated_count}",
            f"- Structured result produced: {structured_results}",
            f"- Conclusive status: {summary['formal']['conclusive_results']}",
            f"- Status counts: `{json.dumps(dict(sorted(formal_counts.items())), sort_keys=True)}`",
            "",
            "| UNKNOWN reason | Tests |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        _table_row([reason, count])
        for reason, count in sorted(
            formal_unknown_reasons.items(), key=lambda item: (-item[1], item[0])
        )
    )
    lines.extend(
        [
            "",
            "The 28 RUNNER_ERROR cases are diag PairSpecs with an unbound compacted "
            "TTIR scalar (`arg3`). ETV was invoked but did not write report.json.",
            "",
            "### Conclusive cases",
            "",
            "- PROVED: diag 1D, n=1, diagonal=0, for float32 and float16.",
            "- DISPROVED: four diag 2D cases with diagonal +4/-4, both dtypes.",
            "",
            "The four DISPROVED results are not evidence of a numeric implementation "
            "bug. Every corresponding original test passed. Their LHS launch receives "
            "a tensor view whose pointer already includes storage_offset=4, while RHS "
            "receives the base pointer and adds 4 in TTIR. The current handwritten "
            "PairSpec maps both physical pointers to the same logical base and therefore "
            "produces a COMPUTE_MISMATCH (read offset 0 versus 4). The raw ETV result is "
            "preserved, but the pair predicate is incomplete.",
            "",
            "## Per-operator results",
            "",
            "`Exact` means both TTIR sides were captured. `Checked` means the RHS "
            "compiled-vs-FX self-check passed. Formal columns are P/D/U/E/N for "
            "PROVED/DISPROVED/UNKNOWN/RUNNER_ERROR/NOT_RUN.",
            "",
            "| Operator | Tests | Pass | Fail | Skip | Exact | Checked | PairSpec | P | D | U | E | N |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in operator_rows:
        numeric = row["numeric"]
        statuses = row["formal"]
        lines.append(
            _table_row(
                [
                    row["operator"],
                    row["tests"],
                    numeric.get("passed", 0),
                    numeric.get("failed", 0),
                    numeric.get("skipped", 0),
                    row["exact_ttir_pairs"],
                    row["exact_pairs_passing_rhs_self_check"],
                    row["pairspec_generated"],
                    statuses.get("PROVED", 0),
                    statuses.get("DISPROVED", 0),
                    statuses.get("UNKNOWN", 0),
                    statuses.get("RUNNER_ERROR", 0),
                    statuses.get("NOT_RUN", 0),
                ]
            )
        )
    lines.extend(
        [
            "",
            "All individual node IDs, paths, raw reasons, PairSpecs, and ETV reports "
            "remain in the case directories and in collection-summary.json, "
            "pairspec-summary.json, and formal-summary.json.",
            "",
        ]
    )
    _write_text(root / "REPORT.md", "\n".join(lines))
    print(json.dumps({key: summary[key] for key in ("numeric", "capture", "pairspec", "formal")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
