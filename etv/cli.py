"""Command-line interface for ETV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from .model import Status
from .observability import configure_logging, get_logger, log_context, log_event
from .reporting import render_markdown, write_report
from .schema import InputError
from .ttir import parse_ttir
from .verify import verify_spec
from .z3_validator import validated_builtin_rules


LOGGER = get_logger(__name__)


def _check(args: argparse.Namespace) -> int:
    report = verify_spec(Path(args.spec))
    if args.out:
        write_report(report, Path(args.out))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"{report['status']} {report['pair_id']}: {report['reason']}")
        if args.out:
            print(f"artifacts: {Path(args.out).resolve()}")
    return {
        Status.PROVED.value: 0,
        Status.DISPROVED.value: 1,
        Status.UNKNOWN.value: 2,
    }[report["status"]]


def _inspect(args: argparse.Namespace) -> int:
    path = Path(args.program)
    try:
        if path.suffix not in {".ttir", ".mlir"}:
            raise InputError(
                "inspect accepts only raw .ttir or .mlir files",
                "TTIR_PAIR_REQUIRED",
            )
        module = parse_ttir(path, function=args.function)
    except InputError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "cli_input_rejected",
            "inspect command rejected the input",
            code=getattr(exc, "code", "INVALID_INPUT"),
            error=str(exc),
        )
        print(f"UNKNOWN {getattr(exc, 'code', 'INVALID_INPUT')}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(module.to_json(include_assembly=False), indent=2, sort_keys=True))
    return 0


def _parse(args: argparse.Namespace) -> int:
    try:
        module = parse_ttir(Path(args.input), function=args.function)
    except InputError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "cli_input_rejected",
            "parse command rejected the input",
            code=getattr(exc, "code", "INVALID_INPUT"),
            error=str(exc),
        )
        print(f"UNKNOWN {getattr(exc, 'code', 'INVALID_INPUT')}: {exc}", file=sys.stderr)
        return 2
    value = module.to_json(include_assembly=not args.no_assembly)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(f"parsed {module.function}: {len(module.operations)} operations -> {output.resolve()}")
    else:
        print(encoded, end="")
    return 0


def _rules(args: argparse.Namespace) -> int:
    _, values = validated_builtin_rules()
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for value in values:
            result = value.get("validation", {}).get("result", "not-run")
            print(f"{value['id']}: {value['statement']} [{value['evidence']}, z3={result}]")
    return 0


def _validate_rule(args: argparse.Namespace) -> int:
    _, values = validated_builtin_rules()
    selected = [value for value in values if args.rule_id is None or value["id"] == args.rule_id]
    if not selected:
        print(f"unknown rule: {args.rule_id}", file=sys.stderr)
        return 2
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0 if all(value["status"] == "proved" for value in selected) else 1


def _explain(args: argparse.Namespace) -> int:
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "report_read_failed",
            "explain command could not read the report",
            error=str(exc),
        )
        print(f"cannot read report: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(report), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etv",
        description="Symbolic equivalence verification for pairs of raw TTIR programs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_logging_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            default=None,
            help="runtime log level (default: WARNING)",
        )
        command.add_argument(
            "--log-file",
            help="write runtime logs to this UTF-8 file (check defaults to <out>/etv.log)",
        )
        command.add_argument(
            "--log-format",
            choices=("text", "json"),
            default="text",
            help="runtime log format",
        )

    check = subparsers.add_parser("check", help="verify a pair specification")
    add_logging_arguments(check)
    check.add_argument("spec", help="path to an etv-pair-v2 JSON file")
    check.add_argument("--out", help="directory for report.json and report.md")
    check.add_argument("--json", action="store_true", help="print the full machine report")
    check.set_defaults(handler=_check)

    inspect = subparsers.add_parser(
        "inspect", help="inspect a raw TTIR program"
    )
    add_logging_arguments(inspect)
    inspect.add_argument("program")
    inspect.add_argument("--function", help="TTIR function to inspect when the module is ambiguous")
    inspect.set_defaults(handler=_inspect)

    parse = subparsers.add_parser("parse", help="parse and verify raw TTIR with libtriton")
    add_logging_arguments(parse)
    parse.add_argument("input", help="path to a .ttir or .mlir file")
    parse.add_argument("--function", help="entry function when the module has multiple functions")
    parse.add_argument("--out", help="write the etv-ttir-snapshot-v1 JSON file")
    parse.add_argument("--no-assembly", action="store_true", help="omit canonical assembly from JSON")
    parse.set_defaults(handler=_parse)

    rules = subparsers.add_parser("rules", help="list accepted equality rules")
    add_logging_arguments(rules)
    rules.add_argument("--json", action="store_true")
    rules.set_defaults(handler=_rules)

    validate_rule = subparsers.add_parser("validate-rule", help="replay Z3 admission checks")
    add_logging_arguments(validate_rule)
    validate_rule.add_argument("rule_id", nargs="?")
    validate_rule.set_defaults(handler=_validate_rule)

    explain = subparsers.add_parser("explain", help="render a machine report as Markdown")
    add_logging_arguments(explain)
    explain.add_argument("report")
    explain.set_defaults(handler=_explain)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configured_file = args.log_file
    auto_log_file = configured_file is None and args.command == "check" and args.out
    if auto_log_file:
        configured_file = str(Path(args.out) / "etv.log")
    run_id = configure_logging(
        args.log_level or "WARNING",
        log_file=Path(configured_file) if configured_file else None,
        log_format=args.log_format,
        file_level=(
            logging.INFO
            if auto_log_file and args.log_level is None
            else None
        ),
    )
    with log_context(run_id=run_id, command=args.command):
        log_event(
            LOGGER,
            logging.INFO,
            "cli_command_started",
            "CLI command started",
            command=args.command,
            log_file=configured_file,
        )
        result = int(args.handler(args))
        log_event(
            LOGGER,
            logging.INFO,
            "cli_command_finished",
            "CLI command finished",
            command=args.command,
            exit_code=result,
        )
        return result


if __name__ == "__main__":
    raise SystemExit(main())
