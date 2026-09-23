"""Stable public commands; diagnostics always use stderr."""

import argparse
import json
from pathlib import Path

from .errors import ETVError
from .frontend import parse_ttir
from .logging import configure
from .report import report_json, serialize, write_report
from .rewrites import parse_rules
from .verify import verify


def main() -> int:
    parser = argparse.ArgumentParser(prog="etv")
    commands = parser.add_subparsers(dest="command", required=True)
    verifier = commands.add_parser("verify")
    verifier.add_argument("pair", type=Path)
    verifier.add_argument("--out", type=Path, required=True)
    verifier.add_argument("--json", action="store_true")
    verifier.add_argument("--allow-external-paths", action="store_true")
    verifier.add_argument("--log-format", choices=("text", "jsonl"), default="text")
    verifier.add_argument("--log-file", type=Path)
    frontend = commands.add_parser("parse")
    frontend.add_argument("file", type=Path)
    frontend.add_argument("--function", required=True)
    frontend.add_argument("--out", type=Path, required=True)
    rules = commands.add_parser("rules").add_subparsers(dest="rules_command", required=True)
    check = rules.add_parser("check")
    check.add_argument("file", type=Path)
    check.add_argument("--json", action="store_true")
    args = parser.parse_args()
    configure(getattr(args, "log_format", "text") == "jsonl", getattr(args, "log_file", None))
    if args.command == "verify":
        report = verify(args.pair, args.allow_external_paths)
        write_report(report, args.out)
        print(
            json.dumps(report_json(report), sort_keys=True)
            if args.json
            else f"{report.status}: {report.reason}"
        )
        return {"PROVED": 0, "DISPROVED": 1, "UNKNOWN": 2}[report.status]
    try:
        if args.command == "parse":
            module = parse_ttir(args.file, args.function)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(serialize(module), indent=2, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {"status": "PARSED", "sha256": module.source_hash, "snapshot": str(args.out)}
                )
            )
        else:
            candidates = parse_rules(args.file)
            print(
                json.dumps(
                    {
                        "status": "VALID_SCHEMA",
                        "rules": [r.id for r in candidates],
                        "admission": "performed in the context of a PairSpec during verify",
                    }
                )
            )
        return 0
    except ETVError as exc:
        print(json.dumps({"status": "UNKNOWN", "reason": exc.reason, "detail": exc.detail}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
