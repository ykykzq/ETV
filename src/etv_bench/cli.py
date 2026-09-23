"""Independent artifact-generation and benchmark commands."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .capture import collect
from .pair import pair_capture
from .run import run_case
from .summarize import summarize
from .review import review_case


def main() -> int:
    parser = argparse.ArgumentParser(prog="etv-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("entry", help="Python module:function accepting a Capture object")
    capture.add_argument("--out", required=True, type=Path)
    capture.add_argument("--repository", required=True)
    capture.add_argument("--commit", required=True)
    capture.add_argument("--test-id", required=True)
    pair = commands.add_parser("pair")
    pair.add_argument("capture", type=Path)
    pair.add_argument("--verifier", default="etv")
    run = commands.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--verifier", default="etv")
    summary = commands.add_parser("summarize")
    summary.add_argument("reports", nargs="+", type=Path)
    review = commands.add_parser("review")
    review.add_argument("case", type=Path)
    review.add_argument("--note", required=True)
    review.add_argument("--verifier", default="etv")
    args = parser.parse_args()
    try:
        if args.command == "collect":
            result = {
                "capture": str(
                    collect(
                        args.entry,
                        args.out,
                        {
                            "repository": args.repository,
                            "commit": args.commit,
                            "test_id": args.test_id,
                        },
                    )
                )
            }
        elif args.command == "pair":
            result = {"manifest": str(pair_capture(args.capture, shlex.split(args.verifier)))}
        elif args.command == "run":
            result = run_case(args.manifest, args.out, shlex.split(args.verifier))
        elif args.command == "summarize":
            result = summarize(args.reports)
        else:
            result = review_case(args.case, args.note, shlex.split(args.verifier))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
