#!/usr/bin/env python3
"""Extract auditable specification evidence from the pinned ntops tests.

The output deliberately records test syntax rather than claiming that finite
pytest cases are universal operator contracts.  The human-reviewed test-domain
catalog lives in ``docs/ntops_test_specs.md``.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "ntops-test-evidence-v1"
EXCLUDED_WRAPPERS = {"matmul": "dispatch wrapper over mm/bmm; no independent kernel"}
TENSOR_FACTORIES = {
    "torch.arange",
    "torch.complex",
    "torch.empty",
    "torch.empty_like",
    "torch.full",
    "torch.full_like",
    "torch.randint",
    "torch.ones",
    "torch.rand",
    "torch.randn",
    "torch.tensor",
    "torch.zeros",
    "torch.zeros_like",
}


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _calls(node: ast.AST) -> Iterable[tuple[str, ast.Call]]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _dotted_name(child.func)
            if name is not None:
                yield name, child


def _source(node: ast.AST, source_text: str) -> str:
    segment = ast.get_source_segment(source_text, node)
    if segment is None:
        raise ValueError(f"source segment unavailable for {type(node).__name__}")
    return segment


def _test_function(
    node: ast.FunctionDef, candidate_aliases: set[str], source_text: str
) -> dict[str, Any]:
    calls = list(_calls(node))
    assertions = sorted(
        (child for child in ast.walk(node) if isinstance(child, ast.Assert)),
        key=lambda child: (child.lineno, child.col_offset),
    )
    return {
        "name": node.name,
        "line": node.lineno,
        "arguments": [argument.arg for argument in node.args.args],
        "decorators": [
            _source(decorator, source_text) for decorator in node.decorator_list
        ],
        "parametrization": [
            _source(decorator, source_text)
            for decorator in node.decorator_list
            if "parametrize" in _source(decorator, source_text)
        ],
        "candidate_calls": sorted(
            {
                _source(call, source_text)
                for name, call in calls
                if name.startswith("ntops.torch.") or name in candidate_aliases
            }
        ),
        "torch_calls": sorted(
            {
                _source(call, source_text)
                for name, call in calls
                if name.startswith("torch.") or name.startswith("F.")
            }
        ),
        "tensor_factories": sorted(
            {
                _source(call, source_text)
                for name, call in calls
                if name in TENSOR_FACTORIES
            }
        ),
        "assertions": [
            _source(child.test, source_text)
            for child in assertions
        ],
        "body": "\n".join(
            _source(statement, source_text) for statement in node.body
        ),
    }


def _module(path: Path, tests_root: Path) -> dict[str, Any]:
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    imports = [
        _source(node, source_text)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    candidate_aliases = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("ntops.torch")
        for alias in node.names
    }
    helpers = [
        {
            "name": node.name,
            "line": node.lineno,
            "body": _source(node, source_text),
        }
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("test_")
    ]
    tests = [
        _test_function(node, candidate_aliases, source_text)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    return {
        "operator": path.stem.removeprefix("test_"),
        "source": path.relative_to(tests_root.parent).as_posix(),
        "imports": imports,
        "constants": [
            _source(node, source_text)
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ],
        "helpers": helpers,
        "tests": tests,
    }


def _git_value(repo: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _license_metadata(repo: Path) -> dict[str, str | None]:
    license_path = repo / "LICENSE"
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except OSError:
        return {"spdx": None, "file": None}

    spdx = None
    if "Apache License" in license_text and "Version 2.0" in license_text:
        spdx = "Apache-2.0"
    return {"spdx": spdx, "file": "LICENSE"}


def extract(ntops_repo: Path, include_wrappers: bool = False) -> dict[str, Any]:
    tests_root = ntops_repo / "tests"
    if not tests_root.is_dir():
        raise ValueError(f"ntops tests directory does not exist: {tests_root}")

    modules = []
    excluded = []
    for path in sorted(tests_root.glob("test_*.py")):
        operator = path.stem.removeprefix("test_")
        reason = EXCLUDED_WRAPPERS.get(operator)
        if reason is not None and not include_wrappers:
            excluded.append(
                {
                    "operator": operator,
                    "source": path.relative_to(ntops_repo).as_posix(),
                    "reason": reason,
                }
            )
            continue
        modules.append(_module(path, tests_root))

    shared_helpers = _module_helpers(tests_root / "utils.py", tests_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": _git_value(ntops_repo, "remote", "get-url", "origin"),
            "commit": _git_value(ntops_repo, "rev-parse", "HEAD"),
            "tests_root": "tests",
            "license": _license_metadata(ntops_repo),
        },
        "interpretation": {
            "kind": "finite_test_evidence",
            "warning": (
                "Entries reproduce pytest input generation and checks; they are not "
                "universal formal contracts. See docs/ntops_test_specs.md for the "
                "human-reviewed test-domain catalog."
            ),
        },
        "scope": {
            "operator_count": len(modules),
            "excluded": excluded,
        },
        "shared_helpers": shared_helpers,
        "operators": modules,
    }


def _module_helpers(path: Path, tests_root: Path) -> dict[str, Any]:
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    return {
        "source": path.relative_to(tests_root.parent).as_posix(),
        "constants": [
            _source(node, source_text)
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ],
        "functions": [
            {
                "name": node.name,
                "line": node.lineno,
                "body": _source(node, source_text),
            }
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ntops-repo",
        type=Path,
        default=Path("build/upstream/ntops"),
        help="path to the ntops checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of stdout",
    )
    parser.add_argument(
        "--include-wrappers",
        action="store_true",
        help="include wrapper-only API tests such as matmul",
    )
    arguments = parser.parse_args()

    result = extract(arguments.ntops_repo.resolve(), arguments.include_wrappers)
    rendered = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
