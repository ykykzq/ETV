"""Host semantics are never selected by a missing CLI option."""

from pathlib import Path
import sys

import pytest

from etv_bench import cli


@pytest.mark.parametrize(
    "option,expected",
    [
        ([], "ttir-only"),
        (["--execution-scope", "ttir-only"], "ttir-only"),
        (["--execution-scope", "ttir-with-host-ops"], "ttir-with-host-ops"),
    ],
)
def test_pair_cli_passes_explicit_execution_scope(monkeypatch, option, expected):
    called = []

    def pair(path, command, *, execution_scope):
        called.append((path, command, execution_scope))
        return Path("manifest.json")

    monkeypatch.setattr(cli, "pair_capture", pair)
    monkeypatch.setattr(sys, "argv", ["etv-bench", "pair", "capture.json", *option])
    assert cli.main() == 0
    assert called == [(Path("capture.json"), ["etv"], expected)]
