from pathlib import Path

from etv.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_cli_exit_codes_and_artifacts(tmp_path, capsys):
    proved = main(
        [
            "check",
            str(ROOT / "examples/specs/add_proved.json"),
            "--out",
            str(tmp_path / "proof"),
        ]
    )
    disproved = main(["check", str(ROOT / "examples/specs/add_bad_mask.json")])
    unknown = main(["check", str(ROOT / "examples/specs/add_missing_alias.json")])

    assert proved == 0
    assert disproved == 1
    assert unknown == 2
    assert (tmp_path / "proof/report.json").is_file()
    output = capsys.readouterr().out
    assert "PROVED add_ntops_2d_vs_inductor_linear" in output
    assert "DISPROVED add_bad_mask" in output
    assert "UNKNOWN add_missing_alias_fact" in output
