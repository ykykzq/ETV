from pathlib import Path

from etv.cli import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


def test_check_cli_rejects_internal_ir_pair(tmp_path, capsys):
    result = main(
        [
            "check",
            str(FIXTURES / "specs/add_proved.json"),
            "--out",
            str(tmp_path / "rejected"),
        ]
    )

    assert result == 2
    assert (tmp_path / "rejected/report.json").is_file()
    assert "UNKNOWN add_proved: TTIR_PAIR_REQUIRED" in capsys.readouterr().out


def test_inspect_cli_rejects_internal_ir_program(capsys):
    result = main(["inspect", str(FIXTURES / "programs/add_ntops_2d.json")])

    assert result == 2
    assert "TTIR_PAIR_REQUIRED" in capsys.readouterr().err
