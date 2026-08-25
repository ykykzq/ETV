import json
import logging
from pathlib import Path

from etv.cli import main
from etv.observability import configure_logging, get_logger, log_context, log_event


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/semantic"


def _remove_etv_handlers() -> None:
    logger = logging.getLogger("etv")
    for handler in list(logger.handlers):
        if getattr(handler, "_etv_handler", False):
            logger.removeHandler(handler)
            handler.close()


def test_structured_logging_has_context_and_replaces_handlers(tmp_path):
    path = tmp_path / "events.jsonl"
    try:
        run_id = configure_logging("INFO", log_file=path, log_format="json", run_id="run-a")
        with log_context(pair_id="pair-a", phase="unit"):
            log_event(
                get_logger("test_logging"),
                logging.INFO,
                "unit_event",
                "one event",
                count=1,
            )
        configure_logging("INFO", log_file=path, log_format="json", run_id="run-b")
        log_event(
            get_logger("test_logging"),
            logging.INFO,
            "second_event",
            "another event",
        )

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert run_id == "run-a"
        assert [record["event"] for record in records] == [
            "unit_event",
            "second_event",
        ]
        assert records[0]["run_id"] == "run-a"
        assert records[0]["pair_id"] == "pair-a"
        assert records[0]["phase"] == "unit"
        assert records[0]["fields"] == {"count": 1}
        assert records[1]["run_id"] == "run-b"
    finally:
        _remove_etv_handlers()


def test_check_cli_keeps_json_report_on_stdout_and_writes_run_log(tmp_path, capsys):
    output = tmp_path / "artifacts"
    result = main(
        [
            "check",
            str(FIXTURES / "specs/add_proved.json"),
            "--out",
            str(output),
            "--json",
            "--log-format",
            "json",
        ]
    )
    try:
        captured = capsys.readouterr()
        assert result == 2
        report = json.loads(captured.out)
        assert report["reason"] == "TTIR_PAIR_REQUIRED"
        assert "verification_started" not in captured.err
        log_path = output / "etv.log"
        assert log_path.is_file()
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert records
        assert len({record["run_id"] for record in records}) == 1
        assert any(record["event"] == "spec_rejected" for record in records)
    finally:
        _remove_etv_handlers()
