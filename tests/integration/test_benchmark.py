import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from etv_bench.pair import pair_capture
from etv_bench.review import review_case
from etv_bench.run import run_case
from etv_bench.summarize import summarize

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = [sys.executable, "-m", "etv"]


def capture_fixture(tmp_path):
    template = (ROOT / "examples/compute_mismatch/lhs.ttir").read_text()
    template = template.replace("%arg1: !tt.ptr<f32>)", "%arg1: !tt.ptr<f32>, %arg2: i32)")
    data = {
        "format": "etv-runtime-capture-v1",
        "source": {"repository": "fixture", "commit": "fixture", "test_id": "runtime_one"},
        "environment": {},
        "launches": [],
        "input_roles": {"input": "Input"},
        "outputs": {"lhs": {"out-lhs": "Output"}, "rhs": {"out-rhs": "Output"}},
        "output_numel": 4,
    }
    for side in ("lhs", "rhs"):
        path = tmp_path / (side + ".ttir")
        path.write_text(template)
        slots = [
            {
                "slot": index,
                "source_name": name,
                "compiler_type": "*fp32",
                "runtime": {
                    "kind": "buffer",
                    "storage_id": identity,
                    "offset": 0,
                    "dtype": "torch.float32",
                },
            }
            for index, name, identity in ((0, "x", "input"), (1, "out", "out-" + side))
        ]
        slots.append(
            {
                "slot": 2,
                "source_name": "n",
                "compiler_type": "i32",
                "runtime": {"kind": "integer", "value": 1},
            }
        )
        data["launches"].append(
            {
                "file": path.name,
                "side": side,
                "ordinal": 0,
                "step": 0,
                "function": "lhs",
                "grid": [1, 1, 1],
                "slots": slots,
                "selected_runtime_launch": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(data))
    return path, data


def test_pair_run_review_and_summarize(tmp_path):
    capture, original = capture_fixture(tmp_path)
    manifest = pair_capture(capture, VERIFIER)
    assert json.loads(manifest.read_text())["status"] == "ready"
    pair = json.loads((tmp_path / "pair.json").read_text())
    assert pair["programs"]["lhs"][0]["bindings"] == {"arg2": 1}
    result = run_case(manifest, tmp_path / "result", VERIFIER)
    assert result["status"] == "PROVED"
    review_case(tmp_path, "checked runtime slot and output storage", VERIFIER)
    assert json.loads(capture.read_text()) == original
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["review"][-1]["status"] == "reviewed"
    summary = summarize([tmp_path / "result/report.json"])
    assert summary["statuses"]["PROVED"] == 1


def test_ambiguous_slot_mapping_needs_review(tmp_path):
    path, data = capture_fixture(tmp_path)
    data["launches"][0]["slots"].pop()
    path.write_text(json.dumps(data))
    manifest = pair_capture(path, VERIFIER)
    assert json.loads(manifest.read_text())["status"] == "needs_review"


def test_runner_checks_hashes_before_launch(tmp_path):
    shutil.copytree(ROOT / "examples/multilaunch", tmp_path / "case")
    (tmp_path / "case/lhs.ttir").write_text("module {}")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_case(tmp_path / "case/manifest.json", tmp_path / "result", ["must-not-execute"])


def test_runner_rejects_stale_report(tmp_path):
    path = tmp_path / "result"
    path.mkdir()
    (path / "report.json").write_text('{"status": "PROVED"}')
    with pytest.raises(ValueError, match="fresh report"):
        run_case(ROOT / "examples/multilaunch/manifest.json", path, [sys.executable, "-c", "pass"])


def test_unknown_is_counted_without_reclassification(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"pair_id": "unknown", "status": "UNKNOWN", "reason": "TTIR_REGION_UNSUPPORTED"})
    )
    result = summarize([path])
    assert result["statuses"]["UNKNOWN"] == 1
