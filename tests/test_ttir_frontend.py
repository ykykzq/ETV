import importlib.util
import json
from pathlib import Path

import pytest

from etv.cli import main
from etv.model import InputError, Status, UnsupportedSemantics, int_const
from etv.schema import load_pair_spec
from etv.ttir import LibTritonParser, REQUIRED_TRITON_VERSION, parse_ttir
from etv.ttir.lift import lift_ttir
from etv.verify import verify_spec


ROOT = Path(__file__).resolve().parents[1]


def _has_pinned_libtriton() -> bool:
    if importlib.util.find_spec("triton") is None:
        return False
    import triton

    return triton.__version__ == REQUIRED_TRITON_VERSION


HAS_LIBTRITON = _has_pinned_libtriton()


def test_pair_spec_accepts_explicit_ttir_frontends():
    spec = load_pair_spec(ROOT / "examples/specs/add_raw_ttir_proved.json")

    assert spec.lhs_frontend.kind == "ttir"
    assert spec.lhs_frontend.function == "add_mul"
    assert spec.lhs_frontend.programs.data == 1


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_libtriton_snapshot_is_complete_and_stable():
    path = ROOT / "examples/ttir/add_mul.ttir"

    first = parse_ttir(path)
    second = parse_ttir(path)

    assert first.function == "add_mul"
    assert first.parser_version == REQUIRED_TRITON_VERSION
    assert len(first.arguments) == 5
    assert {"tt.get_program_id", "tt.load", "tt.store", "arith.mulf"} <= {
        operation.name for operation in first.operations
    }
    assert first.to_json() == second.to_json()
    with pytest.raises(InputError) as error:
        LibTritonParser(required_version="0.0.0").parse(path)
    assert error.value.code == "TTIR_VERSION_MISMATCH"


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_libtriton_parses_region_operations_outside_lifting_subset(tmp_path):
    path = ROOT / "tests/fixtures/loop.ttir"
    module = parse_ttir(path)

    names = {operation.name for operation in module.operations}
    assert {"scf.for", "scf.yield", "tt.store"} <= names
    with pytest.raises(UnsupportedSemantics) as error:
        lift_ttir(module, int_const(1))
    assert error.value.code == "TTIR_REGION_SEMANTICS_UNSUPPORTED"

    spec = {
        "format": "etv-pair-v1",
        "pair_id": "loop_outside_lifting_subset",
        "lhs": str(path),
        "rhs": str(path),
        "frontends": {
            "lhs": {"kind": "ttir", "function": "loop", "programs": 1},
            "rhs": {"kind": "ttir", "function": "loop", "programs": 1},
        },
        "semantic_mode": "abstract_float",
        "roles": {
            "Output": {
                "lhs": {"kind": "block", "name": "arg0"},
                "rhs": {"kind": "block", "name": "arg0"},
            }
        },
        "facts": {"bindings": {"arg1": 4}, "assumptions": [], "disjoint": []},
        "contract": {
            "output_role": "Output",
            "output_numel": 1,
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "limits": {},
    }
    spec_path = tmp_path / "loop-pair.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    report = verify_spec(spec_path)
    assert report["status"] == Status.UNKNOWN.value
    assert report["reason"] == "TTIR_REGION_SEMANTICS_UNSUPPORTED"


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_raw_ttir_pair_is_lifted_and_proved():
    report = verify_spec(ROOT / "examples/specs/add_raw_ttir_proved.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["inputs"]["frontends"]["lhs"] == {
        "name": "libtriton",
        "version": REQUIRED_TRITON_VERSION,
    }
    assert any(block["kind"] == "FRONTEND" for block in report["blocks"])


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_parse_cli_writes_snapshot(tmp_path):
    output = tmp_path / "snapshot.json"

    result = main(
        [
            "parse",
            str(ROOT / "examples/ttir/add_mul.ttir"),
            "--out",
            str(output),
            "--no-assembly",
        ]
    )

    assert result == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["format"] == "etv-ttir-snapshot-v1"
    assert "canonical_assembly" not in value
    assert all("assembly" not in operation for operation in value["operations"])


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_invalid_ttir_is_rejected_by_libtriton(tmp_path):
    path = tmp_path / "invalid.ttir"
    path.write_text("module { tt.func public @broken( { }", encoding="utf-8")

    with pytest.raises(InputError) as error:
        parse_ttir(path)

    assert error.value.code == "TTIR_PARSE_ERROR"
