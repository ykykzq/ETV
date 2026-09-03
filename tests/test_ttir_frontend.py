import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from etv.cli import main
from etv.ir import Expr, Sort, int_const
from etv.model import InputError, Status, UnsupportedSemantics
from etv.schema import load_pair_spec
from etv.ttir import LibTritonParser, REQUIRED_TRITON_VERSION, parse_ttir
from etv.ttir.lift import lift_ttir
from etv.ttir.model import TTIRArgument, TTIRModule, TTIROperation, TTIRValue
from etv.verify import verify_spec

ROOT = Path(__file__).resolve().parents[1]


def _has_pinned_libtriton() -> bool:
    if importlib.util.find_spec("triton") is None:
        return False
    import triton

    return triton.__version__ == REQUIRED_TRITON_VERSION


HAS_LIBTRITON = _has_pinned_libtriton()


def _cast_module(operation_name: str, source_type: str, result_type: str) -> TTIRModule:
    output = TTIRValue(1, "!tt.ptr<f32>")
    source = TTIRValue(2, source_type)
    casted = TTIRValue(3, result_type)
    zero = TTIRValue(4, "f32")
    pointer = TTIRValue(5, "!tt.ptr<f32>")

    def operation(index, name, operands=(), results=(), attributes=None, assembly=None):
        return TTIROperation(
            index=index,
            name=name,
            operands=tuple(operands),
            results=tuple(results),
            block_id=0,
            regions=0,
            attributes={} if attributes is None else attributes,
            assembly=assembly,
        )

    return TTIRModule(
        source=ROOT / "tests/fixtures/synthetic_cast.ttir",
        source_sha256="0" * 64,
        parser="test",
        parser_version=REQUIRED_TRITON_VERSION,
        function="cast_kernel",
        arguments=(TTIRArgument(0, output), TTIRArgument(1, source)),
        operations=(
            operation(0, operation_name, (source,), (casted,)),
            operation(
                1, "arith.constant", results=(zero,), attributes={"value": "0.0"}
            ),
            operation(2, "tt.addptr", (output, casted), (pointer,)),
            operation(3, "tt.store", (pointer, zero)),
            operation(4, "tt.func", attributes={"sym_name": "cast_kernel"}),
        ),
        canonical_assembly="",
    )


def _numeric_cast_module(
    operation_name: str, source_type: str, result_type: str
) -> TTIRModule:
    output = TTIRValue(1, "!tt.ptr<f32>")
    source = TTIRValue(2, source_type)
    casted = TTIRValue(3, result_type)
    return TTIRModule(
        source=ROOT / "tests/fixtures/synthetic_numeric_cast.ttir",
        source_sha256="0" * 64,
        parser="test",
        parser_version=REQUIRED_TRITON_VERSION,
        function="numeric_cast_kernel",
        arguments=(TTIRArgument(0, output), TTIRArgument(1, source)),
        operations=(
            TTIROperation(
                index=0,
                name=operation_name,
                operands=(source,),
                results=(casted,),
                block_id=0,
                regions=0,
                attributes={},
                assembly=None,
            ),
            TTIROperation(
                index=1,
                name="tt.store",
                operands=(output, casted),
                results=(),
                block_id=0,
                regions=0,
                attributes={},
                assembly=None,
            ),
            TTIROperation(
                index=2,
                name="tt.func",
                operands=(),
                results=(),
                block_id=0,
                regions=0,
                attributes={"sym_name": "numeric_cast_kernel"},
                assembly=None,
            ),
        ),
        canonical_assembly="",
    )


def _find_expr(expression: Expr, op: str):
    if expression.op == op:
        return expression
    return next(
        (
            found
            for argument in expression.args
            if (found := _find_expr(argument, op)) is not None
        ),
        None,
    )


@pytest.mark.parametrize(
    ("operation_name", "source_type", "result_type", "semantic_op"),
    [
        ("arith.extsi", "i8", "i32", "sext"),
        ("arith.extui", "i8", "i32", "zext"),
        ("arith.trunci", "i32", "i8", "trunc"),
        ("arith.index_cast", "i32", "index", "index_cast"),
        ("arith.index_castui", "i32", "index", "index_castui"),
    ],
)
def test_lifter_preserves_integer_cast_operators(
    operation_name, source_type, result_type, semantic_op
):
    program = lift_ttir(
        _cast_module(operation_name, source_type, result_type), int_const(1)
    )

    cast = _find_expr(program.stores[0].offset, semantic_op)
    assert cast is not None
    assert cast.data == (source_type, result_type)
    assert cast.sort == Sort.INT


@pytest.mark.parametrize(
    ("operation_name", "semantic_op"),
    [("arith.sitofp", "sitofp"), ("arith.uitofp", "uitofp")],
)
def test_lifter_preserves_integer_to_float_cast_operators(operation_name, semantic_op):
    program = lift_ttir(
        _numeric_cast_module(operation_name, "i32", "f32"), int_const(1)
    )

    cast = program.stores[0].value
    assert cast.op == semantic_op
    assert cast.data == ("i32", "f32")
    assert cast.sort == Sort.FLOAT


def test_lifter_selects_one_store_from_multi_output_ttir():
    module = _cast_module("arith.extsi", "i8", "i32")
    original_store = module.operations[-2]
    second_store = TTIROperation(
        index=original_store.index + 1,
        name="tt.store",
        operands=original_store.operands,
        results=(),
        block_id=0,
        regions=0,
        attributes={},
        assembly=None,
    )
    function = module.operations[-1]
    multi_store = TTIRModule(
        source=module.source,
        source_sha256=module.source_sha256,
        parser=module.parser,
        parser_version=module.parser_version,
        function=module.function,
        arguments=module.arguments,
        operations=module.operations[:-1]
        + (second_store,)
        + (
            TTIROperation(
                index=second_store.index + 1,
                name=function.name,
                operands=function.operands,
                results=function.results,
                block_id=function.block_id,
                regions=function.regions,
                attributes=function.attributes,
                assembly=function.assembly,
            ),
        ),
        canonical_assembly=module.canonical_assembly,
    )

    with pytest.raises(UnsupportedSemantics) as error:
        lift_ttir(multi_store, int_const(1))
    assert error.value.code == "MULTIPLE_STORES_UNSUPPORTED"
    assert len(lift_ttir(multi_store, int_const(1), store_index=0).stores) == 1
    assert len(lift_ttir(multi_store, int_const(1), store_index=1).stores) == 1


def test_real_add_artifacts_match_provenance():
    root = ROOT / "examples/add"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["upstreams"]["ntops"]["commit"] == (
        "9ae4166ad342e4745f0eed13a5a20d069e994fc0"
    )
    for relative, expected in provenance["artifacts"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert "/private/tmp" not in (root / "ttir/ntops_add.ttir").read_text(
        encoding="utf-8"
    )
    assert "/private/tmp" not in (root / "ttir/torch_inductor_add.ttir").read_text(
        encoding="utf-8"
    )


def test_pair_spec_accepts_explicit_ttir_frontends():
    spec = load_pair_spec(ROOT / "examples/add/pair.json")

    assert spec.lhs_frontend.kind == "ttir"
    assert spec.lhs_frontend.function == "ntops_add_kernel"
    assert spec.lhs_frontend.programs.data == 1
    assert spec.rhs_frontend.kind == "ttir"
    assert spec.rhs_frontend.function == "triton_poi_fused_0"
    assert spec.rhs_frontend.programs.data == 1


def test_parametric_add_pair_declares_symbolic_raw_ttir_launches():
    spec = load_pair_spec(ROOT / "examples/add/pair_parametric.json")

    assert spec.lhs_frontend.kind == "ttir"
    assert spec.lhs_frontend.programs.render() == "ceildiv(var(c), 256)"
    assert spec.rhs_frontend.kind == "ttir"
    assert spec.rhs_frontend.programs.render() == "ceildiv(var(c), 128)"
    assert spec.facts.parameters["a"].minimum == 1
    assert spec.facts.parameters["c"].maximum == 2**31 - 1
    assert spec.facts.constraints[0].render() == "eq(imul(var(a), var(b)), var(c))"


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_libtriton_snapshot_is_complete_and_stable():
    path = ROOT / "examples/add/ttir/torch_inductor_add.ttir"

    first = parse_ttir(path)
    second = parse_ttir(path)

    assert first.function == "triton_poi_fused_0"
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
    report = verify_spec(ROOT / "examples/add/pair.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["inputs"]["frontends"]["lhs"] == {
        "name": "libtriton",
        "version": REQUIRED_TRITON_VERSION,
    }
    assert report["inputs"]["frontends"]["rhs"] == {
        "name": "libtriton",
        "version": REQUIRED_TRITON_VERSION,
    }
    assert any(block["kind"] == "FRONTEND" for block in report["blocks"])


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_raw_ttir_mul_add_and_fma_are_proved_by_the_same_frontend():
    report = verify_spec(ROOT / "tests/fixtures/ttir/add_mul_vs_fma.json")

    assert report["status"] == Status.PROVED.value
    assert report["inputs"]["frontends"]["lhs"]["name"] == "libtriton"
    assert report["inputs"]["frontends"]["rhs"]["name"] == "libtriton"
    assert "fma_def" in report["proof"]["egraph"]["stats"]["rule_matches"]


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_extended_benchmark_dialect_ops_are_lifted_and_proved():
    root = ROOT / "tests/fixtures/ttir"
    lhs = parse_ttir(root / "extended_ops_lhs.ttir")
    rhs = parse_ttir(root / "extended_ops_rhs.ttir")

    assert {"tt.clampf", "math.exp", "math.erf"} <= {
        operation.name for operation in lhs.operations
    }
    assert {"arith.cmpf", "arith.ori", "tt.extern_elementwise"} <= {
        operation.name for operation in rhs.operations
    }

    report = verify_spec(root / "extended_ops_pair.json")
    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_raw_ttir_multilaunch_side_is_prepartitioned_and_proved(monkeypatch):
    def fake_complete(self, purpose, system, payload):
        assert purpose == "launch_partition_matching"
        assert payload["prepartitioned_side"] == "lhs"
        return {
            "partitions": [
                {
                    "id": "multiply",
                    "family": "family_0",
                    "anchor": "root.dep[0]",
                    "semantic": "scale Other by Alpha",
                    "counterpart_path": "root.args[1]",
                },
                {
                    "id": "output",
                    "family": "family_0",
                    "anchor": "root",
                    "semantic": "add Input and store Output",
                    "counterpart_path": "root",
                },
            ]
        }, {"purpose": purpose, "provider": "deepseek", "model": "fake"}

    monkeypatch.setattr("etv.partition.DeepSeekClient.complete_json", fake_complete)
    report = verify_spec(ROOT / "tests/fixtures/ttir/multilaunch_pair.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["proof"]["partitioning"]["mode"] == ("prepartitioned_launch_sequence")
    assert [item["launch_id"] for item in report["inputs"]["frontends"]["lhs"]] == [
        "multiply",
        "add",
    ]


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_parametric_raw_ttir_pair_is_lifted_and_proved():
    report = verify_spec(ROOT / "examples/add/pair_parametric.json")

    assert report["status"] == Status.PROVED.value
    assert report["reason"] == "OBSERVABLE_MEMORY_EQUIVALENT"
    assert report["scope"] == (
        "fixed-rank symbolic-shape single-store parametric translation validation"
    )
    assert report["inputs"]["frontends"]["lhs"] == {
        "name": "libtriton",
        "version": REQUIRED_TRITON_VERSION,
    }
    assert report["inputs"]["frontends"]["rhs"] == {
        "name": "libtriton",
        "version": REQUIRED_TRITON_VERSION,
    }
    assert report["proof"]["parametric_domain"]["complete_for_parameter_domain"] is True
    egraph = report["proof"]["egraph"]
    assert egraph["root_pairs"] == 1
    assert egraph["after_predicate_rewrites"]["unmatched_root_pairs"] == 0
    assert "PARAMETRIC_SMT" in next(
        block["proof_levels"]
        for block in report["blocks"]
        if block["kind"] == "ADDRESS"
    )


@pytest.mark.skipif(not HAS_LIBTRITON, reason="requires Triton/libtriton 3.7.1")
def test_parse_cli_writes_snapshot(tmp_path):
    output = tmp_path / "snapshot.json"

    result = main(
        [
            "parse",
            str(ROOT / "examples/add/ttir/ntops_add.ttir"),
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
