"""Explicit output encodings preserve types and the entire writer domain."""

import importlib

import pytest
import z3

from etv.errors import InputError
from etv.ir import BOOL, FLOAT, Expr, Kernel, ProgramPair, ProgramSequence, StorageRef, Store
from etv.ir import const, var
from etv.memory import build_obligations
from etv.pairspec import parse_pair_spec
from etv.report import report_json
from etv.smt import Encoder
from test_contracts import minimal, write


def encoded_data(tmp_path, reverse=False, numel=1):
    data = minimal(tmp_path)
    data["predicates"]["roles"] = {
        "Output_lhs": {"kind": "buffer", "element": "bool" if reverse else "abstract_float"},
        "Output_rhs": {"kind": "buffer", "element": "abstract_float" if reverse else "bool"},
    }
    representation = {"kind": "boolean_numeric_01"}
    for side in ("lhs", "rhs"):
        role = "Output_" + side
        boolean_side = (side == "lhs") == reverse
        representation[side] = {
            "role": role,
            "encoding": "bool_to_numeric_01" if boolean_side else "identity",
        }
        launch = data["programs"][side][0]
        launch["abi"] = {role: {"kind": "block", "name": "arg0"}}
        launch["stores"][0]["role"] = role
    data["observation"].update(numel=numel, value_representation=representation)
    return data


def program(spec, left, right, *, rhs_mask=None, numel=1, rhs_offset=None):
    sequences = []
    for side, value in (("lhs", left), ("rhs", right)):
        role = spec.observation.role_for_side(side)
        store = Store(
            0,
            StorageRef(side, side + ".0", "arg0", role),
            var("__lane"),
            rhs_offset if side == "rhs" and rhs_offset is not None else var("__lane"),
            rhs_mask if side == "rhs" and rhs_mask is not None else const(True, BOOL),
            value,
            (numel,),
        )
        kernel = Kernel(side + ".0", "kernel", const(1), (numel,), (store,), "unit-ir")
        sequences.append(ProgramSequence(((kernel,),)))
    return ProgramPair(*sequences)


def verify_ir(monkeypatch, path, pair):
    # This isolates the core proof state machine, not the official TTIR frontend.
    module = importlib.import_module("etv.verify")
    monkeypatch.setattr(module, "_programs", lambda *_: pair)
    return module.verify(path)


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_bool_numeric_proof_and_visible_contract(tmp_path, monkeypatch, reverse):
    path = write(tmp_path, encoded_data(tmp_path, reverse))
    spec = parse_pair_spec(path)
    boolean = var("p", BOOL)
    numeric = Expr("uitofp", FLOAT, (boolean,))
    left, right = (boolean, numeric) if reverse else (numeric, boolean)
    pair = program(spec, left, right)
    report = verify_ir(monkeypatch, path, pair)
    assert (report.status, report.reason) == ("PROVED", "LOGICAL_BOOLEAN_OUTPUT_EQUIVALENT")
    data = report_json(report)
    contract = data["comparison_contract"]
    assert contract["kind"] == "logical-value-under-encoding"
    assert contract["encoding"] == "boolean_numeric_01"
    assert contract["logical_role"] == "Output"
    assert contract["physical_roles"] == {"lhs": "Output_lhs", "rhs": "Output_rhs"}
    assert contract["physical_elements"]["lhs"] == ("bool" if reverse else "abstract_float")
    assert contract["encodings"]["rhs"] == ("identity" if reverse else "bool_to_numeric_01")
    assert contract["bit_identical"] is False
    assert contract["output_dtype_equality_claimed"] is False
    assert data["execution_scope"] == "ttir-only"
    assert data["inputs"]["host_operations"] == []
    assert data["physical_launch_count"] == data["execution_step_count"] == 2
    assert "observation.encoding.injective" in {o.id for o in report.obligations}
    predicate = next(p for p in report.predicates if p.id == "observation.value_representation")
    assert predicate.expr is None and not predicate.trusted
    assert all(r.admission == "formal" for r in report.rewrites if r.used)


@pytest.mark.parametrize("reverse", [False, True])
def test_float_two_is_not_boolean_true(tmp_path, monkeypatch, reverse):
    path = write(tmp_path, encoded_data(tmp_path, reverse))
    spec = parse_pair_spec(path)
    left, right = (
        (const(True, BOOL), const(2, FLOAT)) if reverse else (const(2, FLOAT), const(True, BOOL))
    )
    report = verify_ir(monkeypatch, path, program(spec, left, right))
    assert (report.status, report.reason) == ("DISPROVED", "COMPUTE_MISMATCH")
    assert {report.counterexample.lhs_value, report.counterexample.rhs_value} == {"1", "2"}


def test_bool_encoding_is_injective_for_all_values(tmp_path):
    spec = parse_pair_spec(write(tmp_path, encoded_data(tmp_path)))
    roots = build_obligations(program(spec, const(0, FLOAT), var("b", BOOL)), spec).roots
    encoded = roots[0].rhs
    encoder = Encoder()
    b = encoder.encode(var("b", BOOL))
    value = encoder.encode(encoded)
    solver = z3.Solver()
    solver.add(
        z3.Or(
            z3.And(b, value != encoder.encode(const(1, FLOAT))),
            z3.And(z3.Not(b), value != encoder.encode(const(0, FLOAT))),
        )
    )
    assert solver.check() == z3.unsat
    solver = z3.Solver()
    solver.add(
        z3.substitute(value, (b, z3.BoolVal(True))) == z3.substitute(value, (b, z3.BoolVal(False)))
    )
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("numel", [2, 512])
def test_encoded_proof_observes_every_element_fixed_and_symbolic(tmp_path, numel):
    spec = parse_pair_spec(write(tmp_path, encoded_data(tmp_path, numel=numel)))
    memory = build_obligations(program(spec, const(1, FLOAT), const(True, BOOL), numel=numel), spec)
    assert memory.status == "closed"
    assert len(memory.roots) == (1 if numel > 256 else numel)
    assert all(root.lhs == root.rhs == const(1, FLOAT) for root in memory.roots)
    if numel > 256:
        assert {"memory.lhs.coverage", "memory.rhs.coverage"} <= {o.id for o in memory.obligations}
    else:
        assert {int(root.index.attr("value")) for root in memory.roots} == set(range(numel))


def test_encoding_does_not_hide_missing_writer_or_out_of_domain_write(tmp_path):
    spec = parse_pair_spec(write(tmp_path, encoded_data(tmp_path, numel=2)))
    pair = program(
        spec,
        const(1, FLOAT),
        const(True, BOOL),
        numel=2,
        rhs_mask=Expr("eq", BOOL, (var("__lane"), const(0))),
    )
    memory = build_obligations(pair, spec)
    assert (memory.status, memory.reason) == ("DISPROVED", "MASK_MISMATCH")
    assert memory.counterexample.index == "1"
    assert memory.roots == ()
    pair = program(spec, const(1, FLOAT), const(True, BOOL), rhs_offset=const(2))
    memory = build_obligations(pair, spec)
    assert (memory.status, memory.reason) == ("DISPROVED", "ADDRESS_MISMATCH")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["observation"]["value_representation"].update(kind="truthiness"),
        lambda d: d["observation"]["value_representation"]["lhs"].update(
            encoding="bool_to_numeric_01"
        ),
        lambda d: d["observation"]["value_representation"]["rhs"].update(encoding="identity"),
        lambda d: d["observation"]["value_representation"]["rhs"].update(role="Missing"),
        lambda d: d["predicates"]["roles"]["Output_rhs"].update(element="abstract_float"),
        lambda d: d["predicates"]["roles"]["Output_lhs"].update(element="bool"),
        lambda d: d["observation"].update(require_full_coverage=False),
        lambda d: d["observation"]["value_representation"].pop("rhs"),
    ],
)
def test_invalid_or_weakening_encodings_are_rejected(tmp_path, mutation):
    data = encoded_data(tmp_path)
    mutation(data)
    with pytest.raises(InputError):
        parse_pair_spec(write(tmp_path, data))


def test_default_contract_is_unchanged_and_does_not_claim_bit_identity(tmp_path, monkeypatch):
    path = write(tmp_path, minimal(tmp_path))
    spec = parse_pair_spec(path)
    report = verify_ir(monkeypatch, path, program(spec, const(1, FLOAT), const(1, FLOAT)))
    assert (report.status, report.reason) == ("PROVED", "OBSERVABLE_MEMORY_EQUIVALENT")
    contract = report_json(report)["comparison_contract"]
    assert contract["kind"] == "typed-observable-values" and contract["encoding"] is None
    assert contract["physical_roles"] == {"lhs": "Output", "rhs": "Output"}
    assert not contract["bit_identical"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("numeric_type", ["f16", "f32"])
def test_real_ttir_boolean_numeric_observation(tmp_path, reverse, numeric_type):
    """Run with the official pinned Linux frontend; never replace the parser."""
    pytest.importorskip("triton._C.libtriton")
    pytest.importorskip("etv._mlir_native")
    data = encoded_data(tmp_path, reverse)
    data["predicates"]["roles"]["Input"] = {"kind": "scalar", "element": "abstract_float"}
    for side in ("lhs", "rhs"):
        boolean_side = (side == "lhs") == reverse
        typ = "i1" if boolean_side else numeric_type
        conversion = "" if boolean_side else f"%value = arith.uitofp %pred : i1 to {typ}"
        value = "%pred" if boolean_side else "%value"
        (tmp_path / f"{side}.ttir").write_text(f"""module {{
          tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<{typ}>) {{
            %zero = arith.constant 0.0 : f32
            %pred = arith.cmpf ogt, %arg0, %zero : f32
            {conversion}
            tt.store %arg1, {value} : !tt.ptr<{typ}>
            tt.return
          }}
        }}""")
        launch = data["programs"][side][0]
        launch["file"] = f"{side}.ttir"
        launch["abi"]["Output_" + side]["name"] = "arg1"
        launch["abi"]["Input"] = {"kind": "scalar", "name": "arg0"}
    report = importlib.import_module("etv.verify").verify(write(tmp_path, data))
    assert (report.status, report.reason) == ("PROVED", "LOGICAL_BOOLEAN_OUTPUT_EQUIVALENT"), report
    assert all(launch.kind == "ttir" for launch in report.launches)
