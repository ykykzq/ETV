"""Aliased read/write access needs a dependence proof, not snapshot semantics."""

import importlib

import pytest

from etv.ir import BOOL, FLOAT, INDEX, Expr, Kernel, ProgramPair, ProgramSequence, StorageRef, Store
from etv.ir import const, var, walk
from etv.memory import build_obligations
from etv.pairspec import parse_pair_spec
from etv.verify import verify
from test_contracts import minimal, write


def load(role, offset, side="lhs", physical_arg="arg0"):
    storage = StorageRef(side, "load", physical_arg, role)
    return Expr(
        "load",
        FLOAT,
        (offset, const(True, BOOL), Expr("undefined", FLOAT)),
        (("storage", storage),),
    )


def kernel(name, role, offset, value, programs=1, effects=()):
    storage = StorageRef(name[:3], name, "arg0", role)
    store = Store(0, storage, offset, offset, const(True, BOOL), value)
    return Kernel(name, "kernel", const(programs), (), (store,), "unit-ir", effects=effects)


def case(tmp_path, lhs_steps, rhs_steps, numel=3):
    data = minimal(tmp_path)
    data["predicates"]["roles"] = {
        name: {"kind": "buffer", "element": "abstract_float"}
        for name in ("Input", "Internal", "Output")
    }
    data["predicates"]["disjoint"] = [["Input", "Internal", "Output"]]
    data["observation"]["numel"] = numel
    for side, steps in (("lhs", lhs_steps), ("rhs", rhs_steps)):
        data["programs"][side] = []
        for index, step in enumerate(steps):
            for item in step:
                roles = {s.storage.logical_role for s in item.stores}
                for expr in item.effects + tuple(s.value for s in item.stores):
                    roles.update(
                        n.attr("storage").logical_role for n in walk(expr) if n.op == "load"
                    )
                data["programs"][side].append(
                    {
                        "id": item.launch_id,
                        "file": "kernel.ttir",
                        "function": "kernel",
                        "step": index,
                        "grid": {"programs": int(item.programs.attr("value"))},
                        "stores": [
                            {"index": s.store_index, "role": s.storage.logical_role}
                            for s in item.stores
                        ],
                        "abi": {
                            role: {"kind": "block", "name": f"arg{i}"}
                            for i, role in enumerate(sorted(roles))
                        },
                        "bindings": {},
                    }
                )
    path = write(tmp_path, data)
    return (
        path,
        parse_pair_spec(path),
        ProgramPair(ProgramSequence(lhs_steps), ProgramSequence(rhs_steps)),
    )


def test_initialized_internal_cross_program_shift_cannot_be_proved(tmp_path, monkeypatch):
    pid = var("__pid")

    def choose(index, yes, no):
        return Expr("select", FLOAT, (Expr("eq", BOOL, (pid, const(index))), yes, no))

    initial = choose(2, const(2, FLOAT), choose(1, const(1, FLOAT), const(0, FLOAT)))
    init = kernel("lhs.0", "Internal", pid, initial, 3)
    update = kernel(
        "lhs.1", "Internal", Expr("add", INDEX, (pid, const(1))), load("Internal", pid), 2
    )
    output = kernel("lhs.2", "Output", pid, load("Internal", pid), 3)
    reference = kernel("rhs.0", "Output", pid, choose(2, const(1, FLOAT), const(0, FLOAT)), 3)
    path, spec, pair = case(tmp_path, ((init,), (update,), (output,)), ((reference,),))
    memory = build_obligations(pair, spec)
    assert (memory.status, memory.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED")
    assert not memory.roots
    assert "Internal" in memory.obligations[-1].detail
    # A permitted inter-program execution differs from the snapshot result.
    actual = [0, 1, 2]
    for program in (0, 1):
        actual[program + 1] = actual[program]
    assert actual == [0, 0, 0] and actual != [0, 0, 1]
    module = importlib.import_module("etv.verify")
    monkeypatch.setattr(module, "_programs", lambda *_: pair)
    report = verify(path)
    assert (report.status, report.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED")
    assert not report.equalities


def test_same_step_producer_consumer_requires_ordering(tmp_path):
    producer = kernel("lhs.0", "Internal", const(0), const(1, FLOAT))
    consumer = kernel("lhs.1", "Output", const(0), load("Internal", const(0)))
    reference = kernel("rhs.0", "Output", const(0), const(1, FLOAT))
    _, spec, pair = case(tmp_path, ((producer, consumer),), ((reference,),), numel=1)
    memory = build_obligations(pair, spec)
    assert (memory.status, memory.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED")


def test_ordered_prior_producer_and_readonly_alias_views_remain_supported(tmp_path):
    pid = var("__pid")
    producer = kernel("lhs.0", "Internal", pid, load("Input", pid), 3)
    a = load("Internal", const(0), physical_arg="arg0")
    b = load("Internal", const(2), physical_arg="arg1")
    consumer = kernel("lhs.1", "Output", const(0), Expr("fadd", FLOAT, (a, b)))
    expected = Expr("fadd", FLOAT, (load("Input", const(0), "rhs"), load("Input", const(2), "rhs")))
    reference = kernel("rhs.0", "Output", const(0), expected)
    _, spec, pair = case(tmp_path, ((producer,), (consumer,)), ((reference,),), numel=1)
    memory = build_obligations(pair, spec)
    assert memory.status == "closed", memory.reason
    assert len(memory.roots) == 1 and memory.roots[0].lhs == memory.roots[0].rhs


def test_unused_load_effect_also_requires_dependency_proof(tmp_path):
    unused = load("Output", const(0))
    lhs = kernel("lhs.0", "Output", const(0), const(1, FLOAT), effects=(unused,))
    rhs = kernel("rhs.0", "Output", const(0), const(1, FLOAT))
    _, spec, pair = case(tmp_path, ((lhs,),), ((rhs,),), numel=1)
    assert build_obligations(pair, spec).reason == "INPLACE_MEMORY_DEPENDENCY_UNPROVED"


@pytest.mark.parametrize("numel", [1, 512])
def test_inplace_guard_applies_before_fixed_or_symbolic_proof(tmp_path, numel):
    pid = var("__pid")
    lhs = kernel("lhs.0", "Output", pid, load("Output", pid), numel)
    rhs = kernel("rhs.0", "Output", pid, const(1, FLOAT), numel)
    _, spec, pair = case(tmp_path, ((lhs,),), ((rhs,),), numel=numel)
    assert build_obligations(pair, spec).reason == "INPLACE_MEMORY_DEPENDENCY_UNPROVED"


def test_real_ttir_initialized_alias_update_is_not_a_proof(tmp_path):
    pytest.importorskip("triton._C.libtriton")
    pytest.importorskip("etv._mlir_native")
    data = minimal(tmp_path)
    data["predicates"]["roles"]["Internal"] = {"kind": "buffer", "element": "abstract_float"}
    data["predicates"]["disjoint"] = [["Internal", "Output"]]
    data["observation"]["numel"] = 3
    bodies = {
        "init": (
            "%arg0: !tt.ptr<f32>",
            """
          %value = arith.sitofp %pid : i32 to f32
          %p = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
          tt.store %p, %value : !tt.ptr<f32>
        """,
        ),
        "update": (
            "%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>",
            """
          %one = arith.constant 1 : i32
          %next = arith.addi %pid, %one : i32
          %p = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
          %q = tt.addptr %arg1, %next : !tt.ptr<f32>, i32
          %value = tt.load %p : !tt.ptr<f32>
          tt.store %q, %value : !tt.ptr<f32>
        """,
        ),
        "output": (
            "%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>",
            """
          %p = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
          %q = tt.addptr %arg1, %pid : !tt.ptr<f32>, i32
          %value = tt.load %p : !tt.ptr<f32>
          tt.store %q, %value : !tt.ptr<f32>
        """,
        ),
        "reference": (
            "%arg0: !tt.ptr<f32>",
            """
          %two = arith.constant 2 : i32
          %zero = arith.constant 0.0 : f32
          %one = arith.constant 1.0 : f32
          %condition = arith.cmpi eq, %pid, %two : i32
          %value = arith.select %condition, %one, %zero : f32
          %p = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
          tt.store %p, %value : !tt.ptr<f32>
        """,
        ),
    }
    for name, (args, body) in bodies.items():
        (tmp_path / f"{name}.ttir").write_text(
            f"module {{ tt.func @kernel({args}) {{\n"
            f"%pid = tt.get_program_id x : i32\n{body}\ntt.return\n}} }}"
        )

    def endpoint(index):
        return {"kind": "block", "name": f"arg{index}"}

    def launch(side, step, name, role, abi, programs):
        return {
            "id": f"{side}.{step}",
            "file": f"{name}.ttir",
            "function": "kernel",
            "step": step,
            "grid": {"programs": programs},
            "bindings": {},
            "abi": abi,
            "stores": [{"index": 0, "role": role}],
        }

    data["programs"] = {
        "lhs": [
            launch("lhs", 0, "init", "Internal", {"Internal": endpoint(0)}, 3),
            launch("lhs", 1, "update", "Internal", {"Internal": [endpoint(0), endpoint(1)]}, 2),
            launch(
                "lhs", 2, "output", "Output", {"Internal": endpoint(0), "Output": endpoint(1)}, 3
            ),
        ],
        "rhs": [launch("rhs", 0, "reference", "Output", {"Output": endpoint(0)}, 3)],
    }
    report = verify(write(tmp_path, data))
    assert (report.status, report.reason) == ("UNKNOWN", "INPLACE_MEMORY_DEPENDENCY_UNPROVED"), (
        report
    )
    assert not report.equalities
