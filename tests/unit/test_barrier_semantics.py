"""A redundant uniform barrier must not add support for memory communication."""

from pathlib import Path

import pytest

from etv.errors import UnsupportedSemantics
from etv.frontend import Block, IRType, Operation, TTIRModule, Value
from etv.ir import const
from etv.lift import lift_kernel
from etv.pairspec import Endpoint, LaunchSpec, SelectedStore


FLOAT = IRType("float", "f32", bits=32)
POINTER = IRType("pointer", "!tt.ptr<f32>", element=FLOAT, address_space=1)
INPUT = Value("input", POINTER)
OUTPUT = Value("output", POINTER)
LOADED = Value("loaded", FLOAT)


def operation(name, operands=(), results=(), regions=()):
    return Operation(0, name, operands, results, (), 0, regions, "", name)


LOAD = operation("tt.load", (INPUT,), (LOADED,))
STORE = operation("tt.store", (OUTPUT, LOADED))
BARRIER = operation("gpu.barrier")
RETURN = operation("tt.return")


def lift(operations):
    path = Path("barrier-fixture.ttir")
    block = Block(0, (INPUT, OUTPUT), tuple(operations))
    function = operation("tt.func", regions=((block,),))
    module = TTIRModule(path, "fixture", "3.7.1", function, function)
    launch = LaunchSpec(
        "lhs.0",
        "lhs",
        path,
        "kernel",
        0,
        const(1),
        (SelectedStore(0, "Output"),),
        (Endpoint("Input", "block", "arg0"), Endpoint("Output", "block", "arg1")),
        (),
        64,
        100,
    )
    return lift_kernel(module, launch)


def test_read_barrier_write_has_identical_symbolic_semantics():
    reference = lift((LOAD, STORE, RETURN))
    synchronized = lift((LOAD, BARRIER, STORE, RETURN))
    assert synchronized == reference
    assert synchronized.stores[0].value.op == "load"
    assert synchronized.effects == (synchronized.stores[0].value,)


@pytest.mark.parametrize(
    "prefix,reason",
    [
        ((LOAD, STORE), "GPU_BARRIER_MEMORY_DEPENDENCE_UNSUPPORTED"),
        (
            (LOAD, operation("tt.atomic_rmw", (OUTPUT, LOADED), (Value("atomic", FLOAT),))),
            "TTIR_TYPE_UNSUPPORTED",
        ),
    ],
)
def test_barrier_does_not_admit_prior_store_or_atomic_communication(prefix, reason):
    with pytest.raises(UnsupportedSemantics) as error:
        lift((*prefix, BARRIER, LOAD, STORE, RETURN))
    assert error.value.reason == reason
