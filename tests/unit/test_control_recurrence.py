from pathlib import Path

from etv.frontend import Attribute, Block, IRType, Operation, TTIRModule, Value
from etv.ir import const
from etv.lift import lift_kernel
from etv.pairspec import Endpoint, LaunchSpec, SelectedStore
from etv.semantics import evaluate


def test_signed_constant_recurrence_uses_modular_closed_form():
    integer = IRType("int", "i32", bits=32)
    out = Value("out", IRType("pointer", "!tt.ptr<i32>", element=integer, address_space=1))
    lower, upper, step, seed, induction, carried, next_value, result = (
        Value(name, integer)
        for name in ("lower", "upper", "step", "seed", "iv", "carried", "next", "result")
    )

    def op(name, operands=(), results=(), attributes=(), regions=()):
        return Operation(0, name, operands, results, attributes, None, regions, "", name)

    def constant(target, value):
        return op(
            "arith.constant",
            results=(target,),
            attributes=(Attribute("value", "integer", str(value), (value,)),),
        )

    body = Block(
        1,
        (induction, carried),
        (
            op("arith.addi", (carried, step), (next_value,)),
            op("scf.yield", (next_value,)),
        ),
    )
    function = op(
        "tt.func",
        regions=(
            (
                Block(
                    0,
                    (out,),
                    (
                        constant(lower, 0),
                        constant(upper, 2000000),
                        constant(step, 1),
                        constant(seed, 2147483647),
                        op("scf.for", (lower, upper, step, seed), (result,), regions=((body,),)),
                        op("tt.store", (out, result)),
                        op("tt.return"),
                    ),
                ),
            ),
        ),
    )
    path = Path("recurrence.ttir")
    module = TTIRModule(path, "fixture", "3.7.1", function, function)
    launch = LaunchSpec(
        "lhs.0",
        "lhs",
        path,
        "kernel",
        0,
        const(1),
        (SelectedStore(0, "Output"),),
        (Endpoint("Output", "block", "arg0"),),
        (),
        64,
        10,
    )
    kernel = lift_kernel(module, launch)
    assert evaluate(kernel.stores[0].value) == -2145483649
