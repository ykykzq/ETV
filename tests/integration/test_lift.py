from fractions import Fraction
from pathlib import Path

import pytest

from etv.errors import ETVError
from etv.frontend import parse_ttir
from etv.ir import const, substitute
from etv.lift import lift_kernel
from etv.pairspec import Endpoint, LaunchSpec, SelectedStore
from etv.semantics import evaluate


def lift_text(tmp_path: Path, text: str, abi: tuple[Endpoint, ...], bindings=()):
    path = tmp_path / "kernel.ttir"
    path.write_text(text)
    module = parse_ttir(path, "kernel")
    launch = LaunchSpec(
        "lhs.0",
        "lhs",
        path,
        "kernel",
        0,
        const(1),
        (SelectedStore(0, "Output"),),
        abi,
        bindings,
        64,
        10000,
    )
    return lift_kernel(module, launch)


@pytest.mark.parametrize("op", "absf ceil cos erf exp exp2 floor log rsqrt sin sqrt".split())
def test_math_operations(tmp_path, op):
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<f32>) {{
      %value = math.{op} %arg0 : f32
      tt.store %arg1, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert kernel.stores[0].value.op == op


@pytest.mark.parametrize(
    "op,semantic",
    [
        ("addf", "fadd"),
        ("subf", "fsub"),
        ("mulf", "fmul"),
        ("divf", "fdiv"),
        ("minnumf", "fmin"),
        ("maxnumf", "fmax"),
    ],
)
def test_float_binary_operations(tmp_path, op, semantic):
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: f32, %arg2: !tt.ptr<f32>) {{
      %value = arith.{op} %arg0, %arg1 : f32
      tt.store %arg2, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path,
        text,
        (
            Endpoint("A", "scalar", "arg0"),
            Endpoint("B", "scalar", "arg1"),
            Endpoint("Output", "block", "arg2"),
        ),
    )
    assert kernel.stores[0].value.op == semantic


@pytest.mark.parametrize(
    "op,source,target",
    [
        ("extsi", "i8", "i32"),
        ("extui", "i8", "i32"),
        ("trunci", "i64", "i32"),
        ("index_cast", "index", "i32"),
        ("index_castui", "index", "i32"),
    ],
)
def test_integer_cast_is_retained(tmp_path, op, source, target):
    text = f"""module {{ tt.func @kernel(%arg0: {source}, %arg1: !tt.ptr<f32>) {{
      %cast = arith.{op} %arg0 : {source} to {target}
      %value = arith.sitofp %cast : {target} to f32
      tt.store %arg1, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Output", "block", "arg1"),), (("arg0", const(-1)),)
    )
    assert kernel.stores[0].value.args[0].op == op
    assert kernel.stores[0].value.op == "sitofp"


@pytest.mark.parametrize("source,target,op", [("f32", "f64", "extf"), ("f64", "f32", "truncf")])
def test_float_cast_has_source_and_target_widths(tmp_path, source, target, op):
    text = f"""module {{ tt.func @kernel(%arg0: {source}, %arg1: !tt.ptr<{target}>) {{
      %value = arith.{op} %arg0 : {source} to {target}
      tt.store %arg1, %value : !tt.ptr<{target}>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    value = kernel.stores[0].value
    assert value.op == op
    assert value.attr("source_bits") == int(source[1:])
    assert value.attr("target_bits") == int(target[1:])


@pytest.mark.parametrize(
    "symbol",
    "acosh acoshf atan atanf ceilf coshf erff expm1f floorf nearbyintf powf rsqrtf sqrtf tanhf".split(),
)
def test_extern_whitelist(tmp_path, symbol):
    operands = "%arg0, %arg0" if symbol == "powf" else "%arg0"
    types = "f32, f32" if symbol == "powf" else "f32"
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<f32>) {{
      %value = "tt.extern_elementwise"({operands}) <{{libname = "", libpath = "", symbol = "__nv_{symbol}", pure = true}}> : ({types}) -> f32
      tt.store %arg1, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert kernel.stores[0].value.type.kind == "abstract_float"


def test_broadcast_and_reshape_preserve_element_mapping(tmp_path):
    text = """module { tt.func @kernel(%arg0: !tt.ptr<f32>) {
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %e = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %b = tt.broadcast %e : tensor<1x4xi32> -> tensor<2x4xi32>
      %flat = tt.reshape %b : tensor<2x4xi32> -> tensor<8xi32>
      %value = arith.uitofp %flat : tensor<8xi32> to tensor<8xf32>
      %offset = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
      %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %pointer = tt.addptr %base, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      tt.store %pointer, %value : tensor<8x!tt.ptr<f32>>
      tt.return
    } }"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    values = [
        evaluate(substitute(kernel.stores[0].value, (("__lane", const(i)),)), index_bits=64)
        for i in range(8)
    ]
    assert values == [Fraction(i % 4) for i in range(8)]


def test_wrong_types_rejected_by_official_verifier(tmp_path):
    text = """module { tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<f32>) {
      %v = arith.addi %arg0, %arg0 : f32
      tt.store %arg1, %v : !tt.ptr<f32>
      tt.return
    } }"""
    with pytest.raises(ETVError):
        lift_text(
            tmp_path,
            text,
            (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1")),
        )


@pytest.mark.parametrize(
    "operation,semantic",
    [("addi", "add"), ("subi", "sub"), ("muli", "mul"), ("divsi", "div"), ("remsi", "rem")],
)
def test_integer_binary_lift(tmp_path, operation, semantic):
    text = f"""module {{ tt.func @kernel(%arg0: i32, %arg1: i32, %arg2: !tt.ptr<f32>) {{
      %integer = arith.{operation} %arg0, %arg1 : i32
      %value = arith.sitofp %integer : i32 to f32
      tt.store %arg2, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path,
        text,
        (Endpoint("Output", "block", "arg2"),),
        (("arg0", const(-7)), ("arg1", const(3))),
    )
    assert kernel.stores[0].value.args[0].op == semantic


@pytest.mark.parametrize("op,semantic", [("andi", "and"), ("ori", "or"), ("xori", "xor")])
def test_boolean_lift(tmp_path, op, semantic):
    text = f"""module {{ tt.func @kernel(%arg0: !tt.ptr<f32>) {{
      %a = arith.constant true
      %b = arith.constant false
      %boolean = arith.{op} %a, %b : i1
      %value = arith.uitofp %boolean : i1 to f32
      tt.store %arg0, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert kernel.stores[0].value.args[0].op == semantic


@pytest.mark.parametrize("predicate", "eq ne slt sle sgt sge".split())
def test_signed_comparisons(tmp_path, predicate):
    text = f"""module {{ tt.func @kernel(%arg0: !tt.ptr<f32>) {{
      %a = arith.constant -7 : i32
      %b = arith.constant 3 : i32
      %boolean = arith.cmpi {predicate}, %a, %b : i32
      %value = arith.uitofp %boolean : i1 to f32
      tt.store %arg0, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    expected = predicate in ("ne", "slt", "sle")
    assert evaluate(kernel.stores[0].value) == int(expected)


@pytest.mark.parametrize(
    "predicate", "false oeq ogt oge olt ole one ord ueq ugt uge ult ule une uno true".split()
)
def test_real_comparisons(tmp_path, predicate):
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<f32>) {{
      %boolean = arith.cmpf {predicate}, %arg0, %arg0 : f32
      %value = arith.select %boolean, %arg0, %arg0 : f32
      tt.store %arg1, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert kernel.stores[0].value.op == "select"


@pytest.mark.parametrize(
    "operation,semantic",
    [
        ("arith.negf %arg0", "fneg"),
        ("math.fma %arg0, %arg0, %arg0", "fma"),
        ("tt.clampf %arg0, %arg0, %arg0, propagateNan = none", "fmin"),
    ],
)
def test_ternary_and_negation(tmp_path, operation, semantic):
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<f32>) {{
      %value = {operation} : f32
      tt.store %arg1, %value : !tt.ptr<f32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert kernel.stores[0].value.op == semantic
