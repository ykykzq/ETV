from fractions import Fraction
from pathlib import Path

import pytest

from etv.errors import ETVError
from etv.frontend import parse_ttir
from etv.ir import const, substitute
from etv.lift import lift_kernel
from etv.pairspec import Endpoint, LaunchSpec, SelectedStore
from etv.semantics import evaluate


def lift_text(
    tmp_path: Path, text: str, abi: tuple[Endpoint, ...], bindings=(), canonical_bool=False
):
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
        canonical_bool,
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


@pytest.mark.parametrize(
    "operation,expected",
    [("arith.shrui", 1073741822), ("tt.mulhiui", 1), ("arith.minsi", -7), ("arith.maxsi", 2)],
)
def test_extended_integer_operations(tmp_path, operation, expected):
    text = f"""module {{ tt.func @kernel(%out: !tt.ptr<i32>) {{
      %a = arith.constant -7 : i32
      %b = arith.constant 2 : i32
      %v = {operation} %a, %b : i32
      tt.store %out, %v : !tt.ptr<i32>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) == expected


@pytest.mark.parametrize(
    "predicate,expected", [("ult", False), ("ule", False), ("ugt", True), ("uge", True)]
)
def test_unsigned_comparison_high_bit(tmp_path, predicate, expected):
    text = f"""module {{ tt.func @kernel(%out: !tt.ptr<i1>) {{
      %a = arith.constant -1 : i8
      %b = arith.constant 1 : i8
      %v = arith.cmpi {predicate}, %a, %b : i8
      tt.store %out, %v : !tt.ptr<i1>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) is expected


@pytest.mark.parametrize("operation,value", [("fptosi", -3), ("fptoui", 3)])
def test_float_integer_cast_truncates_toward_zero(tmp_path, operation, value):
    literal = "-3.75" if value < 0 else "3.75"
    text = f"""module {{ tt.func @kernel(%out: !tt.ptr<i32>) {{
      %a = arith.constant {literal} : f32
      %v = arith.{operation} %a : f32 to i32
      tt.store %out, %v : !tt.ptr<i32>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) == value


def test_gather_static_indices_and_bounds(tmp_path):
    text = """module { tt.func @kernel(%out: !tt.ptr<i32>) {
      %a = arith.constant dense<[11, 22, 33, 44]> : tensor<4xi32>
      %i = arith.constant dense<[3, 0, 2, 1]> : tensor<4xi32>
      %v = tt.gather %a[%i] {axis = 0 : i32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %p = tt.splat %out : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %q = tt.addptr %p, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      tt.store %q, %v : tensor<4x!tt.ptr<i32>>
      tt.return
    } }"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert [evaluate(kernel.stores[0].value, {"__lane": i}) for i in range(4)] == [44, 11, 33, 22]
    bad = lift_text(
        tmp_path,
        text.replace("[3, 0, 2, 1]", "[4, 0, 2, 1]"),
        (Endpoint("Output", "block", "arg0"),),
    )
    with pytest.raises(ETVError, match="UNDEFINED_LOAD_REACHABLE"):
        evaluate(bad.stores[0].value, {"__lane": 0})


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
    "predicate", "false oeq ogt oge olt ole one ord ueq ugt uge ult ule une uno true".split()
)
@pytest.mark.parametrize("left,right", [(-1, 0), (1, 0), (0, 0)])
def test_all_float_predicates_match_finite_real_truth_table(tmp_path, predicate, left, right):
    text = f"""module {{ tt.func @kernel(%arg0: !tt.ptr<i1>) {{
      %a = arith.constant {left}.0 : f32
      %b = arith.constant {right}.0 : f32
      %v = arith.cmpf {predicate}, %a, %b : f32
      tt.store %arg0, %v : !tt.ptr<i1>
      tt.return
    }} }}"""
    expected = {
        "false": False,
        "true": True,
        "ord": True,
        "uno": False,
        "eq": left == right,
        "gt": left > right,
        "ge": left >= right,
        "lt": left < right,
        "le": left <= right,
        "ne": left != right,
    }
    key = predicate[1:] if predicate not in ("false", "true", "ord", "uno") else predicate
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) == expected[key]


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


def test_signed_i1_comparison(tmp_path):
    text = """module { tt.func @kernel(%arg0: !tt.ptr<f32>) {
      %a = arith.constant true
      %b = arith.constant false
      %lt = arith.cmpi slt, %a, %b : i1
      %value = arith.uitofp %lt : i1 to f32
      tt.store %arg0, %value : !tt.ptr<f32>
      tt.return
    } }"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) == 1


@pytest.mark.parametrize("typ", ["i1", "i8", "i32", "i64"])
def test_typed_memory_round_trip(tmp_path, typ):
    text = f"""module {{ tt.func @kernel(%arg0: !tt.ptr<{typ}>, %arg1: !tt.ptr<{typ}>) {{
      %value = tt.load %arg0 : !tt.ptr<{typ}>
      tt.store %arg1, %value : !tt.ptr<{typ}>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "block", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert kernel.stores[0].value.op == "load"
    assert str(kernel.stores[0].value.type) == ("bool" if typ == "i1" else typ)


def test_boolean_byte_store_preserves_logical_value(tmp_path):
    text = """module { tt.func @kernel(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i1>) {
      %value = tt.load %arg0 : !tt.ptr<i1>
      %byte = arith.extui %value : i1 to i8
      %pointer = tt.bitcast %arg1 : !tt.ptr<i1> -> !tt.ptr<i8>
      tt.store %pointer, %byte : !tt.ptr<i8>
      tt.return
    } }"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "block", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert str(kernel.stores[0].value.type) == "bool"
    assert kernel.stores[0].value.op == "load"
    bad = text.replace("%byte = arith.extui %value : i1 to i8", "%byte = arith.constant 2 : i8")
    with pytest.raises(ETVError, match="noncanonical boolean byte store"):
        lift_text(
            tmp_path, bad, (Endpoint("Input", "block", "arg0"), Endpoint("Output", "block", "arg1"))
        )


@pytest.mark.parametrize("axis,expected", [(0, [6, 8, 10, 12]), (1, [10, 26])])
def test_static_reduction_reindexes_removed_axis(tmp_path, axis, expected):
    size = len(expected)
    text = f"""module {{ tt.func @kernel(%arg0: !tt.ptr<f32>) {{
      %v = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
      %s = "tt.reduce"(%v) <{{axis = {axis} : i32}}> ({{
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }}) : (tensor<2x4xf32>) -> tensor<{size}xf32>
      %offset = tt.make_range {{start = 0 : i32, end = {size} : i32}} : tensor<{size}xi32>
      %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>>
      %p = tt.addptr %base, %offset : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32>
      tt.store %p, %s : tensor<{size}x!tt.ptr<f32>>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    actual = [
        evaluate(substitute(kernel.stores[0].value, (("__lane", const(i)),))) for i in range(size)
    ]
    assert actual == expected


@pytest.mark.parametrize("symbol", ["isnan", "isnanf"])
def test_isnan_is_false_in_finite_real_semantics(tmp_path, symbol):
    text = f"""module {{ tt.func @kernel(%arg0: f32, %arg1: !tt.ptr<i32>) {{
      %value = "tt.extern_elementwise"(%arg0) <{{libname = "", libpath = "", symbol = "__nv_{symbol}", pure = true}}> : (f32) -> i32
      tt.store %arg1, %value : !tt.ptr<i32>
      tt.return
    }} }}"""
    kernel = lift_text(
        tmp_path, text, (Endpoint("Input", "scalar", "arg0"), Endpoint("Output", "block", "arg1"))
    )
    assert evaluate(kernel.stores[0].value) == 0


def test_boolean_byte_read_requires_explicit_representation(tmp_path):
    text = """module { tt.func @kernel(%in: !tt.ptr<i1>, %out: !tt.ptr<i1>) {
      %p = tt.bitcast %in : !tt.ptr<i1> -> !tt.ptr<i8>
      %b = tt.load %p : !tt.ptr<i8>
      %z = arith.constant 0 : i8
      %v = arith.cmpi ne, %b, %z : i8
      tt.store %out, %v : !tt.ptr<i1>
      tt.return
    } }"""
    abi = (Endpoint("Input", "block", "arg0"), Endpoint("Output", "block", "arg1"))
    with pytest.raises(ETVError, match="canonical storage contract"):
        lift_text(tmp_path, text, abi)
    kernel = lift_text(tmp_path, text, abi, canonical_bool=True)
    assert kernel.stores[0].value.args[0].op == "extui"
    assert kernel.stores[0].value.args[0].args[0].type.kind == "bool"


@pytest.mark.parametrize("operation,expected", [("minsi", True), ("maxsi", False)])
def test_signed_boolean_minmax(tmp_path, operation, expected):
    text = f"""module {{ tt.func @kernel(%out: !tt.ptr<i1>) {{
      %a = arith.constant true
      %b = arith.constant false
      %v = arith.{operation} %a, %b : i1
      tt.store %out, %v : !tt.ptr<i1>
      tt.return
    }} }}"""
    kernel = lift_text(tmp_path, text, (Endpoint("Output", "block", "arg0"),))
    assert evaluate(kernel.stores[0].value) is expected
