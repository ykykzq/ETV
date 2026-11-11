from fractions import Fraction

import pytest

from etv.errors import UnsupportedSemantics
from etv.frontend import Attribute, Block, IRType, Operation, Value, _attribute
from etv.ir import FLOAT
from etv.lift import _constant, _nonzero_only_isinf_uses


@pytest.mark.parametrize("special", ["+inf", "-inf", "nan"])
def test_apfloat_special_tag_is_preserved(special):
    attr = _attribute(
        "value", {"kind": "float", "assembly": "opaque spelling", "value": {"special": special}}
    )
    assert attr.values == (special,)


def test_dense_apfloat_tags_preserve_positions_and_finite_values():
    attr = _attribute(
        "value",
        {
            "kind": "dense",
            "assembly": "opaque spelling",
            "value": [0.5, {"special": "+inf"}, {"special": "-inf"}, -2.0],
        },
    )
    assert attr.values == (Fraction(1, 2), "+inf", "-inf", Fraction(-2))
    integers = _attribute("value", {"kind": "dense", "assembly": "", "value": ["-1", "2"]})
    assert integers.values == (-1, 2)


@pytest.mark.parametrize("raw", [{"special": "unknown"}, {"special": "+inf", "extra": 1}])
def test_unknown_special_tag_fails_closed(raw):
    with pytest.raises(UnsupportedSemantics, match="invalid APFloat tag"):
        _attribute("value", {"kind": "float", "assembly": "", "value": raw})


def test_nan_tag_is_not_replaced_by_real_or_infinity():
    attr = Attribute("value", "float", "NaN", ("nan",))
    op = Operation(0, "arith.constant", (), (), (attr,), 0, (), "", "")
    with pytest.raises(UnsupportedSemantics) as error:
        _constant(op, FLOAT, ())
    assert error.value.reason == "TTIR_NAN_CONSTANT_UNSUPPORTED"


@pytest.mark.parametrize("special", ["+inf", "-inf"])
def test_lift_keeps_tagged_infinity_exact(special):
    attr = Attribute("value", "float", "opaque spelling", (special,))
    op = Operation(0, "arith.constant", (), (), (attr,), 0, (), "", "")
    value = _constant(op, FLOAT, ())
    assert value.op == "const" and value.attr("value") == special


def test_isinf_zero_use_audit_checks_nested_uses_too():
    integer = IRType("int", "i32", bits=32)
    floating = IRType("float", "f32", bits=32)
    x, raw, zero = Value("x", floating), Value("raw", integer), Value("zero", integer)
    external = Operation(
        1,
        "tt.extern_elementwise",
        (x,),
        (raw,),
        (Attribute("symbol", "string", "", ("__nv_isinff",)),),
        0,
        (),
        "",
        "",
    )
    constant = Operation(
        2, "arith.constant", (), (zero,), (Attribute("value", "integer", "", (0,)),), 0, (), "", ""
    )
    compare = Operation(
        3,
        "arith.cmpi",
        (raw, zero),
        (),
        (Attribute("predicate", "integer", "", (1,)),),
        0,
        (),
        "",
        "",
    )
    assert _nonzero_only_isinf_uses((external, constant, compare)) == frozenset({1})
    escaped = Operation(4, "arith.addi", (raw, zero), (), (), 1, (), "", "")
    region = Operation(5, "tt.reduce", (), (), (), 0, ((Block(1, (), (escaped,)),),), "", "")
    assert not _nonzero_only_isinf_uses((external, constant, compare, region))
