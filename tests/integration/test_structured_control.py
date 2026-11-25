from pathlib import Path

import pytest

from etv.errors import UnsupportedSemantics
from etv.ir import BOOL, const, substitute
from etv.pairspec import Endpoint
from etv.semantics import evaluate

from test_lift import lift_text


OUTPUT = Endpoint("Output", "block", "arg0")


def test_if_yield_selects_taken_value(tmp_path: Path):
    kernel = lift_text(
        tmp_path,
        """module {
      tt.func @kernel(%out: !tt.ptr<i32>, %c: i1) {
        %one = arith.constant 1 : i32
        %two = arith.constant 2 : i32
        %x = scf.if %c -> i32 {
          scf.yield %one : i32
        } else {
          scf.yield %two : i32
        }
        tt.store %out, %x : !tt.ptr<i32>
        tt.return
      }
    }""",
        (OUTPUT, Endpoint("Condition", "scalar", "arg1")),
    )
    assert evaluate(kernel.stores[0].value, {"scalar.Condition": True}) == 1
    assert evaluate(kernel.stores[0].value, {"scalar.Condition": False}) == 2


@pytest.mark.parametrize("upper, expected", [(0, (1, 2)), (5, (2, 1)), (127, (1, 2))])
def test_for_yields_parallel_carried_values(tmp_path: Path, upper, expected):
    kernel = lift_text(
        tmp_path,
        f"""module {{
      tt.func @kernel(%out: !tt.ptr<i32>) {{
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %two = arith.constant 2 : i32
        %upper = arith.constant {upper} : i32
        %x, %y = scf.for %i = %zero to %upper step %two
            iter_args(%a = %one, %b = %two) -> (i32, i32) : i32 {{
          scf.yield %b, %a : i32, i32
        }}
        %t = arith.muli %x, %two : i32
        %v = arith.addi %t, %y : i32
        tt.store %out, %v : !tt.ptr<i32>
        tt.return
      }}
    }}""",
        (OUTPUT,),
    )
    assert evaluate(kernel.stores[0].value) == expected[0] * 2 + expected[1]


def test_for_retains_induction_offsets_and_static_store_identity(tmp_path: Path):
    kernel = lift_text(
        tmp_path,
        """module {
      tt.func @kernel(%out: !tt.ptr<i32>) {
        %lb = arith.constant -1 : i32
        %ub = arith.constant 4 : i32
        %step = arith.constant 2 : i32
        scf.for %i = %lb to %ub step %step : i32 {
          %off = arith.addi %i, %step : i32
          %p = tt.addptr %out, %off : !tt.ptr<i32>, i32
          tt.store %p, %i : !tt.ptr<i32>
        }
        tt.return
      }
    }""",
        (OUTPUT,),
    )
    assert [(s.store_index, evaluate(s.offset), evaluate(s.value)) for s in kernel.stores] == [
        (0, 1, -1),
        (0, 3, 1),
        (0, 5, 3),
    ]


def test_branch_store_mask_contains_condition(tmp_path: Path):
    kernel = lift_text(
        tmp_path,
        """module {
      tt.func @kernel(%out: !tt.ptr<i32>, %c: i1) {
        %one = arith.constant 1 : i32
        scf.if %c {
          tt.store %out, %one : !tt.ptr<i32>
        }
        tt.return
      }
    }""",
        (OUTPUT, Endpoint("Condition", "scalar", "arg1")),
    )
    mask = kernel.stores[0].mask
    for taken in (True, False):
        assert evaluate(substitute(mask, (("scalar.Condition", const(taken, BOOL)),))) is taken


def test_for_rejects_dynamic_bound(tmp_path: Path):
    with pytest.raises(UnsupportedSemantics, match="scf.for.*static"):
        lift_text(
            tmp_path,
            """module {
          tt.func @kernel(%out: !tt.ptr<i32>, %n: i32) {
            %zero = arith.constant 0 : i32
            %one = arith.constant 1 : i32
            scf.for %i = %zero to %n step %one : i32 {
              scf.yield
            }
            tt.store %out, %zero : !tt.ptr<i32>
            tt.return
          }
        }""",
            (OUTPUT, Endpoint("Count", "scalar", "arg1")),
        )


@pytest.mark.parametrize("reference_delta, expected", [(0, "PROVED"), (1, "DISPROVED")])
def test_structured_kernel_verifies_against_straight_line(tmp_path, reference_delta, expected):
    import json

    from etv.verify import verify

    header = "module { tt.func @kernel(%out: !tt.ptr<i32>, %condition: i1) {"
    body = """
      %from = arith.constant 0 : i32
      %to = arith.constant 3 : i32
      %one = arith.constant 1 : i32
      %initial = scf.if %condition -> i32 {
        scf.yield %one : i32
      } else {
        scf.yield %from : i32
      }
      %result = scf.for %i = %from to %to step %one
          iter_args(%acc = %initial) -> (i32) : i32 {
        %next = arith.addi %acc, %one : i32
        scf.yield %next : i32
      }
    """
    reference = f"""
      %true = arith.constant {4 + reference_delta} : i32
      %false = arith.constant {3 + reference_delta} : i32
      %result = arith.select %condition, %true, %false : i32
    """
    for side, code in (("lhs", body), ("rhs", reference)):
        (tmp_path / f"{side}.ttir").write_text(
            header + code + "tt.store %out, %result : !tt.ptr<i32>\ntt.return\n} }"
        )
    pair = {
        "format": "etv-pair-v3",
        "metadata": {
            "pair_id": "structured_control",
            "execution_scope": "ttir-only",
            "semantic_mode": "abstract_float",
            "limits": {
                "smt_timeout_ms": 2000,
                "egraph_timeout_ms": 10000,
                "max_iterations": 8,
                "max_enodes": 20000,
            },
        },
        "programs": {
            side: [
                {
                    "id": f"{side}.0",
                    "file": f"{side}.ttir",
                    "function": "kernel",
                    "step": 0,
                    "grid": {"programs": 1},
                    "stores": [{"index": 0, "role": "Output"}],
                    "abi": {
                        "Output": {"kind": "block", "name": "arg0"},
                        "Condition": {"kind": "scalar", "name": "arg1"},
                    },
                    "bindings": {},
                }
            ]
            for side in ("lhs", "rhs")
        },
        "assumptions": {"for_llm": []},
        "predicates": {
            "roles": {
                "Output": {"kind": "buffer", "element": "i32"},
                "Condition": {"kind": "scalar", "element": "bool"},
            },
            "bindings": {},
            "parameters": {},
            "constraints": [],
            "disjoint": [],
            "custom": [],
        },
        "observation": {
            "role": "Output",
            "numel": 1,
            "require_full_coverage": True,
            "require_disjoint": [],
        },
        "rewrites": [],
    }
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(pair))
    report = verify(path)
    assert report.status == expected, (report.reason, report.diagnostic)
    if reference_delta:
        assert report.reason == "COMPUTE_MISMATCH"
        assert report.counterexample is not None
