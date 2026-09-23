"""Generate small manual regression artifacts from stable TTIR inputs."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def contract(case: Path, pair: dict, expected: dict, mode: str = "manual") -> None:
    write(case / "pair.json", pair)
    write(case / "expected.json", expected)
    artifacts = []
    for side in ("lhs", "rhs"):
        for ordinal, launch in enumerate(pair["programs"][side]):
            path = case / launch["file"]
            artifacts.append(
                {
                    "path": launch["file"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "side": side,
                    "launch": ordinal,
                    "step": launch["step"],
                }
            )
    write(
        case / "manifest.json",
        {
            "format": "etv-benchmark-case-v1",
            "case_id": pair["metadata"]["pair_id"],
            "status": "ready",
            "generation": {"mode": mode, "tool_version": "0.3.0"},
            "source": {
                "repository": "https://github.com/ykykzq/ETV",
                "commit": "309d01d93427fc557265de5a2f452c5da27b20d7",
                "test_id": pair["metadata"]["pair_id"],
            },
            "environment": {
                "python": "3.12",
                "torch": "not required",
                "triton": "3.7.1",
                "cuda": "not required",
                "gpu": "not required",
            },
            "artifacts": artifacts,
            "pair": "pair.json",
            "provenance": "provenance.json",
            "expected": "expected.json",
        },
    )
    if not (case / "provenance.json").exists():
        write(
            case / "provenance.json",
            {
                "format": "etv-artifact-provenance-v1",
                "capture": artifacts,
                "facts": {
                    "abi": "explicit positional signature",
                    "grid": "manual fixed specialization",
                    "storage": "logical roles declared by fixture",
                },
                "review": [
                    {
                        "status": "reviewed",
                        "pair_sha256": hashlib.sha256(
                            (case / "pair.json").read_bytes()
                        ).hexdigest(),
                    }
                ],
            },
        )


def kernel(function: str, computation: str = "%x", mask: str = "%mask", delta: int = 0) -> str:
    return f"""module {{
  tt.func public @{function}(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {{
    %range = tt.make_range {{start = 0 : i32, end = 4 : i32}} : tensor<4xi32>
    %four = arith.constant dense<4> : tensor<4xi32>
    %mask = arith.cmpi slt, %range, %four : tensor<4xi32>
    %three = arith.constant dense<3> : tensor<4xi32>
    %badmask = arith.cmpi slt, %range, %three : tensor<4xi32>
    %shift = arith.constant dense<{delta}> : tensor<4xi32>
    %offset = arith.addi %range, %shift : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pointer = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %x = tt.load %pointer, %mask : tensor<4x!tt.ptr<f32>>
    %two = arith.constant dense<2.0> : tensor<4xf32>
    %one = arith.constant dense<1.0> : tensor<4xf32>
    %double = arith.mulf %x, %two : tensor<4xf32>
    %increment = arith.addf %x, %one : tensor<4xf32>
    %fused = arith.addf %double, %one : tensor<4xf32>
    %out = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %outptr = tt.addptr %out, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %outptr, {computation}, {mask} : tensor<4x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def main() -> None:
    base_path = ROOT / "examples/add"
    base = json.loads((base_path / "pair.json").read_text())
    contract(base_path, base, {"status": "PROVED"})
    audit_path = base_path / "provenance.json"
    audit = json.loads(audit_path.read_text())
    audit["review"] = [
        {
            "status": "reviewed",
            "change": "Declare Alpha separated from buffer roles: RHS scalar_block is memory, LHS scalar is passed by value.",
            "pair_sha256": hashlib.sha256((base_path / "pair.json").read_bytes()).hexdigest(),
        }
    ]
    write(audit_path, audit)
    param = copy.deepcopy(base)
    param["metadata"]["pair_id"] = "parametric_2d_1d_add"
    param["metadata"]["limits"]["smt_timeout_ms"] = 10000
    param["predicates"]["bindings"] = {}
    param["predicates"]["parameters"] = {
        name: {"type": "i32", "min": 1, "max": maximum}
        for name, maximum in (("a", 16), ("b", 16), ("c", 256))
    }
    param["predicates"]["constraints"] = [
        {
            "id": "shape.product",
            "expr": {
                "op": "eq",
                "args": [{"op": "mul", "args": [{"var": "a"}, {"var": "b"}]}, {"var": "c"}],
            },
        }
    ]
    param["observation"]["numel"] = {"var": "c"}
    for side, size, rank in (("lhs", 256, "2d"), ("rhs", 128, "1d")):
        launch = param["programs"][side][0]
        launch["file"] = f"ttir/parametric_{rank}_add.ttir"
        launch["function"] = f"parametric_{rank}_add"
        launch["grid"] = {"programs": {"op": "ceildiv", "args": [{"var": "c"}, size]}}
        launch["bindings"] = (
            {"arg4": {"var": "a"}, "arg5": {"var": "b"}}
            if side == "lhs"
            else {"arg4": {"var": "c"}}
        )
    write(base_path / "pair_parametric.json", param)
    simple = copy.deepcopy(base)
    simple["predicates"] = {
        "roles": {
            role: {"kind": "buffer", "element": "abstract_float"} for role in ("Input", "Output")
        },
        "bindings": {},
        "parameters": {},
        "constraints": [],
        "disjoint": [["Input", "Output"]],
        "custom": [],
    }
    simple["observation"] = {
        "role": "Output",
        "numel": 4,
        "require_full_coverage": True,
        "require_disjoint": ["Input", "Output"],
    }
    for side in ("lhs", "rhs"):
        simple["programs"][side] = [
            {
                "id": side + ".0",
                "file": side + ".ttir",
                "function": side,
                "step": 0,
                "grid": {"programs": 1},
                "stores": [{"index": 0, "role": "Output"}],
                "abi": {
                    "Input": {"kind": "block", "name": "arg0"},
                    "Output": {"kind": "block", "name": "arg1"},
                },
                "bindings": {},
            }
        ]
    for name, computation, mask, delta, reason in (
        ("compute_mismatch", "%increment", "%mask", 0, "COMPUTE_MISMATCH"),
        ("mask_mismatch", "%x", "%badmask", 0, "MASK_MISMATCH"),
        ("address_mismatch", "%x", "%mask", 1, "ADDRESS_MISMATCH"),
    ):
        case = ROOT / "examples" / name
        case.mkdir(parents=True, exist_ok=True)
        pair = copy.deepcopy(simple)
        pair["metadata"]["pair_id"] = name
        (case / "lhs.ttir").write_text(kernel("lhs"))
        (case / "rhs.ttir").write_text(kernel("rhs", computation, mask, delta))
        contract(case, pair, {"status": "DISPROVED", "reason": reason})
    case = ROOT / "examples/multilaunch"
    case.mkdir(parents=True, exist_ok=True)
    pair = copy.deepcopy(simple)
    pair["metadata"]["pair_id"] = "fused_vs_two_launches"
    pair["predicates"]["roles"]["Internal0"] = {"kind": "buffer", "element": "abstract_float"}
    pair["predicates"]["disjoint"] = [["Input", "Internal0", "Output"]]
    (case / "lhs.ttir").write_text(kernel("lhs", "%fused"))
    first = copy.deepcopy(pair["programs"]["rhs"][0])
    first["file"], first["function"] = "scale.ttir", "scale"
    first["stores"] = [{"index": 0, "role": "Internal0"}]
    first["abi"]["Internal0"] = first["abi"].pop("Output")
    second = copy.deepcopy(pair["programs"]["rhs"][0])
    second["id"], second["step"] = "rhs.1", 1
    second["file"], second["function"] = "increment.ttir", "increment"
    second["abi"]["Internal0"] = second["abi"].pop("Input")
    pair["programs"]["rhs"] = [first, second]
    (case / "scale.ttir").write_text(kernel("scale", "%double"))
    (case / "increment.ttir").write_text(kernel("increment", "%increment"))
    contract(case, pair, {"status": "PROVED"})
    case = ROOT / "examples/unsupported"
    case.mkdir(parents=True, exist_ok=True)
    pair = copy.deepcopy(simple)
    pair["metadata"]["pair_id"] = "unsupported_loop"
    for side in ("lhs", "rhs"):
        text = kernel(side).replace(
            "    tt.return",
            """    %zero = arith.constant 0 : index
    %upper = arith.constant 1 : index
    scf.for %i = %zero to %upper step %upper {
      scf.yield
    }
    tt.return""",
        )
        (case / f"{side}.ttir").write_text(text)
    contract(case, pair, {"status": "UNKNOWN", "reason": "TTIR_REGION_UNSUPPORTED"})


if __name__ == "__main__":
    main()
