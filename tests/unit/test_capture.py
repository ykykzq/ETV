from types import SimpleNamespace

from etv_bench.capture import Capture


def frame(module, name, local, parent=None, globals_extra=None):
    return SimpleNamespace(
        f_globals={"__name__": module, **(globals_extra or {})},
        f_code=SimpleNamespace(co_name=name),
        f_locals=local,
        f_back=parent,
    )


def kernel():
    return SimpleNamespace(
        name="copy",
        asm={"ttir": "module {}"},
        src=SimpleNamespace(signature={"n": "i32", "BLOCK": "constexpr"}, constants={"BLOCK": 4}),
    )


def test_runtime_one_is_not_a_constexpr(tmp_path):
    capture = Capture(tmp_path)
    capture.current_side = "lhs"
    call = frame(
        "triton.runtime.jit",
        "run",
        {
            "bound_args": {"n": 1, "BLOCK": 4},
            "warmup": False,
            "grid_0": 1,
            "grid_1": 1,
            "grid_2": 1,
        },
    )
    capture._profile(call, "return", kernel())
    assert not capture.diagnostics
    slots = capture.launches[0]["slots"]
    assert len(slots) == 1 and slots[0]["runtime"] == {"kind": "integer", "value": 1}
    assert capture.launches[0]["sha256"]


def test_autotuning_candidates_are_excluded(tmp_path):
    capture = Capture(tmp_path)
    capture.current_side = "lhs"
    parent = frame("triton.runtime.autotuner", "_bench", {})
    call = frame("triton.runtime.jit", "run", {"warmup": False}, parent)
    capture._profile(call, "return", kernel())
    assert not capture.launches and not capture.diagnostics


def test_selected_inductor_launcher_uses_actual_signature(tmp_path):
    capture = Capture(tmp_path)
    capture.current_side = "rhs"
    call = frame(
        "generated",
        "launcher",
        {"n": 1, "grid_0": 2, "grid_1": 1, "grid_2": 1},
        globals_extra={"bin": kernel()},
    )
    launcher = SimpleNamespace(__code__=call.f_code)
    call.f_back = frame("torch._inductor.runtime.triton_heuristics", "run", {"launcher": launcher})
    capture._profile(call, "return", None)
    assert len(capture.launches) == 1 and capture.launches[0]["grid"] == [2, 1, 1]
    call.f_back.f_locals["benchmark_run"] = True
    capture._profile(call, "return", None)
    assert len(capture.launches) == 1
