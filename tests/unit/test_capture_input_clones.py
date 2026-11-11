"""Both captured programs start from independent copies of the same input state."""

from contextlib import contextmanager
from copy import deepcopy
import json
import sys
from types import ModuleType

import pytest

from etv_bench import pytest_capture


def test_capture_call_prepares_two_input_sets_and_an_isolated_warmup(tmp_path, monkeypatch):
    """Exercise the capture orchestration even where PyTorch is unavailable."""

    class Tensor:
        def __init__(self, value):
            self.value = value

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.compile = lambda function, **kwargs: function
    torch.cuda = ModuleType("torch.cuda")
    torch.cuda.synchronize = lambda: None
    torch._inductor = ModuleType("torch._inductor")
    torch._inductor.config = ModuleType("torch._inductor.config")
    torch._inductor.config.use_static_cuda_launcher = True
    registry = ModuleType("torch._dynamo.backends.registry")
    registry.lookup_backend = lambda name: None
    for name, module in {
        "torch": torch,
        "torch._inductor": torch._inductor,
        "torch._inductor.config": torch._inductor.config,
        "torch._dynamo": ModuleType("torch._dynamo"),
        "torch._dynamo.backends": ModuleType("torch._dynamo.backends"),
        "torch._dynamo.backends.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    captures = []

    class Capture:
        def __init__(self, directory, **kwargs):
            self.registered = []
            self.current_side = None
            self.launches = []
            self.host_operations = []
            self.diagnostics = []
            captures.append(self)

        def inputs(self, values):
            self.registered.append(values)

        @contextmanager
        def side(self, side, **kwargs):
            self.current_side = side
            yield
            self.launches.append({"side": side})
            self.current_side = None

        def outputs(self, *args):
            pass

        def write(self, source):
            pass

    clones = []

    def clone(arguments):
        result = deepcopy(arguments)
        clones.append((arguments, result))
        return result

    reference_values = []

    def reference(input, alias, running_mean):
        assert input is alias
        reference_values.append((captures[0].current_side, running_mean.value))
        running_mean.value = 100
        return Tensor(input.value)

    def upstream(input, alias, running_mean, out):
        assert out is None and input is alias
        assert running_mean.value == 3
        running_mean.value = 77
        return Tensor(input.value)

    monkeypatch.setattr(pytest_capture, "Capture", Capture)
    monkeypatch.setattr(pytest_capture, "clone_arguments", clone)
    monkeypatch.setattr(pytest_capture, "reference", lambda *args: reference)
    source = Tensor(7)
    arguments = {"input": source, "alias": source, "running_mean": Tensor(3), "out": None}

    pytest_capture.capture_call(
        tmp_path, "tests/test_example.py::test_example", upstream, arguments
    )

    status = json.loads((tmp_path / "case_status.json").read_text())
    assert status["status"] == "captured", status.get("error")
    assert status["capture_configuration"]["input_preparation"] == (
        "independent_equal_initial_input_clones"
    )
    assert len(clones) == 3
    assert clones[0][0] is arguments and clones[1][0] is arguments
    assert clones[0][1] is not clones[1][1]
    assert clones[2][1]["running_mean"] is not clones[1][1]["running_mean"]
    assert reference_values == [(None, 3), ("rhs", 3)]
    assert arguments["running_mean"].value == 3
    lhs, rhs = captures[0].registered
    assert lhs.keys() == rhs.keys() == {"Input_input", "Input_alias", "Input_running_mean"}
    assert lhs["Input_input"] is lhs["Input_alias"]
    assert rhs["Input_input"] is rhs["Input_alias"]
    assert lhs["Input_input"] is not rhs["Input_input"]
    assert lhs["Input_running_mean"] is not rhs["Input_running_mean"]


def test_real_cpu_clones_preserve_views_and_register_equal_roles(tmp_path):
    torch = pytest.importorskip("torch")
    from etv_bench.capture import Capture

    storage = torch.arange(24, dtype=torch.float32)
    arguments = {
        "input": storage.as_strided((2, 3), (6, 2), 1),
        "alias": storage[3:9],
        "running_mean": torch.tensor([2.0, 4.0, 6.0]),
        "out": None,
    }
    lhs = pytest_capture.clone_arguments(arguments)
    rhs = pytest_capture.clone_arguments(arguments)
    warmup = pytest_capture.clone_arguments(rhs)

    for name in ("input", "alias", "running_mean"):
        for cloned in (lhs, rhs, warmup):
            assert torch.equal(cloned[name], arguments[name])
            assert cloned[name].stride() == arguments[name].stride()
            assert cloned[name].storage_offset() == arguments[name].storage_offset()
            assert cloned[name].shape == arguments[name].shape
        assert (
            len(
                {
                    value[name].untyped_storage().data_ptr()
                    for value in (arguments, lhs, rhs, warmup)
                }
            )
            == 4
        )
    for cloned in (lhs, rhs, warmup):
        assert cloned["input"].untyped_storage().data_ptr() == (
            cloned["alias"].untyped_storage().data_ptr()
        )

    lhs["running_mean"].add_(10)
    lhs["alias"].add_(20)
    warmup["running_mean"].zero_()
    warmup["alias"].zero_()
    for name in ("input", "alias", "running_mean"):
        assert torch.equal(rhs[name], arguments[name])

    capture = Capture(tmp_path)
    capture.inputs(pytest_capture.tensor_inputs(lhs))
    capture.inputs(pytest_capture.tensor_inputs(rhs))
    # Storage aliases use one logical role within a side. Both cloned allocations
    # receive that same role; distinct running-mean allocations do as well.
    for role in ("Input_alias", "Input_running_mean"):
        assert list(capture.input_roles.values()).count(role) == 2
    lhs_mean = capture.argument(lhs["running_mean"])["storage_id"]
    rhs_mean = capture.argument(rhs["running_mean"])["storage_id"]
    assert lhs_mean != rhs_mean
    assert capture.input_roles[lhs_mean] == capture.input_roles[rhs_mean] == "Input_running_mean"
