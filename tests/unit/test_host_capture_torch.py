"""Real dispatcher smoke tests; available on the remote torch benchmark runtime."""

import pytest

torch = pytest.importorskip("torch")

from etv_bench.capture import Capture  # noqa: E402
from etv_bench.host_capture import instrument_compiled_callable  # noqa: E402


def test_actual_cpu_dispatch_retains_host_data_relations(tmp_path):
    capture = Capture(tmp_path, capture_host_operations=True)
    source = torch.arange(2, dtype=torch.float32).reshape(2, 1)
    capture.inputs({"Input": source})
    with capture.side("lhs"):
        q = torch.tensor(0.25, dtype=torch.float32)
        expanded = source.expand(2, 3).contiguous()
        divided = expanded.div(2)
        padded = torch.nn.functional.pad(divided, (1, 1), value=float("-inf"))
        output = torch.empty_like(padded)
        output.copy_(padded)
    assert q.item() == 0.25
    assert torch.equal(output[:, 1:-1], source.expand(2, 3) / 2)
    assert not capture.diagnostics
    events = capture.host_operations
    assert [event["kind"] for event in events] == ["constant", "copy", "div_scalar", "pad", "copy"]
    assert events[1]["inputs"][0]["stride"] == [1, 0]
    assert events[3]["attributes"]["value"]["value"] == "-inf"


def test_runtime_instrumentation_preserves_compile_backend_execution(tmp_path):
    capture = Capture(tmp_path, capture_host_operations=True)
    compiled_executions = []

    def backend(graph, example_inputs):
        def execute(*args):
            compiled_executions.append(True)
            return graph.forward(*args)

        return instrument_compiled_callable(capture, execute)

    compiled = torch.compile(lambda x: x.clone().div(4), backend=backend, fullgraph=True)
    source = torch.arange(4, dtype=torch.float32)
    compiled(source)
    assert not capture.host_operations
    with capture.side("rhs", host_mode=False):
        actual = compiled(source)
    assert len(compiled_executions) == 2
    assert torch.equal(actual, source / 4)
    assert not capture.diagnostics
    assert [event["kind"] for event in capture.host_operations] == ["copy", "div_scalar"]


def test_actual_inductor_runtime_does_not_bypass_external_sort_capture(tmp_path):
    from torch._dynamo.backends.registry import lookup_backend

    capture = Capture(tmp_path, capture_host_operations=True)
    inductor = lookup_backend("inductor")

    def backend(graph, example_inputs):
        return instrument_compiled_callable(capture, inductor(graph, example_inputs))

    source = torch.arange(64, 0, -1, dtype=torch.float32)
    capture.inputs({"Input": source})
    compiled = torch.compile(
        lambda x: torch.sort(x, stable=False, dim=0), backend=backend, fullgraph=True
    )
    compiled(source)
    assert not capture.host_operations
    with capture.side("rhs", host_mode=False):
        actual = compiled(source)
    assert torch.equal(actual[0], torch.sort(source)[0])
    events = capture.host_operations
    assert len(events) == 1
    assert events[0]["operator"] == "aten.sort.stable"
    assert events[0]["kind"] == "unsupported"
    assert len(events[0]["inputs"]) == 1 and len(events[0]["outputs"]) == 2
    assert events[0]["inputs"][0]["storage_id"] in capture.input_roles
    assert capture.diagnostics
