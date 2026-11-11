"""Capture facts without installing torch or reading any tensor values."""

from fractions import Fraction
import json
import struct
from types import SimpleNamespace

import pytest

from etv_bench.capture import Capture
from etv_bench.host_capture import HostOperationRecorder, instrument_compiled_callable, typed_scalar


class Storage:
    def __init__(self, base, size=256):
        self.base, self.size = base, size

    def data_ptr(self):
        return self.base

    def nbytes(self):
        return self.size


class Tensor:
    def __init__(self, storage, shape=(2,), stride=(1,), offset=0, dtype="torch.float32"):
        self.storage, self.shape = storage, shape
        self.strides, self.offset, self.dtype = stride, offset, dtype
        self.device = "cuda:0"

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return self.storage.data_ptr() + self.offset * self.element_size()

    def element_size(self):
        return 4

    def stride(self):
        return self.strides

    def numel(self):
        import math

        return math.prod(self.shape)

    def item(self):
        raise AssertionError("capture must never read tensor contents")

    def tolist(self):
        raise AssertionError("capture must never read tensor contents")


class Operation:
    def __init__(self, name, output, *, writes=False):
        self.name, self.output = name, output
        self._schema = SimpleNamespace(
            arguments=[SimpleNamespace(alias_info=SimpleNamespace(is_write=writes))]
        )

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.output


@pytest.fixture
def recorder(tmp_path):
    capture = Capture(tmp_path, capture_host_operations=True)
    capture.current_side = "lhs"
    return HostOperationRecorder(capture)


def test_exact_scalar_rounding_and_nonfinite_json():
    scalar = typed_scalar(0.1, "torch.float32")
    rounded = struct.unpack("f", struct.pack("f", 0.1))[0]
    assert scalar["value"] == str(Fraction(rounded))
    assert scalar["source"]["value"] == str(Fraction(0.1))
    assert typed_scalar(1.00390625, "torch.bfloat16")["value"] == "1"
    assert typed_scalar(1.01171875, "torch.bfloat16")["value"] == "65/64"
    assert typed_scalar(1 + 2**-11 + 2**-25, "torch.float16")["value"] == "1"
    assert typed_scalar(-0.0, "torch.float32")["negative_zero"]
    nonfinite = typed_scalar(float("-inf"), "torch.float32")
    assert nonfinite["value"] == "-inf" and nonfinite["encoding"] == "nonfinite"
    json.dumps(nonfinite, allow_nan=False)
    with pytest.raises(ValueError, match="not a Python scalar"):
        typed_scalar(Tensor(Storage(1000)), "torch.float32")
    with pytest.raises(ValueError, match="out-of-range"):
        typed_scalar(-1, "torch.uint8")


def test_quantile_literal_constructor_retains_provenance(recorder):
    output = Tensor(Storage(1000), (), ())
    recorder.constructor(lambda *args, **kwargs: output, (0.25,), {"device": "cuda"})
    event = recorder.capture.host_operations[0]
    assert event["kind"] == "constant" and event["operator"] == "torch.tensor"
    assert event["inputs"] == [] and event["attributes"]["value"]["value"] == "1/4"
    assert event["output"]["shape"] == []
    assert not recorder.capture.diagnostics


def test_unknown_lift_fresh_is_not_a_literal(recorder):
    tensor = Tensor(Storage(1000))
    recorder.dispatch(Operation("aten.lift_fresh.default", tensor), (tensor,), {})
    assert recorder.capture.host_operations[0]["kind"] == "unsupported"
    assert recorder.capture.diagnostics


def test_broadcast_clone_records_zero_stride_source_and_fresh_destination(recorder):
    original = Tensor(Storage(1000), (2, 1), (1, 1))
    view = Tensor(original.storage, (2, 3), (1, 0))
    output = Tensor(Storage(2000), (2, 3), (3, 1))
    recorder.dispatch(Operation("aten.expand.default", view), (original, [2, 3]), {})
    recorder.dispatch(Operation("aten.clone.default", output), (view,), {})
    event = recorder.capture.host_operations[0]
    assert len(recorder.capture.host_operations) == 1
    assert event["inputs"][0]["stride"] == [1, 0]
    assert event["output"]["stride"] == [3, 1]
    assert event["inputs"][0]["storage_id"] != event["output"]["storage_id"]


def test_logsumexp_copy_records_source_and_strided_output(recorder):
    source = Tensor(Storage(1000), (2, 3), (3, 1))
    out = Tensor(Storage(2000), (2, 3), (1, 2), offset=2)
    recorder.dispatch(Operation("aten.copy_.default", out, writes=True), (out, source), {})
    recorder.capture.outputs("lhs", {"Output": out})
    event = recorder.capture.host_operations[0]
    assert event["inputs"] == [recorder.capture.argument(source)]
    assert event["output"]["offset"] == 2
    assert recorder.capture.output_views["lhs"]["Output"]["stride"] == [1, 2]


def test_padding_minus_infinity_is_not_replaced(recorder):
    source, output = Tensor(Storage(1000)), Tensor(Storage(2000), (4,))
    recorder.dispatch(
        Operation("aten.constant_pad_nd.default", output), (source, [1, 1], float("-inf")), {}
    )
    event = recorder.capture.host_operations[0]
    assert event["kind"] == "pad"
    assert event["attributes"]["value"]["value"] == "-inf"
    assert event["attributes"]["padding"] == [1, 1]
    json.dumps(event, allow_nan=False)


def test_mean_division_and_launches_share_one_order(recorder):
    source, output = Tensor(Storage(1000)), Tensor(Storage(2000))
    recorder.dispatch(Operation("aten.clone.default", source), (output,), {})
    capture = recorder.capture
    kernel = SimpleNamespace(
        name="sum", asm={"ttir": "module {}"}, src=SimpleNamespace(signature={}, constants={})
    )
    frame = SimpleNamespace(
        f_globals={"__name__": "triton.runtime.jit"},
        f_code=SimpleNamespace(co_name="run"),
        f_back=None,
        f_locals={"bound_args": {}, "grid_0": 1, "grid_1": 1, "grid_2": 1},
    )
    capture._profile(frame, "return", kernel)
    recorder.dispatch(Operation("aten.div.Tensor", output), (source, 64), {})
    assert [event["step"] for event in capture.host_operations] == [0, 2]
    assert capture.launches[0]["step"] == 1
    assert capture.host_operations[1]["attributes"]["divisor"]["value"] == "64"
    capture.current_side = "rhs"
    assert capture.next_step() == 0


def test_scalar_division_provenance_survives_tensor_wrapping(recorder):
    source, divisor, output = Tensor(Storage(1000)), Tensor(Storage(2000)), Tensor(Storage(3000))

    def wrapped_divide(tensor, scalar):
        return recorder.dispatch(Operation("aten.div.Tensor", output), (tensor, divisor), {})

    recorder.scalar_div(wrapped_divide, (source, 8), {})
    assert recorder.capture.host_operations[0]["attributes"]["divisor"]["value"] == "8"
    recorder.dispatch(Operation("aten.div.Tensor", output), (source, divisor), {})
    assert recorder.capture.host_operations[1]["kind"] == "unsupported"


def test_fill_records_known_literal_but_unknown_mutation_is_diagnostic(recorder):
    output = Tensor(Storage(1000))
    recorder.dispatch(Operation("aten.fill_.Scalar", output, writes=True), (output, 0), {})
    assert recorder.capture.host_operations[0]["inputs"] == []
    recorder.dispatch(Operation("aten.add_.Tensor", output, writes=True), (output, output), {})
    event = recorder.capture.host_operations[1]
    assert event["kind"] == "unsupported" and event["attributes"]["writes"]
    assert "aten.add_.Tensor" in recorder.capture.diagnostics[0]


def test_random_state_is_not_assumed_paired(recorder):
    output = Tensor(Storage(1000), dtype="torch.int64")
    recorder.dispatch(Operation("aten.randint.low", output), (0, 100, [1]), {})
    event = recorder.capture.host_operations[0]
    assert event["kind"] == "unsupported"
    assert "unpaired" in event["attributes"]["reason"]


def test_optional_extension_and_output_views_are_written(tmp_path):
    capture = Capture(tmp_path, capture_host_operations=True)
    tensor = Tensor(Storage(1000), (2, 3), (1, 2))
    capture.outputs("lhs", {"Output": tensor})
    data = json.loads(capture.write({"test_id": "example"}).read_text())
    assert data["format"] == "etv-runtime-capture-v1"
    assert data["host_capture_format"] == "etv-host-operations-v1"
    assert data["host_operations"] == []
    assert data["output_views"]["lhs"]["Output"]["stride"] == [1, 2]
    legacy = Capture(tmp_path / "legacy")
    data = json.loads(legacy.write({}).read_text())
    assert "host_operations" not in data


def test_compiled_wrapper_keeps_boxed_abi_without_dynamo_bypass(tmp_path):
    capture = Capture(tmp_path, capture_host_operations=True)

    def compiled(value):
        return value

    compiled._torchdynamo_orig_callable = lambda value: value
    compiled._boxed_call = True
    wrapped = instrument_compiled_callable(capture, compiled)
    assert not hasattr(wrapped, "_torchdynamo_orig_callable")
    assert wrapped._boxed_call
    assert wrapped(7) == 7
