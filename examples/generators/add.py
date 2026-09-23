"""CUDA capture driver for a Triton kernel and a TorchInductor reference."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, n, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < n) + tl.load(y + index, index < n)
    tl.store(output + index, value, index < n)


def collect(capture):
    x = torch.randn(128, device="cuda")
    y = torch.randn_like(x)
    output = torch.empty_like(x)
    reference = torch.compile(torch.add, fullgraph=True)
    # Compile and tune before capturing the selected runtime launches.
    add_kernel[(1,)](x, y, output, x.numel(), BLOCK=128)
    reference(x, y)
    torch.cuda.synchronize()
    capture.inputs({"Input": x, "Other": y})
    with capture.side("lhs"):
        add_kernel[(1,)](x, y, output, x.numel(), BLOCK=128)
    capture.outputs("lhs", {"Output": output})
    with capture.side("rhs"):
        expected = reference(x, y)
    capture.outputs("rhs", {"Output": expected})
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected)
