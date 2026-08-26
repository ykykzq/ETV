# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 2, 495, 32), (31680, 15840, 32, 1))
    assert_size_stride(arg1_1, (2, 2, 495, 32), (31680, 15840, 32, 1))
    assert_size_stride(arg2_1, (2, 2, 455, 32), (29120, 14560, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [_scaled_dot_product_efficient_attention_default], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf0 = torch.ops.aten._scaled_dot_product_efficient_attention.default(arg2_1, arg1_1, arg0_1, None, False, scale=0.23980596605894045)
        del arg0_1
        del arg1_1
        del arg2_1
        buf1 = buf0[0]
        assert_size_stride(buf1, (2, 2, 455, 32), (29120, 32, 64, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf1, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf0
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 2, 495, 32), (31680, 15840, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 2, 495, 32), (31680, 15840, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 2, 455, 32), (29120, 14560, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
