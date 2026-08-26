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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-2-74-32-2-458-attn_mask_type72-False-0.23980596605894045-True-2---0a2de7c56e/rhs/reference-000-000/torchinductor-cache/np/cnphdjmtsboexo4q45uq5uylbj6whx2vg3cahmjef6uohoxjsbze.py
# Topologically Sorted Source Nodes: [_scaled_dot_product_efficient_attention_default], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   _scaled_dot_product_efficient_attention_default => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%arg3_1, %arg2_1, %arg1_1, %expand, False), kwargs = {scale: 0.23980596605894045})
triton_poi_fused__scaled_dot_product_efficient_attention_0 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 406704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33892
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 458)
    x2 = xindex
    x1 = xindex // 458
    tmp0 = x0
    tmp1 = tl.full([1], 458, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2 & xmask, other=0.0)
    tl.store(out_ptr0 + (x0 + 464*x1), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (74, 458), (458, 1))
    assert_size_stride(arg1_1, (1, 2, 458, 32), (29312, 14656, 32, 1))
    assert_size_stride(arg2_1, (1, 2, 458, 32), (29312, 14656, 32, 1))
    assert_size_stride(arg3_1, (1, 2, 74, 32), (4736, 2368, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 74, 458), (34336, 0, 464, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_scaled_dot_product_efficient_attention_default], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_0.run(arg0_1, buf0, 33892, stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [_scaled_dot_product_efficient_attention_default], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(arg3_1, arg2_1, arg1_1, reinterpret_tensor(buf0, (1, 2, 74, 458), (34336, 0, 464, 1), 0), False, scale=0.23980596605894045)
        del arg1_1
        del arg2_1
        del arg3_1
        del buf0
        buf2 = buf1[0]
        assert_size_stride(buf2, (1, 2, 74, 32), (4736, 32, 64, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf2, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((74, 458), (458, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 2, 458, 32), (29312, 14656, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 2, 458, 32), (29312, 14656, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 2, 74, 32), (4736, 2368, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
