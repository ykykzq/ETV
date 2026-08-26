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


# kernel path: /data3/yelv.lh/ETV/benchmark/max_pool3d/test_max_pool3d-False-0-True--69a0e5c64d/rhs/reference-000-000/torchinductor-cache/ar/carrvrkgcfdn4jwuzp5bss6avgjmmvmf3vndzy5hlym5lutcamyq.py
# Topologically Sorted Source Nodes: [max_pool3d_with_indices_default], Original ATen: [aten.max_pool3d_with_indices]
# Source node to ATen node mapping:
#   max_pool3d_with_indices_default => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
triton_poi_fused_max_pool3d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool3d_with_indices_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 58240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 14)
    x1 = ((xindex // 14) % 13)
    x2 = xindex // 182
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 15*x1 + 210*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 15*x1 + 210*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (15 + x0 + 15*x1 + 210*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 15*x1 + 210*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (210 + x0 + 15*x1 + 210*x2), xmask)
    tmp9 = tl.load(in_ptr0 + (211 + x0 + 15*x1 + 210*x2), xmask)
    tmp11 = tl.load(in_ptr0 + (225 + x0 + 15*x1 + 210*x2), xmask)
    tmp13 = tl.load(in_ptr0 + (226 + x0 + 15*x1 + 210*x2), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 9, 14, 15), (1890, 1890, 210, 15, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 8, 13, 14), (1456, 1456, 182, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_pool3d_with_indices_default], Original ATen: [aten.max_pool3d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool3d_with_indices_0.run(arg0_1, buf0, 1456, stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 9, 14, 15), (1890, 1890, 210, 15, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
