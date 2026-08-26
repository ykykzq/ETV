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


# kernel path: /data3/yelv.lh/ETV/benchmark/max_pool2d/test_max_pool2d-2-3-112-112-kernel_size1-stride2-1-2-False-dtype0-cuda--faf2593a35/rhs/reference-000-000/torchinductor-cache/sn/csn7jp5zekgzxxq6vj4wn3cigbbs4kdewzcj23kbqv7tawyt4l3z.py
# Topologically Sorted Source Nodes: [max_pool2d_with_indices_default], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d_with_indices_default => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 97680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12210
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 37) % 55)
    x0 = (xindex % 37)
    x2 = xindex // 2035
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 3*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + 3*x0 + 224*x1 + 12544*x2), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 1 + 3*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-111) + 3*x0 + 224*x1 + 12544*x2), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 3 + 3*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-109) + 3*x0 + 224*x1 + 12544*x2), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 1 + 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (111 + 3*x0 + 224*x1 + 12544*x2), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (113 + 3*x0 + 224*x1 + 12544*x2), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (115 + 3*x0 + 224*x1 + 12544*x2), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 3 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (335 + 3*x0 + 224*x1 + 12544*x2), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (337 + 3*x0 + 224*x1 + 12544*x2), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (339 + 3*x0 + 224*x1 + 12544*x2), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 112, 112), (37632, 12544, 112, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 3, 55, 37), (6105, 2035, 37, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_pool2d_with_indices_default], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0.run(arg0_1, buf0, 12210, stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 112, 112), (37632, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
