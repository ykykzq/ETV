# AOT ID: ['1_inference']
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


# kernel path: /data3/yelv.lh/ETV/benchmark/avg_pool2d/test_avg_pool2d-2-3-112-112-kernel_size1-1-1-False-dtype0-1e-05-1e-05-cuda--a989eedb61/rhs/reference-000-001/torchinductor-cache/kx/ckxk7euz2qqi6kegfdqmjcaaredmqyo52vgzfor5tr7rvvwb7gi4.py
# Topologically Sorted Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg0_1, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_0 = async_compile.triton('triton_poi_fused_avg_pool2d_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3311616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 112) % 112)
    x0 = (xindex % 112)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + x4), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + x4), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + x4), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + x4), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + x4), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((113) * ((113) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (113)))*((113) * ((113) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (113))) + ((-1)*x0*((113) * ((113) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (113)))) + ((-1)*x1*((113) * ((113) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (113)))) + ((113) * ((113) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (113))) + ((113) * ((113) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (113)))
    tmp53 = (tmp51 / tmp52)
    tl.store(out_ptr0 + (x4), tmp53, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 112, 112), (37632, 12544, 112, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 3, 112, 112), (37632, 12544, 112, 1), torch.float32)
        # Topologically Sorted Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0.run(arg0_1, buf0, 75264, stream=stream0)
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
