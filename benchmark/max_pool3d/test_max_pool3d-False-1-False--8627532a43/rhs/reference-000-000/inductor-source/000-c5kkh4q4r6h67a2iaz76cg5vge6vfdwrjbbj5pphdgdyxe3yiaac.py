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


# kernel path: /data3/yelv.lh/ETV/benchmark/max_pool3d/test_max_pool3d-False-1-False--8627532a43/rhs/reference-000-000/torchinductor-cache/ac/cacgklmwhmoxjdcgbizfumqb6mdc2ipcauptoys2mygdbueunfs7.py
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 120) % 14)
    x1 = ((xindex // 12) % 10)
    x0 = (xindex % 12)
    x3 = xindex // 1680
    x7 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 13, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 9, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = (-1) + x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tl.full([1], 11, tl.int64)
    tmp14 = tmp11 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 & tmp10
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + ((-111) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp17 & xmask, other=float("-inf"))
    tmp19 = x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp13
    tmp22 = tmp20 & tmp21
    tmp23 = tmp16 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-110) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp8
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tmp30 & tmp15
    tmp32 = tl.load(in_ptr0 + ((-100) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp25, tmp32)
    tmp34 = tmp30 & tmp22
    tmp35 = tl.load(in_ptr0 + ((-99) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp34 & xmask, other=float("-inf"))
    tmp36 = triton_helpers.maximum(tmp33, tmp35)
    tmp37 = x2
    tmp38 = tmp37 >= tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tmp38 & tmp39
    tmp41 = tmp40 & tmp10
    tmp42 = tmp41 & tmp15
    tmp43 = tl.load(in_ptr0 + ((-12) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp42 & xmask, other=float("-inf"))
    tmp44 = triton_helpers.maximum(tmp36, tmp43)
    tmp45 = tmp41 & tmp22
    tmp46 = tl.load(in_ptr0 + ((-11) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp45 & xmask, other=float("-inf"))
    tmp47 = triton_helpers.maximum(tmp44, tmp46)
    tmp48 = tmp40 & tmp29
    tmp49 = tmp48 & tmp15
    tmp50 = tl.load(in_ptr0 + ((-1) + x0 + 11*x1 + 99*x2 + 1287*x3), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp47, tmp50)
    tmp52 = tmp48 & tmp22
    tmp53 = tl.load(in_ptr0 + (x0 + 11*x1 + 99*x2 + 1287*x3), tmp52 & xmask, other=float("-inf"))
    tmp54 = triton_helpers.maximum(tmp51, tmp53)
    tl.store(out_ptr0 + (x7), tmp54, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 1, 13, 9, 11), (1287, 1287, 99, 11, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 1, 14, 10, 12), (1680, 1680, 120, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_pool3d_with_indices_default], Original ATen: [aten.max_pool3d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool3d_with_indices_0.run(arg0_1, buf0, 3360, stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 1, 13, 9, 11), (1287, 1287, 99, 11, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
