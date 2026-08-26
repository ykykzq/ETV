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


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape6-dtype6-cuda-0.001-0.001-higher-False-False--226adff38b/rhs/reference-000-000/torchinductor-cache/gm/cgmime6n2uv2cvqgf44ktiqbwi5xufiirqnslg25pmfdxazklwwk.py
# Topologically Sorted Source Nodes: [sort_default], Original ATen: [aten.sort]
# Source node to ATen node mapping:
#   sort_default => sort
# Graph fragment:
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%permute,), kwargs = {})
triton_per_fused_sort_0 = async_compile.triton('triton_per_fused_sort_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sort_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2592, 'r0_': 3888}}
)
@triton.jit
def triton_per_fused_sort_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 108
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_2 + 24*x1), r0_mask & xmask, other=0.0)
    tmp1 = r0_2
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, rnumel, 1, stable=False, descending=False)
    tl.store(out_ptr0 + (r0_2 + 6*x3), tmp5, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape6-dtype6-cuda-0.001-0.001-higher-False-False--226adff38b/rhs/reference-000-000/torchinductor-cache/7g/c7gpyvec5ll5vwb43hifhe3vtgpvablprbp4erqj4eklf6i7ny7l.py
# Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, ceil_, _to_copy, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten.ceil, aten._to_copy, aten.gather]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   any_1 => any_1
#   ceil_ => ceil
#   gather => gather
#   isnan => isnan
#   masked_fill => full_default, where
# Graph fragment:
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%view,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%isnan, -1, True), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 5.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%any_1, %full_default, %expand), kwargs = {})
#   %ceil : [num_users=1] = call_function[target=torch.ops.aten.ceil.default](args = (%where,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ceil, torch.int64), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view, -1, %convert_element_type), kwargs = {})
triton_poi_fused__to_copy_any_ceil_gather_isnan_masked_fill_1 = async_compile.triton('triton_poi_fused__to_copy_any_ceil_gather_isnan_masked_fill_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_any_ceil_gather_isnan_masked_fill_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_any_ceil_gather_isnan_masked_fill_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 108
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 6*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 6*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 6*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (4 + 6*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (5 + 6*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = (tmp2 != 0)
    tmp5 = libdevice.isnan(tmp4).to(tl.int1)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = (tmp6 != 0)
    tmp8 = tmp3 | tmp7
    tmp10 = libdevice.isnan(tmp9).to(tl.int1)
    tmp11 = tmp10.to(tl.int64)
    tmp12 = (tmp11 != 0)
    tmp13 = tmp8 | tmp12
    tmp15 = libdevice.isnan(tmp14).to(tl.int1)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = (tmp16 != 0)
    tmp18 = tmp13 | tmp17
    tmp20 = libdevice.isnan(tmp19).to(tl.int1)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = (tmp21 != 0)
    tmp23 = tmp18 | tmp22
    tmp25 = libdevice.isnan(tmp24).to(tl.int1)
    tmp26 = tmp25.to(tl.int64)
    tmp27 = (tmp26 != 0)
    tmp28 = tmp23 | tmp27
    tmp29 = 5.0
    tmp30 = 3.9698878917731117
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = libdevice.ceil(tmp31)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.full([XBLOCK], 6, tl.int32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp33 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp33)
    tl.device_assert(((0 <= tmp37) & (tmp37 < 6)) | ~(xmask), "index out of bounds: 0 <= tmp37 < 6")
    tmp39 = tl.load(in_ptr0 + (tmp37 + 6*x0), xmask, eviction_policy='evict_last')
    tl.store(in_out_ptr0 + (x0), tmp39, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 9, 6, 4), (216, 24, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 9, 1, 4, 6), (216, 24, 24, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sort_default], Original ATen: [aten.sort]
        stream0 = get_raw_stream(0)
        triton_per_fused_sort_0.run(arg0_1, buf0, 108, 6, stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((3, 9, 4, 1), (36, 4, 1, 108), torch.float32)
        buf3 = reinterpret_tensor(buf2, (3, 9, 4, 1), (36, 4, 1, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, ceil_, _to_copy, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten.ceil, aten._to_copy, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_any_ceil_gather_isnan_masked_fill_1.run(buf3, buf0, 108, stream=stream0)
        del buf0
    return (reinterpret_tensor(buf3, (3, 9, 4), (36, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 9, 6, 4), (216, 24, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
