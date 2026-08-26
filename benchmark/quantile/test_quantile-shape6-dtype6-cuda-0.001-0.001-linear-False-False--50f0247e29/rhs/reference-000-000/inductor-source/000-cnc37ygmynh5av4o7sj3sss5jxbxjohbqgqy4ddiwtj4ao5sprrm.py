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


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape6-dtype6-cuda-0.001-0.001-linear-False-False--50f0247e29/rhs/reference-000-000/torchinductor-cache/xe/cxej3emt6ez76cn3vf7k2f3gukpyaj7dod6gxevr42kxluusi4n7.py
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
    size_hints={'x': 256, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sort_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2592, 'r0_': 1944}}
)
@triton.jit
def triton_per_fused_sort_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 216
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 216*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = r0_1
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, rnumel, 1, stable=False, descending=False)
    tl.store(out_ptr0 + (r0_1 + 3*x0), tmp5, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape6-dtype6-cuda-0.001-0.001-linear-False-False--50f0247e29/rhs/reference-000-000/torchinductor-cache/za/czaavhfxhyhrhbgd7s47n7vwomjtv6cd5sda5pbn7k2rad4onmsz.py
# Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, _to_copy, sub, lerp_, ceil_, _to_copy_1, gather_1, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten._to_copy, aten.sub, aten.lerp, aten.ceil, aten.gather]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   _to_copy_1 => convert_element_type_1
#   any_1 => any_1
#   ceil_ => ceil
#   gather => gather
#   gather_1 => gather_1
#   isnan => isnan
#   lerp_ => abs_1, add, ge, mul_1, sub_1, sub_2, where_1, where_2
#   masked_fill => full_default, where
#   sub => sub
# Graph fragment:
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%view,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%isnan, -1, True), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%any_1, %full_default, %expand), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.int64), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %convert_element_type), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %ge : [num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, 1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sub_1, %sub), kwargs = {})
#   %ceil : [num_users=1] = call_function[target=torch.ops.aten.ceil.default](args = (%where,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ceil, torch.int64), kwargs = {})
#   %gather_1 : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%view, -1, %convert_element_type_1), kwargs = {})
#   %gather : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%view, -1, %convert_element_type), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%gather_1, %gather), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %sub_2), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %gather_1, %gather), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %where_2), kwargs = {})
triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1 = async_compile.triton('triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
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
    tmp14 = 2.0
    tmp15 = 1.3159825876209328
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 - tmp18
    tmp20 = tl_math.abs(tmp19)
    tmp21 = 0.5
    tmp22 = tmp20 >= tmp21
    tmp23 = 1.0
    tmp24 = tmp19 - tmp23
    tmp25 = tl.where(tmp22, tmp24, tmp19)
    tmp26 = libdevice.ceil(tmp16)
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tl.full([XBLOCK], 3, tl.int32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 3)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 3")
    tmp33 = tl.load(in_ptr0 + (tmp31 + 3*x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp17 + tmp28
    tmp35 = tmp17 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp17)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 3)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 3")
    tmp38 = tl.load(in_ptr0 + (tmp36 + 3*x0), xmask, eviction_policy='evict_last')
    tmp39 = tmp33 - tmp38
    tmp40 = tmp25 * tmp39
    tmp41 = tl.where(tmp22, tmp33, tmp38)
    tmp42 = tmp40 + tmp41
    tl.store(in_out_ptr0 + (x0), tmp42, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 9, 6, 4), (216, 24, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 9, 6, 4, 3), (648, 72, 12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sort_default], Original ATen: [aten.sort]
        stream0 = get_raw_stream(0)
        triton_per_fused_sort_0.run(arg0_1, buf0, 216, 3, stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((9, 6, 4, 1), (24, 4, 1, 216), torch.float32)
        buf3 = reinterpret_tensor(buf2, (9, 6, 4, 1), (24, 4, 1, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, _to_copy, sub, lerp_, ceil_, _to_copy_1, gather_1, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten._to_copy, aten.sub, aten.lerp, aten.ceil, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1.run(buf3, buf0, 216, stream=stream0)
        del buf0
    return (reinterpret_tensor(buf3, (9, 6, 4), (24, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 9, 6, 4), (216, 24, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
