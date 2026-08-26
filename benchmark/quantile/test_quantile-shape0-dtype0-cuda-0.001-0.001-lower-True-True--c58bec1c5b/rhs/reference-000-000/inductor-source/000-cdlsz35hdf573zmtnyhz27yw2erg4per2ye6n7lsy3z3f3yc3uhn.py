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


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape0-dtype0-cuda-0.001-0.001-lower-True-True--c58bec1c5b/rhs/reference-000-000/torchinductor-cache/ta/ctal3imqwgpcx25xkofxhgt523hrjw5g4xo62iqowqj4zjnfbjw6.py
# Topologically Sorted Source Nodes: [sort_default, isnan, any_1], Original ATen: [aten.sort, aten.isnan, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   isnan => isnan
#   sort_default => sort
# Graph fragment:
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%arg0_1,), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%view,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%isnan, -1, True), kwargs = {})
triton_per_fused_any_isnan_sort_0 = async_compile.triton('triton_per_fused_any_isnan_sort_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_isnan_sort_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 5412}}
)
@triton.jit
def triton_per_fused_any_isnan_sort_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 451
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = r0_0
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, rnumel, 0, stable=False, descending=False)
    tmp7 = libdevice.isnan(tmp5).to(tl.int1)
    tmp8 = tmp7.to(tl.int64)
    tmp9 = (tmp8 != 0)
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, False)
    tmp13 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp12, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp5, r0_mask)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp13, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape0-dtype0-cuda-0.001-0.001-lower-True-True--c58bec1c5b/rhs/reference-000-000/torchinductor-cache/ro/crowpkdnh7c6qmnpoaxxub3yt4ggt3fqh6yhihcnpejbf56xi327.py
# Topologically Sorted Source Nodes: [masked_fill, floor_, _to_copy, gather], Original ATen: [aten.masked_fill, aten.floor, aten._to_copy, aten.gather]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   floor_ => floor
#   gather => gather
#   masked_fill => full_default, where
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 450.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%any_1, %full_default, %expand), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%where,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view, -1, %convert_element_type), kwargs = {})
triton_poi_fused__to_copy_floor_gather_masked_fill_1 = async_compile.triton('triton_poi_fused__to_copy_floor_gather_masked_fill_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_floor_gather_masked_fill_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_floor_gather_masked_fill_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = 450.0
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp1, tmp4, tmp5)
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.full([XBLOCK], 451, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert((0 <= tmp12) & (tmp12 < 451), "index out of bounds: 0 <= tmp12 < 451")
    tmp14 = tl.load(in_ptr2 + (tmp12), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp14, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (451, ), (1, ))
    assert_size_stride(arg1_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((451, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((1, 1), (1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [sort_default, isnan, any_1], Original ATen: [aten.sort, aten.isnan, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_isnan_sort_0.run(arg0_1, buf0, buf2, 1, 451, stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [masked_fill, floor_, _to_copy, gather], Original ATen: [aten.masked_fill, aten.floor, aten._to_copy, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_floor_gather_masked_fill_1.run(buf2, arg1_1, buf0, buf3, 1, stream=stream0)
        del arg1_1
        del buf0
        del buf2
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((451, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
