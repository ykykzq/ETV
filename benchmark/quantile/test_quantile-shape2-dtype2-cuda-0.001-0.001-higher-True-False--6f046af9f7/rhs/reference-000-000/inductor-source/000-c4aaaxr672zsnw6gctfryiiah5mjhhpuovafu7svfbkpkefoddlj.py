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


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape2-dtype2-cuda-0.001-0.001-higher-True-False--6f046af9f7/rhs/reference-000-000/torchinductor-cache/xx/cxx22g3qctkjhrfyn2ke2q33lrejtzfoatrdhsgfpoxa6dmfcnvf.py
# Topologically Sorted Source Nodes: [sort_default, isnan, any_1], Original ATen: [aten.sort, aten.isnan, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   isnan => isnan
#   sort_default => sort
# Graph fragment:
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%permute,), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%getitem,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%isnan, -1, True), kwargs = {})
triton_per_fused_any_isnan_sort_0 = async_compile.triton('triton_per_fused_any_isnan_sort_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_isnan_sort_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4002, 'r0_': 7912}}
)
@triton.jit
def triton_per_fused_any_isnan_sort_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 23
    r0_numel = 43
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 23*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = r0_1
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, rnumel, 1, stable=False, descending=False)
    tmp7 = libdevice.isnan(tmp5).to(tl.int1)
    tmp8 = tmp7.to(tl.int64)
    tmp9 = (tmp8 != 0)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, False)
    tmp13 = triton_helpers.any(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (r0_1 + 43*x0), tmp5, r0_mask & xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape2-dtype2-cuda-0.001-0.001-higher-True-False--6f046af9f7/rhs/reference-000-000/torchinductor-cache/lz/clzevbavfhzpdpvfjreixlpuwgxkwvsnvuajxt4fdbotjvwd5anc.py
# Topologically Sorted Source Nodes: [masked_fill, ceil_, _to_copy, gather], Original ATen: [aten.masked_fill, aten.ceil, aten._to_copy, aten.gather]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   ceil_ => ceil
#   gather => gather
#   masked_fill => full_default, where
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 42.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%any_1, %full_default, %expand), kwargs = {})
#   %ceil : [num_users=1] = call_function[target=torch.ops.aten.ceil.default](args = (%where,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ceil, torch.int64), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%getitem, -1, %convert_element_type), kwargs = {})
triton_poi_fused__to_copy_ceil_gather_masked_fill_1 = async_compile.triton('triton_poi_fused__to_copy_ceil_gather_masked_fill_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ceil_gather_masked_fill_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ceil_gather_masked_fill_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp1 = 42.0
    tmp2 = 15.375914272707018
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = libdevice.ceil(tmp3)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.full([XBLOCK], 43, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 43)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 43")
    tmp11 = tl.load(in_ptr1 + (tmp9 + 43*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (43, 23), (23, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 23, 43), (989, 43, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 23, 1), (23, 1, 23), torch.bool)
        # Topologically Sorted Source Nodes: [sort_default, isnan, any_1], Original ATen: [aten.sort, aten.isnan, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_isnan_sort_0.run(arg0_1, buf0, buf2, 23, 43, stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((1, 23, 1), (23, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [masked_fill, ceil_, _to_copy, gather], Original ATen: [aten.masked_fill, aten.ceil, aten._to_copy, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_ceil_gather_masked_fill_1.run(buf2, buf0, buf3, 23, stream=stream0)
        del buf0
        del buf2
    return (reinterpret_tensor(buf3, (1, 23), (23, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((43, 23), (23, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
