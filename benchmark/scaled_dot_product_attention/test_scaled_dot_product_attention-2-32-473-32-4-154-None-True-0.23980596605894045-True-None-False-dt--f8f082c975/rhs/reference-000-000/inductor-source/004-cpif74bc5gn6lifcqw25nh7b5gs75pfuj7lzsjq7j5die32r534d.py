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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-473-32-4-154-None-True-0.23980596605894045-True-None-False-dt--f8f082c975/rhs/reference-000-000/torchinductor-cache/q6/cq6tdrfm3eygmcpjjexsndiq3fk7ashjdfawb5qp2qvbhhi7csvc.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg0_1, 0.48969987345203636), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11624448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 968704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-473-32-4-154-None-True-0.23980596605894045-True-None-False-dt--f8f082c975/rhs/reference-000-000/torchinductor-cache/7z/c7zpv42fldym6liv4dcluxj2z2363vo7iycsofyoyanmqybrpzky.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute, 0.48969987345203636), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1261568, 'x': 2523136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 154
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 32)
    y1 = ((yindex // 32) % 32)
    y2 = yindex // 1024
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x3 + 4928*(y1 // 8) + 19712*y2), xmask, eviction_policy='evict_last')
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3 + 154*y4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-473-32-4-154-None-True-0.23980596605894045-True-None-False-dt--f8f082c975/rhs/reference-000-000/torchinductor-cache/gx/cgx3tva5siasaqsf26vowdqajn4ypwfvkjrnjvty674ezhihdr7m.py
# Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, eq, logical_not
#   add => add
#   ones => full_default
#   scalar_tensor => full_default_1
#   scalar_tensor_1 => full_default_2
#   tril => le, logical_and, sub
#   where => where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([473, 154], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2 = async_compile.triton('triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 544896, 'r0_': 18647552}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 30272
    r0_numel = 154
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 473)
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 15136
    x6 = (xindex % 15136)
    _tmp18_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp18_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 154*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = r0_3 + ((-1)*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 <= tmp2
        tmp4 = tl.full([1, 1], True, tl.int1)
        tmp5 = tmp3 & tmp4
        tmp6 = 0.0
        tmp7 = float("-inf")
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp0 + tmp8
        tmp10 = tmp9 == tmp7
        tmp11 = tmp10 == 0
        tmp12 = tmp11.to(tl.int64)
        tmp13 = (tmp12 != 0)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 | tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp17 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])

        _tmp18_max_next, _tmp18_sum_next = triton_helpers.online_softmax_combine(
            _tmp18_max, _tmp18_sum, tmp17, False
        )

        _tmp18_max = tl.where(r0_mask & xmask, _tmp18_max_next, _tmp18_max)
        _tmp18_sum = tl.where(r0_mask & xmask, _tmp18_sum_next, _tmp18_sum)
    tmp15 = triton_helpers.any(_tmp15.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp20, tmp21 = triton_helpers.online_softmax_reduce(
        _tmp18_max, _tmp18_sum, 1, False)
    tmp20 = tmp20[:, None]
    tmp21 = tmp21[:, None]
    tmp18 = tmp20
    tmp19 = tmp21
    tl.store(out_ptr0 + (x6 + 15232*x2), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp18, xmask)
    tl.store(out_ptr2 + (x5), tmp19, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-473-32-4-154-None-True-0.23980596605894045-True-None-False-dt--f8f082c975/rhs/reference-000-000/torchinductor-cache/bi/cbivxyfifmwrwwmfpwtwd6mg6slzj62t5ix7vrjv5zbqkjrdgqhx.py
# Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   bmm_1 => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_5, [0, 2, 0, 3, 0, 0]), kwargs = {})
triton_poi_fused_bmm_3 = async_compile.triton('triton_poi_fused_bmm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 57028608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4752384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 156) % 476)
    x0 = (xindex % 156)
    x2 = xindex // 74256
    x3 = (xindex % 74256)
    tmp0 = x1
    tmp1 = tl.full([1], 473, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 154, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (x1 + 473*((x2 % 32)) + 15232*(x2 // 32)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tmp7 == 0
    tmp9 = tl.load(in_ptr1 + (x0 + 154*x1 + 72842*x2), tmp6 & xmask, other=0.0)
    tmp10 = x0 + ((-1)*x1)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 <= tmp11
    tmp13 = tl.full([1], True, tl.int1)
    tmp14 = tmp12 & tmp13
    tmp15 = 0.0
    tmp16 = float("-inf")
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tmp9 + tmp17
    tmp19 = tl.load(in_ptr2 + (x1 + 473*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tl.load(in_ptr3 + (x1 + 473*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = (tmp21 / tmp22)
    tmp24 = tl.where(tmp8, tmp15, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tl.store(out_ptr0 + (x3 + 74272*x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-473-32-4-154-None-True-0.23980596605894045-True-None-False-dt--f8f082c975/rhs/reference-000-000/torchinductor-cache/b6/cb6coqylxhidrc2dfh5hlxr6uyu2uglqc4r7mo5ggua5k7dihm7f.py
# Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   bmm_1 => constant_pad_nd_default_1
# Graph fragment:
#   %constant_pad_nd_default_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_6, [0, 0, 0, 2, 0, 0]), kwargs = {})
triton_poi_fused_bmm_4 = async_compile.triton('triton_poi_fused_bmm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3833856}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 319488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 156)
    x2 = xindex // 4992
    x3 = (xindex % 4992)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 154, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + 4928*(((x2 % 32)) // 8) + 19712*(x2 // 32)), tmp2, other=0.0)
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 32, 473, 32), (484352, 15136, 32, 1))
    assert_size_stride(arg1_1, (2, 4, 154, 32), (19712, 4928, 32, 1))
    assert_size_stride(arg2_1, (2, 4, 154, 32), (19712, 4928, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 32, 473, 32), (484352, 15136, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 968704, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((2, 32, 32, 154), (157696, 4928, 154, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 2048, 154, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((64, 473, 154), (72842, 154, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (64, 473, 32), (15136, 32, 1), 0), reinterpret_tensor(buf1, (64, 32, 154), (4928, 154, 1), 0), out=buf2)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((2, 32, 473, 1), (15232, 473, 1, 30464), torch.bool)
        buf4 = empty_strided_cuda((2, 32, 473, 1), (15136, 473, 1, 30272), torch.float32)
        buf5 = empty_strided_cuda((2, 32, 473, 1), (15136, 473, 1, 30272), torch.float32)
        # Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2.run(buf2, buf3, buf4, buf5, 30272, 154, stream=stream0)
        buf6 = empty_strided_cuda((64, 476, 156), (74272, 156, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_3.run(buf3, buf2, buf4, buf5, buf6, 4752384, stream=stream0)
        del buf2
        del buf3
        del buf4
        del buf5
        buf7 = empty_strided_cuda((64, 156, 32), (4992, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_4.run(arg2_1, buf7, 319488, stream=stream0)
        del arg2_1
        buf8 = empty_strided_cuda((64, 476, 32), (15232, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf6, buf7, out=buf8)
        del buf6
        del buf7
    return (reinterpret_tensor(buf8, (2, 32, 473, 32), (487424, 15232, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 32, 473, 32), (484352, 15136, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 4, 154, 32), (19712, 4928, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 4, 154, 32), (19712, 4928, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
