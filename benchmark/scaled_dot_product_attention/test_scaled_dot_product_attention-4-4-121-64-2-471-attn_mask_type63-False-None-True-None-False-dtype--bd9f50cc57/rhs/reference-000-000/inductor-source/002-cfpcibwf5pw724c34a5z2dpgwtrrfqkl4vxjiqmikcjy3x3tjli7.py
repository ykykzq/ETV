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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/pz/cpz65hthzifibexcuq2ujc2gnzqm4e2gy5v24qsmbdi3xb3dmytm.py
# Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   mul => mul
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.3535533905932738), kwargs = {})
triton_poi_fused__to_copy_mul_0 = async_compile.triton('triton_poi_fused__to_copy_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1239040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 123904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3535533905932738
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/cx/ccxsgck57uvnwjvokrekdpri52anleudhs4xrm757sgppq5roozu.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute, 0.3535533905932738), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 964608, 'x': 3858432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 471
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 4)
    y2 = yindex // 256
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x3 + 30144*(y1 // 2) + 60288*y2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3535533905932738
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (x3 + 471*y4), tmp3, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/wu/cwuq6vgpjzrqfefvfix3el35dvnzihf4mw77zcgrjfl22svoa6yd.py
# Topologically Sorted Source Nodes: [add, _safe_softmax], Original ATen: [aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, eq, logical_not
#   add => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %arg3_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
triton_per_fused__safe_softmax_add_2 = async_compile.triton('triton_per_fused__safe_softmax_add_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34848, 'r0_': 3761406}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 1936
    XBLOCK: tl.constexpr = 1
    r0_numel = 471
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 121)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 471*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 471*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = float("-inf")
    tmp5 = tmp3 == tmp4
    tmp6 = tmp5 == 0
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, False)
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp15 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, float("-inf"))
    tmp18 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp17, 0))
    tmp19 = tmp13 - tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp18, None)
    tl.store(out_ptr2 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/ox/coxi7mnvg7itp3c7xeqj7qrvifdwo4h3l3bwpspnunwaoigi5jnp.py
# Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   bmm_1 => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_5, [0, 1, 0, 3, 0, 0]), kwargs = {})
triton_poi_fused_bmm_3 = async_compile.triton('triton_poi_fused_bmm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11354432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 936448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 472) % 124)
    x0 = (xindex % 472)
    x2 = xindex // 58528
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 121, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 471, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (x1 + 121*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tmp7 == 0
    tmp9 = tl.load(in_ptr1 + (x0 + 471*x1 + 56991*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + 471*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.load(in_ptr3 + (x1 + 121*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.load(in_ptr4 + (x1 + 121*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = (tmp15 / tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp8, tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/jt/cjtxryoozuopwvuftsgnsd3onz5y6gcyduxplnppoqcduxbknw3t.py
# Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   bmm_1 => constant_pad_nd_default_1
# Graph fragment:
#   %constant_pad_nd_default_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_6, [0, 0, 0, 1, 0, 0]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4833280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 483328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 472)
    x2 = xindex // 30208
    x3 = (xindex % 30208)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 471, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + 30144*(((x2 % 4)) // 2) + 60288*(x2 // 4)), tmp2, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tl.store(out_ptr0 + (x4), tmp6, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-4-121-64-2-471-attn_mask_type63-False-None-True-None-False-dtype--bd9f50cc57/rhs/reference-000-000/torchinductor-cache/46/c467tffmi236o5ucx4jqewnpxnz7zkgxciutb5rsrmoo6jyhehmt.py
# Topologically Sorted Source Nodes: [_to_copy_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _to_copy_4 => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_7, torch.float16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 991232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 123904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7744)
    x1 = xindex // 7744
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 7936*x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 121, 64), (30976, 7744, 64, 1))
    assert_size_stride(arg1_1, (4, 2, 471, 64), (60288, 30144, 64, 1))
    assert_size_stride(arg2_1, (4, 2, 471, 64), (60288, 30144, 64, 1))
    assert_size_stride(arg3_1, (121, 471), (471, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 121, 64), (30976, 7744, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_0.run(arg0_1, buf0, 123904, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4, 64, 471), (120576, 30144, 471, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 1024, 471, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((16, 121, 471), (56991, 471, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (16, 121, 64), (7744, 64, 1), 0), reinterpret_tensor(buf1, (16, 64, 471), (30144, 471, 1), 0), out=buf2)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((4, 4, 121, 1), (484, 121, 1, 2048), torch.bool)
        buf4 = empty_strided_cuda((4, 4, 121, 1), (484, 121, 1, 1952), torch.float32)
        buf5 = empty_strided_cuda((4, 4, 121, 1), (484, 121, 1, 1952), torch.float32)
        # Topologically Sorted Source Nodes: [add, _safe_softmax], Original ATen: [aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_add_2.run(buf2, arg3_1, buf3, buf4, buf5, 1936, 471, stream=stream0)
        buf6 = empty_strided_cuda((16, 124, 472), (58528, 472, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_3.run(buf3, buf2, arg3_1, buf4, buf5, buf6, 936448, stream=stream0)
        del arg3_1
        del buf2
        del buf3
        del buf4
        del buf5
        buf7 = empty_strided_cuda((16, 472, 64), (30208, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_4.run(arg2_1, buf7, 483328, stream=stream0)
        del arg2_1
        buf8 = empty_strided_cuda((16, 124, 64), (7936, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf6, buf7, out=buf8)
        del buf6
        del buf7
        buf9 = empty_strided_cuda((4, 4, 121, 64), (30976, 7744, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_to_copy_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf8, buf9, 123904, stream=stream0)
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 121, 64), (30976, 7744, 64, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, 2, 471, 64), (60288, 30144, 64, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4, 2, 471, 64), (60288, 30144, 64, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((121, 471), (471, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
