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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-8-268-32-2-445-attn_mask_type54-False-0.23980596605894045-True-1--f5db3b6561/rhs/reference-000-000/torchinductor-cache/nd/cndyx4o7pqzg77pftwsxserowikwkpz2ua4k33ywfkahsmwg4w4j.py
# Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   mul => mul
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.48969987345203636), kwargs = {})
triton_poi_fused__to_copy_mul_0 = async_compile.triton('triton_poi_fused__to_copy_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2058240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.48969987345203636
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-8-268-32-2-445-attn_mask_type54-False-0.23980596605894045-True-1--f5db3b6561/rhs/reference-000-000/torchinductor-cache/m7/cm7qoidgvaxehpbknzbbcqzrnvbmztmnqfxa77cx6ikfxvnxuizq.py
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
    size_hints={'y': 1024, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 683520, 'x': 2734080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 445
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 32)
    y1 = ((yindex // 32) % 8)
    y2 = yindex // 256
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x3 + 14240*(y1 // 4) + 28480*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.48969987345203636
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (x3 + 445*y4), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-8-268-32-2-445-attn_mask_type54-False-0.23980596605894045-True-1--f5db3b6561/rhs/reference-000-000/torchinductor-cache/nu/cnu76nsbxfxr3njttenm7tax6rsco74pui3c45pcccgfq7bfqmo3.py
# Topologically Sorted Source Nodes: [scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, div, eq, full_default_2, logical_not, logical_not_1, where_1
#   add => add
#   scalar_tensor => full_default
#   scalar_tensor_1 => full_default_1
#   where => where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg0_1, %full_default_1, %full_default), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([3, 8, 268, 445], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_2, %div), kwargs = {})
triton_red_fused__safe_softmax_add_scalar_tensor_where_2 = async_compile.triton('triton_red_fused__safe_softmax_add_scalar_tensor_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 34466140}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6432
    r0_numel = 445
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 268)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 2144
    x6 = (xindex % 2144)
    _tmp15_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp15_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 445*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 445*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp2 = 0.0
        tmp3 = float("-inf")
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 + tmp5
        tmp7 = tmp6 == tmp3
        tmp8 = tmp7 == 0
        tmp9 = tmp8.to(tl.int64)
        tmp10 = (tmp9 != 0)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 | tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp15_max_next, _tmp15_sum_next = triton_helpers.online_softmax_combine(
            _tmp15_max, _tmp15_sum, tmp14, False
        )

        _tmp15_max = tl.where(r0_mask & xmask, _tmp15_max_next, _tmp15_max)
        _tmp15_sum = tl.where(r0_mask & xmask, _tmp15_sum_next, _tmp15_sum)
    tmp12 = triton_helpers.any(_tmp12.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp17, tmp18 = triton_helpers.online_softmax_reduce(
        _tmp15_max, _tmp15_sum, 1, False)
    tmp17 = tmp17[:, None]
    tmp18 = tmp18[:, None]
    tmp15 = tmp17
    tmp16 = tmp18
    x4 = xindex // 268
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp20 = tl.load(in_ptr0 + (r0_3 + 445*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr1 + (r0_3 + 445*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp19 = tmp12 == 0
        tmp22 = 0.0
        tmp23 = float("-inf")
        tmp24 = tl.where(tmp21, tmp22, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp20 + tmp25
        tmp27 = tmp26 - tmp15
        tmp28 = tl_math.exp(tmp27)
        tmp29 = (tmp28 / tmp16)
        tmp30 = tl.where(tmp19, tmp22, tmp29)
        tl.store(out_ptr3 + (r0_3 + 445*x0 + 119264*x4), tmp30, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-8-268-32-2-445-attn_mask_type54-False-0.23980596605894045-True-1--f5db3b6561/rhs/reference-000-000/torchinductor-cache/3f/c3fox4gd67ez7oqoq4s2tjkqzhwr2dkdfoditesupitqhlz6jmxd.py
# Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   clone_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2904960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 341760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 14240)
    x2 = xindex // 56960
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 14240*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-8-268-32-2-445-attn_mask_type54-False-0.23980596605894045-True-1--f5db3b6561/rhs/reference-000-000/torchinductor-cache/ae/caen6rekne7q5ydniw2vh6diujcsqf4owp2lu5m6p6hk3qor2yzz.py
# Topologically Sorted Source Nodes: [_to_copy_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _to_copy_4 => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_7, torch.float16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1646592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (268, 445), (445, 1))
    assert_size_stride(arg1_1, (3, 8, 268, 32), (68608, 8576, 32, 1))
    assert_size_stride(arg2_1, (3, 2, 445, 32), (28480, 14240, 32, 1))
    assert_size_stride(arg3_1, (3, 2, 445, 32), (28480, 14240, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 8, 268, 32), (68608, 8576, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_0.run(arg1_1, buf0, 205824, stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((3, 8, 32, 445), (113920, 14240, 445, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg2_1, buf1, 768, 445, stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((24, 268, 445), (119260, 445, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (24, 268, 32), (8576, 32, 1), 0), reinterpret_tensor(buf1, (24, 32, 445), (14240, 445, 1), 0), out=buf2)
        buf6 = empty_strided_cuda((3, 8, 268, 445), (954112, 119264, 445, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__safe_softmax_add_scalar_tensor_where_2.run(buf2, arg0_1, buf6, 6432, 445, stream=stream0)
        del arg0_1
        del buf2
        buf7 = reinterpret_tensor(buf1, (3, 2, 4, 445, 32), (113920, 56960, 14240, 32, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(arg3_1, buf7, 341760, stream=stream0)
        del arg3_1
        buf8 = reinterpret_tensor(buf0, (24, 268, 32), (8576, 32, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (24, 268, 445), (119264, 445, 1), 0), reinterpret_tensor(buf7, (24, 445, 32), (14240, 32, 1), 0), out=buf8)
        del buf6
        del buf7
        buf9 = empty_strided_cuda((3, 8, 268, 32), (68608, 8576, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_to_copy_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf8, buf9, 205824, stream=stream0)
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((268, 445), (445, 1), device='cuda:0', dtype=torch.bool)
    arg1_1 = rand_strided((3, 8, 268, 32), (68608, 8576, 32, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((3, 2, 445, 32), (28480, 14240, 32, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((3, 2, 445, 32), (28480, 14240, 32, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
