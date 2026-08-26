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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/sq/csq5qzgzbvjw36zpy2bmejvexxcryzo234gtv5hp7y3ntjqbpmj6.py
# Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   mul => mul
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.48969987345203636), kwargs = {})
triton_poi_fused__to_copy_mul_0 = async_compile.triton('triton_poi_fused__to_copy_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9768960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 976896
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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/jx/cjxrvpqequblblcleqdi52hbkatfmv5db4b6cudbdmaefzb4x2cv.py
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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 81920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 64)
    x2 = xindex // 320
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64*x0 + 320*(x2 // 4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.48969987345203636
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/2o/c2oahop4fofhfdy3jz3zja5uz2duixumnjpkus3u4iv6vixhf7tp.py
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
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([477, 5], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2 = async_compile.triton('triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 274752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 477)
    tmp0 = tl.load(in_ptr0 + (5*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 5*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (2 + 5*x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (3 + 5*x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (4 + 5*x2), xmask, eviction_policy='evict_last')
    tmp1 = (-1)*x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = 0.0
    tmp7 = float("-inf")
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp9 == tmp7
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.int64)
    tmp13 = (tmp12 != 0)
    tmp15 = 1 + ((-1)*x0)
    tmp16 = tmp15 <= tmp2
    tmp17 = tmp16 & tmp4
    tmp18 = tl.where(tmp17, tmp6, tmp7)
    tmp19 = tmp14 + tmp18
    tmp20 = tmp19 == tmp7
    tmp21 = tmp20 == 0
    tmp22 = tmp21.to(tl.int64)
    tmp23 = (tmp22 != 0)
    tmp24 = tmp13 | tmp23
    tmp26 = 2 + ((-1)*x0)
    tmp27 = tmp26 <= tmp2
    tmp28 = tmp27 & tmp4
    tmp29 = tl.where(tmp28, tmp6, tmp7)
    tmp30 = tmp25 + tmp29
    tmp31 = tmp30 == tmp7
    tmp32 = tmp31 == 0
    tmp33 = tmp32.to(tl.int64)
    tmp34 = (tmp33 != 0)
    tmp35 = tmp24 | tmp34
    tmp37 = 3 + ((-1)*x0)
    tmp38 = tmp37 <= tmp2
    tmp39 = tmp38 & tmp4
    tmp40 = tl.where(tmp39, tmp6, tmp7)
    tmp41 = tmp36 + tmp40
    tmp42 = tmp41 == tmp7
    tmp43 = tmp42 == 0
    tmp44 = tmp43.to(tl.int64)
    tmp45 = (tmp44 != 0)
    tmp46 = tmp35 | tmp45
    tmp48 = 4 + ((-1)*x0)
    tmp49 = tmp48 <= tmp2
    tmp50 = tmp49 & tmp4
    tmp51 = tl.where(tmp50, tmp6, tmp7)
    tmp52 = tmp47 + tmp51
    tmp53 = tmp52 == tmp7
    tmp54 = tmp53 == 0
    tmp55 = tmp54.to(tl.int64)
    tmp56 = (tmp55 != 0)
    tmp57 = tmp46 | tmp56
    tmp58 = triton_helpers.maximum(tmp9, tmp19)
    tmp59 = triton_helpers.maximum(tmp58, tmp30)
    tmp60 = triton_helpers.maximum(tmp59, tmp41)
    tmp61 = triton_helpers.maximum(tmp60, tmp52)
    tmp62 = tmp9 - tmp61
    tmp63 = tl_math.exp(tmp62)
    tmp64 = tmp19 - tmp61
    tmp65 = tl_math.exp(tmp64)
    tmp66 = tmp63 + tmp65
    tmp67 = tmp30 - tmp61
    tmp68 = tl_math.exp(tmp67)
    tmp69 = tmp66 + tmp68
    tmp70 = tmp41 - tmp61
    tmp71 = tl_math.exp(tmp70)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp52 - tmp61
    tmp74 = tl_math.exp(tmp73)
    tmp75 = tmp72 + tmp74
    tl.store(out_ptr0 + (x2), tmp57, xmask)
    tl.store(out_ptr1 + (x2), tmp61, xmask)
    tl.store(out_ptr2 + (x2), tmp75, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/nm/cnm6opkw3qnf6nbsp2g7ynyuyzv4dvkfvpmdg6lxvun5iw5brvua.py
# Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => div, full_default_3, logical_not_1, where_1
#   add => add
#   ones => full_default
#   scalar_tensor => full_default_1
#   scalar_tensor_1 => full_default_2
#   tril => le, logical_and, sub
#   where => where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([477, 5], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 477, 5], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_3, %div), kwargs = {})
triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3 = async_compile.triton('triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 915840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 5
    x4 = xindex
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 477)
    x2 = xindex // 2385
    x5 = (xindex % 2385)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x4), xmask)
    tmp12 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp3 = x0 + ((-1)*x1)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], True, tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.where(tmp1, tmp8, tmp16)
    tl.store(out_ptr0 + (x5 + 2400*x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/xd/cxdhyylfffzpmw54l7g4ufq6x3vm6fvyjg6rx2wqfgarh45cys5q.py
# Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   clone_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 87040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 320)
    x2 = xindex // 1280
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 320*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-32-477-64-8-5-None-True-0.23980596605894045-True-None-True-dtype--5a6a276034/rhs/reference-000-000/torchinductor-cache/ax/caxs3kkmkejuz627fyebqa47ba7bkr5eckc7gf65c7e5wyyljcbf.py
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7815168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 976896
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 32, 477, 64), (976896, 30528, 64, 1))
    assert_size_stride(arg1_1, (1, 8, 5, 64), (2560, 320, 64, 1))
    assert_size_stride(arg2_1, (1, 8, 5, 64), (2560, 320, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 477, 64), (976896, 30528, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_to_copy, mul], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_0.run(arg0_1, buf0, 976896, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 32, 64, 5), (10240, 320, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 10240, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((32, 477, 5), (2385, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (32, 477, 64), (30528, 64, 1), 0), reinterpret_tensor(buf1, (32, 64, 5), (320, 5, 1), 0), out=buf2)
        buf3 = empty_strided_cuda((1, 32, 477, 1), (15360, 477, 1, 15360), torch.bool)
        buf4 = empty_strided_cuda((1, 32, 477, 1), (15264, 477, 1, 15264), torch.float32)
        buf5 = empty_strided_cuda((1, 32, 477, 1), (15264, 477, 1, 15264), torch.float32)
        # Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2.run(buf2, buf3, buf4, buf5, 15264, stream=stream0)
        buf6 = empty_strided_cuda((1, 32, 477, 5), (76800, 2400, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3.run(buf3, buf2, buf4, buf5, buf6, 76320, stream=stream0)
        del buf2
        del buf3
        del buf4
        del buf5
        buf7 = reinterpret_tensor(buf1, (1, 8, 4, 5, 64), (10240, 1280, 320, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(arg2_1, buf7, 10240, stream=stream0)
        del arg2_1
        buf8 = reinterpret_tensor(buf0, (32, 477, 64), (30528, 64, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (32, 477, 5), (2400, 5, 1), 0), reinterpret_tensor(buf7, (32, 5, 64), (320, 64, 1), 0), out=buf8)
        del buf6
        del buf7
        buf9 = empty_strided_cuda((1, 32, 477, 64), (976896, 30528, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_to_copy_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf8, buf9, 976896, stream=stream0)
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 32, 477, 64), (976896, 30528, 64, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, 8, 5, 64), (2560, 320, 64, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, 8, 5, 64), (2560, 320, 64, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
