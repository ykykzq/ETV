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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-362-32-2-482-attn_mask_type51-False-0.23980596605894045-True---68d38829be/rhs/reference-000-000/torchinductor-cache/l7/cl7ivhud6atze6bqbqgyw44ewt26nukmo3mzkfhmxjbq2l4uh3s7.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg1_1, 0.48969987345203636), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8896512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 741376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-362-32-2-482-attn_mask_type51-False-0.23980596605894045-True---68d38829be/rhs/reference-000-000/torchinductor-cache/5u/c5u3s4t3vaaoqjzjhwtumph6k6uhuhcdnax7ibdfnsiovgdanlpu.py
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
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3948544, 'x': 7897088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 482
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
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x3 + 15424*(y1 // 16) + 30848*y2), xmask, eviction_policy='evict_last')
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3 + 482*y4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-362-32-2-482-attn_mask_type51-False-0.23980596605894045-True---68d38829be/rhs/reference-000-000/torchinductor-cache/57/c57imn3y5opn454gvri4mhiek4p5ubhmamd2pgs3e2fn6mfjo22c.py
# Topologically Sorted Source Nodes: [scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, div, eq, full_default_2, logical_not, logical_not_1, where_1
#   add => add
#   scalar_tensor => full_default
#   scalar_tensor_1 => full_default_1
#   where => where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg0_1, %full_default_1, %full_default), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 32, 362, 482], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 134178196}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 23168
    r0_numel = 482
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 362)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 11584
    x6 = (xindex % 11584)
    _tmp14_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp14_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 482*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 482*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp2 = 0.0
        tmp3 = float("-inf")
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp5 = tmp0 + tmp4
        tmp6 = tmp5 == tmp3
        tmp7 = tmp6 == 0
        tmp8 = tmp7.to(tl.int64)
        tmp9 = (tmp8 != 0)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 | tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp13 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])

        _tmp14_max_next, _tmp14_sum_next = triton_helpers.online_softmax_combine(
            _tmp14_max, _tmp14_sum, tmp13, False
        )

        _tmp14_max = tl.where(r0_mask & xmask, _tmp14_max_next, _tmp14_max)
        _tmp14_sum = tl.where(r0_mask & xmask, _tmp14_sum_next, _tmp14_sum)
    tmp11 = triton_helpers.any(_tmp11.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp16, tmp17 = triton_helpers.online_softmax_reduce(
        _tmp14_max, _tmp14_sum, 1, False)
    tmp16 = tmp16[:, None]
    tmp17 = tmp17[:, None]
    tmp14 = tmp16
    tmp15 = tmp17
    x4 = xindex // 362
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp19 = tl.load(in_ptr0 + (r0_3 + 482*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r0_3 + 482*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp18 = tmp11 == 0
        tmp21 = 0.0
        tmp22 = float("-inf")
        tmp23 = tl.where(tmp20, tmp21, tmp22)
        tmp24 = tmp19 + tmp23
        tmp25 = tmp24 - tmp14
        tmp26 = tl_math.exp(tmp25)
        tmp27 = (tmp26 / tmp15)
        tmp28 = tl.where(tmp18, tmp21, tmp27)
        tl.store(out_ptr3 + (r0_3 + 482*x0 + 174496*x4), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-2-32-362-32-2-482-attn_mask_type51-False-0.23980596605894045-True---68d38829be/rhs/reference-000-000/torchinductor-cache/3q/c3qkyztbhcgjtqwpoy73hvlb3nsjpgumqz6grtyc7tqtqifwaa5v.py
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8143872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 987136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 15424)
    x2 = xindex // 246784
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 15424*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (362, 482), (482, 1))
    assert_size_stride(arg1_1, (2, 32, 362, 32), (370688, 11584, 32, 1))
    assert_size_stride(arg2_1, (2, 2, 482, 32), (30848, 15424, 32, 1))
    assert_size_stride(arg3_1, (2, 2, 482, 32), (30848, 15424, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 32, 362, 32), (370688, 11584, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg1_1, buf0, 741376, stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((2, 32, 32, 482), (493568, 15424, 482, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg2_1, buf1, 2048, 482, stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((64, 362, 482), (174484, 482, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (64, 362, 32), (11584, 32, 1), 0), reinterpret_tensor(buf1, (64, 32, 482), (15424, 482, 1), 0), out=buf2)
        buf6 = empty_strided_cuda((2, 32, 362, 482), (5583872, 174496, 482, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__safe_softmax_add_scalar_tensor_where_2.run(buf2, arg0_1, buf6, 23168, 482, stream=stream0)
        del arg0_1
        del buf2
        buf7 = reinterpret_tensor(buf1, (2, 2, 16, 482, 32), (493568, 246784, 15424, 32, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(arg3_1, buf7, 987136, stream=stream0)
        del arg3_1
        buf8 = reinterpret_tensor(buf0, (64, 362, 32), (11584, 32, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (64, 362, 482), (174496, 482, 1), 0), reinterpret_tensor(buf7, (64, 482, 32), (15424, 32, 1), 0), out=buf8)
        del buf6
        del buf7
    return (reinterpret_tensor(buf8, (2, 32, 362, 32), (370688, 11584, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((362, 482), (482, 1), device='cuda:0', dtype=torch.bool)
    arg1_1 = rand_strided((2, 32, 362, 32), (370688, 11584, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 2, 482, 32), (30848, 15424, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2, 2, 482, 32), (30848, 15424, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
