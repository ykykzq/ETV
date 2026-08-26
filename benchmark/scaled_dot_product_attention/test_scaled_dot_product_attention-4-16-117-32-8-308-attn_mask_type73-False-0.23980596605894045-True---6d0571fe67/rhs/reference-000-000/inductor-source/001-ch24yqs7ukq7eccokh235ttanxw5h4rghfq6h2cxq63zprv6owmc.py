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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-16-117-32-8-308-attn_mask_type73-False-0.23980596605894045-True---6d0571fe67/rhs/reference-000-000/torchinductor-cache/ms/cmstqvrce7ng3hfxrw7an7lqkil4zgkgqmvcc3c2oc6rrzi4yllk.py
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2875392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 239616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-16-117-32-8-308-attn_mask_type73-False-0.23980596605894045-True---6d0571fe67/rhs/reference-000-000/torchinductor-cache/xn/cxnvj5d3bytc7kf6ohu26pylijjfdof3za5kjh5t67zpxjypzmsj.py
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2523136, 'x': 5046272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 308
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 32)
    y1 = ((yindex // 32) % 16)
    y2 = yindex // 512
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x3 + 9856*(y1 // 2) + 78848*y2), xmask, eviction_policy='evict_last')
    tmp1 = 0.48969987345203636
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3 + 308*y4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-16-117-32-8-308-attn_mask_type73-False-0.23980596605894045-True---6d0571fe67/rhs/reference-000-000/torchinductor-cache/h6/ch6rr2xgaajsngfpjkmg2q765qc5umqhhb36q4cqmkxdajyz5u4x.py
# Topologically Sorted Source Nodes: [add, _safe_softmax], Original ATen: [aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, div, eq, full_default, logical_not, logical_not_1, where
#   add => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %arg3_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 16, 117, 308], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %div), kwargs = {})
triton_red_fused__safe_softmax_add_2 = async_compile.triton('triton_red_fused__safe_softmax_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 27819792}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 7488
    r0_numel = 308
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 117)
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 1872
    x6 = (xindex % 1872)
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 308*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 308*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = float("-inf")
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tmp6 = tmp5.to(tl.int64)
        tmp7 = (tmp6 != 0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 | tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])

        _tmp12_max_next, _tmp12_sum_next = triton_helpers.online_softmax_combine(
            _tmp12_max, _tmp12_sum, tmp11, False
        )

        _tmp12_max = tl.where(r0_mask & xmask, _tmp12_max_next, _tmp12_max)
        _tmp12_sum = tl.where(r0_mask & xmask, _tmp12_sum_next, _tmp12_sum)
    tmp9 = triton_helpers.any(_tmp9.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp14, tmp15 = triton_helpers.online_softmax_reduce(
        _tmp12_max, _tmp12_sum, 1, False)
    tmp14 = tmp14[:, None]
    tmp15 = tmp15[:, None]
    tmp12 = tmp14
    tmp13 = tmp15
    x5 = xindex // 117
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_3 + 308*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_3 + 308*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp9 == 0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19 - tmp12
        tmp21 = tl_math.exp(tmp20)
        tmp22 = (tmp21 / tmp13)
        tmp23 = 0.0
        tmp24 = tl.where(tmp16, tmp23, tmp22)
        tl.store(out_ptr3 + (r0_3 + 308*x0 + 36064*x5), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-4-16-117-32-8-308-attn_mask_type73-False-0.23980596605894045-True---6d0571fe67/rhs/reference-000-000/torchinductor-cache/db/cdbbm5kou5lhhbr4ujidzsvodsubpoyavoqqq7ntxwgxqbsgspuz.py
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6307840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 630784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 9856)
    x2 = xindex // 19712
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 9856*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 16, 117, 32), (59904, 3744, 32, 1))
    assert_size_stride(arg1_1, (4, 8, 308, 32), (78848, 9856, 32, 1))
    assert_size_stride(arg2_1, (4, 8, 308, 32), (78848, 9856, 32, 1))
    assert_size_stride(arg3_1, (117, 308), (308, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 16, 117, 32), (59904, 3744, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 239616, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 16, 32, 308), (157696, 9856, 308, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 2048, 308, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((64, 117, 308), (36036, 308, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (64, 117, 32), (3744, 32, 1), 0), reinterpret_tensor(buf1, (64, 32, 308), (9856, 308, 1), 0), out=buf2)
        buf6 = empty_strided_cuda((4, 16, 117, 308), (577024, 36064, 308, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, _safe_softmax], Original ATen: [aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__safe_softmax_add_2.run(buf2, arg3_1, buf6, 7488, 308, stream=stream0)
        del arg3_1
        del buf2
        buf7 = reinterpret_tensor(buf1, (4, 8, 2, 308, 32), (157696, 19712, 9856, 32, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(arg2_1, buf7, 630784, stream=stream0)
        del arg2_1
        buf8 = reinterpret_tensor(buf0, (64, 117, 32), (3744, 32, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (64, 117, 308), (36064, 308, 1), 0), reinterpret_tensor(buf7, (64, 308, 32), (9856, 32, 1), 0), out=buf8)
        del buf6
        del buf7
    return (reinterpret_tensor(buf8, (4, 16, 117, 32), (59904, 3744, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 16, 117, 32), (59904, 3744, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 8, 308, 32), (78848, 9856, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 8, 308, 32), (78848, 9856, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((117, 308), (308, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
