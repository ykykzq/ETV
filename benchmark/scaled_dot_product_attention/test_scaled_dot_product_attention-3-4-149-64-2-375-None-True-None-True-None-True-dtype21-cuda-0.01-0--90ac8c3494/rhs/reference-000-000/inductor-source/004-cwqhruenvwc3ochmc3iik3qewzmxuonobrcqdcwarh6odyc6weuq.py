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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-4-149-64-2-375-None-True-None-True-None-True-dtype21-cuda-0.01-0--90ac8c3494/rhs/reference-000-000/torchinductor-cache/3e/c3exg66dvslee7y6352onndiqqfqnfysphvvxh7zg7va2spgm2go.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg0_1, 0.3535533905932738), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1373184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-4-149-64-2-375-None-True-None-True-None-True-dtype21-cuda-0.01-0--90ac8c3494/rhs/reference-000-000/torchinductor-cache/nq/cnqmo4y7i6kmc3rwtfaqfloefkyora2sldkprmswmrj2hai3qgss.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1152000, 'x': 2304000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 375
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 4)
    y2 = yindex // 256
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x3 + 24000*(y1 // 2) + 48000*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3 + 375*y4), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-4-149-64-2-375-None-True-None-True-None-True-dtype21-cuda-0.01-0--90ac8c3494/rhs/reference-000-000/torchinductor-cache/hk/chkcgp7ajgxmtlx77gl222kqthnmmtidyosodb7a47shhrvtgt3e.py
# Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => any_1, div, eq, full_default_3, logical_not, logical_not_1, where_1
#   add => add
#   ones => full_default
#   scalar_tensor => full_default_1
#   scalar_tensor_1 => full_default_2
#   tril => le, logical_and, sub
#   where => where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([149, 375], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([3, 4, 149, 375], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_3, %div), kwargs = {})
triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2 = async_compile.triton('triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 8046000}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr3, xnumel, r0_numel):
    xnumel = 1788
    XBLOCK: tl.constexpr = 1
    r0_numel = 375
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
    x0 = (xindex % 149)
    x1 = xindex // 149
    tmp0 = tl.load(in_ptr0 + (r0_2 + 375*x3), r0_mask, other=0.0)
    tmp1 = r0_2 + ((-1)*x0)
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
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, False)
    tmp17 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp16, 0))
    tmp18 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, float("-inf"))
    tmp23 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp22, 0))
    tmp24 = tmp18 - tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tl.broadcast_to(tmp25, [R0_BLOCK])
    tmp28 = tl.where(r0_mask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tmp17 == 0
    tmp31 = tmp9 - tmp23
    tmp32 = tl_math.exp(tmp31)
    tmp33 = (tmp32 / tmp29)
    tmp34 = tl.where(tmp30, tmp6, tmp33)
    tl.store(out_ptr3 + (r0_2 + 375*x0 + 55904*x1), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-3-4-149-64-2-375-None-True-None-True-None-True-dtype21-cuda-0.01-0--90ac8c3494/rhs/reference-000-000/torchinductor-cache/45/c45zmykbfyej3jc65nqdevkdck2pneicejckpgvzzvatsmuxq4kz.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2880000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 24000)
    x2 = xindex // 48000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24000*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (3, 4, 149, 64), (38144, 9536, 64, 1))
    assert_size_stride(arg1_1, (3, 2, 375, 64), (48000, 24000, 64, 1))
    assert_size_stride(arg2_1, (3, 2, 375, 64), (48000, 24000, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 4, 149, 64), (38144, 9536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 114432, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((3, 4, 64, 375), (96000, 24000, 375, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 768, 375, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((12, 149, 375), (55875, 375, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (12, 149, 64), (9536, 64, 1), 0), reinterpret_tensor(buf1, (12, 64, 375), (24000, 375, 1), 0), out=buf2)
        buf6 = empty_strided_cuda((3, 4, 149, 375), (223616, 55904, 375, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tril, ones, scalar_tensor_1, scalar_tensor, where, add, _safe_softmax], Original ATen: [aten.tril, aten.ones, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2.run(buf2, buf6, 1788, 375, stream=stream0)
        del buf2
        buf7 = reinterpret_tensor(buf1, (3, 2, 2, 375, 64), (96000, 48000, 24000, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(arg2_1, buf7, 288000, stream=stream0)
        del arg2_1
        buf8 = reinterpret_tensor(buf0, (12, 149, 64), (9536, 64, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (12, 149, 375), (55904, 375, 1), 0), reinterpret_tensor(buf7, (12, 375, 64), (24000, 64, 1), 0), out=buf8)
        del buf6
        del buf7
    return (reinterpret_tensor(buf8, (3, 4, 149, 64), (38144, 9536, 64, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 4, 149, 64), (38144, 9536, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((3, 2, 375, 64), (48000, 24000, 64, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((3, 2, 375, 64), (48000, 24000, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
