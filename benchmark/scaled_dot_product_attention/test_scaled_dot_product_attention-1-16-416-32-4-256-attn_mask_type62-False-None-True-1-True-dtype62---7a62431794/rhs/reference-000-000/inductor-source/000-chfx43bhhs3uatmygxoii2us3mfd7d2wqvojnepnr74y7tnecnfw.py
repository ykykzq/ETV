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


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-16-416-32-4-256-attn_mask_type62-False-None-True-1-True-dtype62---7a62431794/rhs/reference-000-000/torchinductor-cache/vw/cvwroh3irzich7d5lscb5mm24ksjytw5i7jm2jmiuimhvrapuwky.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg0_1, 0.42044820762685725), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2555904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 212992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.42044820762685725
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-16-416-32-4-256-attn_mask_type62-False-None-True-1-True-dtype62---7a62431794/rhs/reference-000-000/torchinductor-cache/my/cmyzqwijujublyc2idumd6j5tokxtxsumrgzpy6y7fgsevprroch.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute, 0.42044820762685725), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 524288, 'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x2 + 8192*(y1 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.42044820762685725
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 256*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-16-416-32-4-256-attn_mask_type62-False-None-True-1-True-dtype62---7a62431794/rhs/reference-000-000/torchinductor-cache/uo/cuozcvjudluk7xdpuj5c23722d224rlytw7llpbmxdju4uex2nfs.py
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
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 16, 416, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %div), kwargs = {})
triton_per_fused__safe_softmax_add_2 = async_compile.triton('triton_per_fused__safe_softmax_add_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 20873216}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_2(in_out_ptr0, in_ptr0, xnumel, r0_numel):
    xnumel = 6656
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 416)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (r0_2 + 256*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = float("-inf")
    tmp4 = tmp2 == tmp3
    tmp5 = tmp4 == 0
    tmp6 = tmp5.to(tl.int64)
    tmp7 = (tmp6 != 0)
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp13 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp15 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp13, 0))
    tmp16 = tmp11 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tmp10 == 0
    tmp22 = tmp2 - tmp15
    tmp23 = tl_math.exp(tmp22)
    tmp24 = (tmp23 / tmp20)
    tmp25 = 0.0
    tmp26 = tl.where(tmp21, tmp25, tmp24)
    tl.store(in_out_ptr0 + (r0_2 + 256*x3), tmp26, None)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/scaled_dot_product_attention/test_scaled_dot_product_attention-1-16-416-32-4-256-attn_mask_type62-False-None-True-1-True-dtype62---7a62431794/rhs/reference-000-000/torchinductor-cache/ra/craqb6ijsxxtdqqsrcgzt6s7qa7asamkkdhzh5gh5cvatszcz2km.py
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 16, 416, 32), (212992, 13312, 32, 1))
    assert_size_stride(arg1_1, (1, 4, 256, 32), (32768, 8192, 32, 1))
    assert_size_stride(arg2_1, (1, 4, 256, 32), (32768, 8192, 32, 1))
    assert_size_stride(arg3_1, (416, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 16, 416, 32), (212992, 13312, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf0, 212992, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 16, 32, 256), (131072, 8192, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf1, 512, 256, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((16, 416, 256), (106496, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (16, 416, 32), (13312, 32, 1), 0), reinterpret_tensor(buf1, (16, 32, 256), (8192, 256, 1), 0), out=buf2)
        buf6 = reinterpret_tensor(buf2, (1, 16, 416, 256), (1703936, 106496, 256, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [add, _safe_softmax], Original ATen: [aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_add_2.run(buf6, arg3_1, 6656, 256, stream=stream0)
        del arg3_1
        buf7 = reinterpret_tensor(buf1, (1, 4, 4, 256, 32), (131072, 32768, 8192, 32, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [clone_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(arg2_1, buf7, 131072, stream=stream0)
        del arg2_1
        buf8 = reinterpret_tensor(buf0, (16, 416, 32), (13312, 32, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (16, 416, 256), (106496, 256, 1), 0), reinterpret_tensor(buf7, (16, 256, 32), (8192, 32, 1), 0), out=buf8)
        del buf6
        del buf7
    return (reinterpret_tensor(buf8, (1, 16, 416, 32), (212992, 13312, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16, 416, 32), (212992, 13312, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 4, 256, 32), (32768, 8192, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 4, 256, 32), (32768, 8192, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((416, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
