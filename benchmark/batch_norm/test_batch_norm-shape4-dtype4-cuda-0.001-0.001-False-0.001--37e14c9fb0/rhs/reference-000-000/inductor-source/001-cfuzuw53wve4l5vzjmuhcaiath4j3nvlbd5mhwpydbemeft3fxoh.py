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


# kernel path: /data3/yelv.lh/ETV/benchmark/batch_norm/test_batch_norm-shape4-dtype4-cuda-0.001-0.001-False-0.001--37e14c9fb0/rhs/reference-000-000/torchinductor-cache/if/cifnzw6gda6bmxy7wrlwhalub7krcckb52f5clslhnacxcbbgfe5.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_stats => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused__native_batch_norm_legit_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 320, 'r0_': 3840}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 20
    r0_numel = 48
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
    r0_1 = (r0_index % 12)
    r0_2 = r0_index // 12
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 12*x0 + 240*r0_2), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 48, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/batch_norm/test_batch_norm-shape4-dtype4-cuda-0.001-0.001-False-0.001--37e14c9fb0/rhs/reference-000-000/torchinductor-cache/6f/c6fsijhivm6qbg6q4gfmxc2vnvcnahqfqkiyolfsi7wyw4kkmkag.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_stats => add, mul, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 0.001), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
triton_poi_fused__native_batch_norm_legit_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 12) % 20)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 48.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 20, 12), (240, 12, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 20, 1), (20, 1, 20), torch.float32)
        buf1 = empty_strided_cuda((1, 20, 1), (20, 1, 20), torch.float32)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0.run(arg0_1, buf0, buf1, 20, 48, stream=stream0)
        buf3 = empty_strided_cuda((4, 20, 12), (240, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_1.run(arg0_1, buf0, buf1, buf3, 960, stream=stream0)
        del arg0_1
        del buf0
        del buf1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 20, 12), (240, 12, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
