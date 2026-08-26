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


# kernel path: /data3/yelv.lh/ETV/benchmark/logsumexp/test_logsumexp_with_out_different_strides_dim1-False-dtype1--3d1d8fe02e/rhs/reference-000-000/torchinductor-cache/xg/cxge5pkbui3lny2v6xp46qwd6imz4muyfbxoqitgm4s6ql5gfldl.py
# Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   logsumexp => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
# Graph fragment:
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
triton_per_fused_logsumexp_0 = async_compile.triton('triton_per_fused_logsumexp_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_logsumexp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 600, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_logsumexp_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 5)
    x1 = xindex // 5
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*r0_2 + 40*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tl_math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp0 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp16 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 8, 5), (40, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((3, 5), (5, 1), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf2, arg0_1, 15, 8, stream=stream0)
        del arg0_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 8, 5), (40, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
