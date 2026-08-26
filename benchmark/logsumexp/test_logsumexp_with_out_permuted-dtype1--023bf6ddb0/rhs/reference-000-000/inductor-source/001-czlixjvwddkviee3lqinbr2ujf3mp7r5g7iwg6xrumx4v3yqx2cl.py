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


# kernel path: /data3/yelv.lh/ETV/benchmark/logsumexp/test_logsumexp_with_out_permuted-dtype1--023bf6ddb0/rhs/reference-000-000/torchinductor-cache/md/cmdu7i2i5cojd4a5yvodj5idswckq2l5ooaotvv2ap5dmdrllmnk.py
# Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   logsumexp => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
# Graph fragment:
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [2], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [2], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %where), kwargs = {})
triton_poi_fused_logsumexp_0 = async_compile.triton('triton_poi_fused_logsumexp_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 30*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (6 + x0 + 30*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (12 + x0 + 30*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (18 + x0 + 30*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (24 + x0 + 30*x1), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl_math.abs(tmp8)
    tmp10 = float("inf")
    tmp11 = tmp9 == tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp11, tmp12, tmp8)
    tmp14 = tmp0 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp1 - tmp13
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp3 - tmp13
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp5 - tmp13
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp7 - tmp13
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tl_math.log(tmp27)
    tmp29 = tmp28 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 4, 5, 6), (120, 30, 6, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 4, 1, 6), (24, 6, 72, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (3, 4, 1, 6), (24, 6, 6, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_logsumexp_0.run(buf1, arg0_1, 72, stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 4, 5, 6), (120, 30, 6, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
