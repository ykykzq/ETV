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


# kernel path: /data3/yelv.lh/ETV/benchmark/logsumexp/test_logsumexp-True-shape3-dtype3-cuda-0.01-0.01--c95386543c/rhs/reference-000-000/torchinductor-cache/vs/cvs5ld7qmima4ob6puarw6gtjucoeu4ogldfyaylddgpw2zwzb65.py
# Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   logsumexp => abs_1, add, amax, convert_element_type, convert_element_type_1, eq, exp, full_default, log, sub, sum_1, where
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %where), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float16), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 484}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_0(in_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = triton_helpers.maximum(tmp13, tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = triton_helpers.maximum(tmp16, tmp18)
    tmp20 = tl_math.abs(tmp19)
    tmp21 = float("inf")
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp19)
    tmp25 = tmp1 - tmp24
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp3 - tmp24
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp6 - tmp24
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp9 - tmp24
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp32 + tmp34
    tmp36 = tmp12 - tmp24
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp15 - tmp24
    tmp40 = tl_math.exp(tmp39)
    tmp41 = tmp38 + tmp40
    tmp42 = tmp18 - tmp24
    tmp43 = tl_math.exp(tmp42)
    tmp44 = tmp41 + tmp43
    tmp45 = tl_math.log(tmp44)
    tmp46 = tmp45 + tmp24
    tmp47 = tmp46.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp47, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (121, 7), (7, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((121, 1), (1, 1), torch.float16)
        # Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_logsumexp_0.run(arg0_1, buf2, 121, stream=stream0)
        del arg0_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((121, 7), (7, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
