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


# kernel path: /data3/yelv.lh/ETV/benchmark/max/test_max_dim-False-shape7-dtype7-cuda-0.01-0.01--ea5c662502/rhs/reference-000-000/torchinductor-cache/gv/cgvg72jnlchhxfgtiamnvlia3oji42j3xpglssztcnmnjt6loofa.py
# Topologically Sorted Source Nodes: [max_dim], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_dim => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%max_1, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%max_1, 1), kwargs = {})
triton_poi_fused_max_0 = async_compile.triton('triton_poi_fused_max_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4500}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 25)
    x1 = xindex // 25
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 125*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (25 + x0 + 125*x1), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (50 + x0 + 125*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (75 + x0 + 125*x1), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (100 + x0 + 125*x1), xmask).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp0 > tmp1
    tmp10 = tmp0 == tmp1
    tmp11 = tmp0 != tmp0
    tmp12 = tmp1 != tmp1
    tmp13 = tmp11 > tmp12
    tmp14 = tmp9 | tmp13
    tmp15 = tmp11 & tmp12
    tmp16 = tmp10 | tmp15
    tmp17 = tl.full([1], 0, tl.int64)
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 < tmp18
    tmp20 = tmp16 & tmp19
    tmp21 = tmp14 | tmp20
    tmp22 = tl.where(tmp21, tmp0, tmp1)
    tmp23 = tl.where(tmp21, tmp17, tmp18)
    tmp24 = tmp22 > tmp3
    tmp25 = tmp22 == tmp3
    tmp26 = tmp22 != tmp22
    tmp27 = tmp3 != tmp3
    tmp28 = tmp26 > tmp27
    tmp29 = tmp24 | tmp28
    tmp30 = tmp26 & tmp27
    tmp31 = tmp25 | tmp30
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp23 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tmp29 | tmp34
    tmp36 = tl.where(tmp35, tmp22, tmp3)
    tmp37 = tl.where(tmp35, tmp23, tmp32)
    tmp38 = tmp36 > tmp5
    tmp39 = tmp36 == tmp5
    tmp40 = tmp36 != tmp36
    tmp41 = tmp5 != tmp5
    tmp42 = tmp40 > tmp41
    tmp43 = tmp38 | tmp42
    tmp44 = tmp40 & tmp41
    tmp45 = tmp39 | tmp44
    tmp46 = tl.full([1], 3, tl.int64)
    tmp47 = tmp37 < tmp46
    tmp48 = tmp45 & tmp47
    tmp49 = tmp43 | tmp48
    tmp50 = tl.where(tmp49, tmp36, tmp5)
    tmp51 = tl.where(tmp49, tmp37, tmp46)
    tmp52 = tmp50 > tmp7
    tmp53 = tmp50 == tmp7
    tmp54 = tmp50 != tmp50
    tmp55 = tmp7 != tmp7
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tl.full([1], 4, tl.int64)
    tmp61 = tmp51 < tmp60
    tmp62 = tmp59 & tmp61
    tmp63 = tmp57 | tmp62
    tmp64 = tl.where(tmp63, tmp50, tmp7)
    tmp65 = tl.where(tmp63, tmp51, tmp60)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
    tl.store(out_ptr1 + (x2), tmp65, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (6, 5, 5, 5), (125, 25, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((6, 5, 5), (25, 5, 1), torch.float16)
        buf1 = empty_strided_cuda((6, 5, 5), (25, 5, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_dim], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_0.run(arg0_1, buf0, buf1, 150, stream=stream0)
        del arg0_1
    return (buf0, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((6, 5, 5, 5), (125, 25, 5, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
