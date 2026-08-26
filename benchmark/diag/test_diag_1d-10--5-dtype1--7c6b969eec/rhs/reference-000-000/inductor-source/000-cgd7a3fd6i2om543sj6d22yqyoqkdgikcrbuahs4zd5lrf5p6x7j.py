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


# kernel path: /data3/yelv.lh/ETV/benchmark/diag/test_diag_1d-10--5-dtype1--7c6b969eec/rhs/reference-000-000/torchinductor-cache/k7/ck7fmqnmnxl4pc3hqyxwtvnnbrvws7mvnmrmm27mr6ymlkqvh6jw.py
# Topologically Sorted Source Nodes: [diag_embed], Original ATen: [aten.diag_embed]
# Source node to ATen node mapping:
#   diag_embed => eq, full_default_1, iota, where
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (15,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%iota, %unsqueeze_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %permute, %full_default_1), kwargs = {})
triton_poi_fused_diag_embed_0 = async_compile.triton('triton_poi_fused_diag_embed_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_diag_embed_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 930}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 15)
    x1 = xindex // 15
    x2 = xindex
    tmp0 = x0
    tmp1 = (-5) + x1
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tl.full([1], 10, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tl.load(in_ptr0 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 15, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = 0.0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp7, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((15, 15), (15, 1), torch.float16)
        # Topologically Sorted Source Nodes: [diag_embed], Original ATen: [aten.diag_embed]
        stream0 = get_raw_stream(0)
        triton_poi_fused_diag_embed_0.run(arg0_1, buf0, 225, stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
