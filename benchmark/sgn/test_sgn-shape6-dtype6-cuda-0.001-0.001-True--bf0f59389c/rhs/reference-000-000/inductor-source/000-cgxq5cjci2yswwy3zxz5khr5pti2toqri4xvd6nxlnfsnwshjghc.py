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


# kernel path: /data3/yelv.lh/ETV/benchmark/sgn/test_sgn-shape6-dtype6-cuda-0.001-0.001-True--bf0f59389c/rhs/reference-000-000/torchinductor-cache/j5/cj5pciyc6dznbp6w7sdyjbjtyckaflxjr2nn2ftk2jq4ktwnwbgw.py
# Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
# Source node to ATen node mapping:
#   sgn => eq
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, 0), kwargs = {})
triton_poi_fused_sgn_0 = async_compile.triton('triton_poi_fused_sgn_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sgn_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sgn_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (6, 15, 1, 10), (150, 10, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
        buf0 = torch.ops.aten.abs.default(arg0_1)
        buf1 = buf0
        assert_size_stride(buf1, (6, 15, 1, 10), (150, 10, 10, 1), 'torch.ops.aten.abs.default')
        assert_alignment(buf1, 16, 'torch.ops.aten.abs.default')
        del buf0
        # Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
        buf2 = torch.ops.aten.full.default([], 0j, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf3 = buf2
        assert_size_stride(buf3, (), (), 'torch.ops.aten.full.default')
        assert_alignment(buf3, 16, 'torch.ops.aten.full.default')
        del buf2
        # Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
        buf4 = torch.ops.aten.div.Tensor(arg0_1, buf1)
        del arg0_1
        buf5 = buf4
        assert_size_stride(buf5, (6, 15, 1, 10), (150, 10, 10, 1), 'torch.ops.aten.div.Tensor')
        assert_alignment(buf5, 16, 'torch.ops.aten.div.Tensor')
        del buf4
        buf6 = empty_strided_cuda((6, 15, 1, 10), (150, 10, 900, 1), torch.bool)
        # Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sgn_0.run(buf1, buf6, 900, stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [sgn], Original ATen: [aten.sgn]
        buf7 = torch.ops.aten.where.self(buf6, buf3, buf5)
        del buf3
        del buf5
        del buf6
        buf8 = buf7
        assert_size_stride(buf8, (6, 15, 1, 10), (150, 10, 900, 1), 'torch.ops.aten.where.self')
        assert_alignment(buf8, 16, 'torch.ops.aten.where.self')
        del buf7
    return (reinterpret_tensor(buf8, (6, 15, 1, 10), (150, 10, 10, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((6, 15, 1, 10), (150, 10, 10, 1), device='cuda:0', dtype=torch.complex64)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
