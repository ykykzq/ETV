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


# kernel path: /data3/yelv.lh/ETV/benchmark/softmax/test_softmax-shape6-dtype6-cuda-0.001-0.001--ed87a99b04/rhs/reference-000-000/torchinductor-cache/wu/cwu6cuphyzfp7hngrjcnckooa4uptartmf5myhboll6iel6qlw64.py
# Topologically Sorted Source Nodes: [_to_copy], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float64), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, 2), kwargs = {})
triton_per_fused__to_copy_0 = async_compile.triton('triton_per_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1472, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 15
    R0_BLOCK: tl.constexpr = 16
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
    r0_2 = r0_index
    x0 = (xindex % 2)
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*r0_2 + 30*x1), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp2 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/softmax/test_softmax-shape6-dtype6-cuda-0.001-0.001--ed87a99b04/rhs/reference-000-000/torchinductor-cache/ga/cgaqnjest2v5h7zt2ykn47d2pflbxe4mibr7va5nhulpuuvjym5g.py
# Topologically Sorted Source Nodes: [_to_copy, _softmax], Original ATen: [aten._to_copy, aten._softmax]
# Source node to ATen node mapping:
#   _softmax => div
#   _to_copy => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float64), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
triton_poi_fused__softmax__to_copy_1 = async_compile.triton('triton_poi_fused__softmax__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 2)
    x2 = xindex // 30
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr1 + (x0 + 2*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tmp1 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp6 = (tmp4 / tmp5)
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 4, 15, 2), (120, 30, 2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 4, 1, 2), (8, 2, 16, 1), torch.float64)
        buf1 = empty_strided_cuda((2, 4, 1, 2), (8, 2, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [_to_copy], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_0.run(arg0_1, buf0, buf1, 16, 15, stream=stream0)
        buf2 = empty_strided_cuda((2, 4, 15, 2), (120, 30, 2, 1), torch.float64)
        # Topologically Sorted Source Nodes: [_to_copy, _softmax], Original ATen: [aten._to_copy, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy_1.run(arg0_1, buf0, buf1, buf2, 240, stream=stream0)
        del arg0_1
        del buf0
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 4, 15, 2), (120, 30, 2, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
