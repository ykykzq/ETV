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


# kernel path: /data3/yelv.lh/ETV/benchmark/softmax/test_softmax-shape7-dtype7-cuda-0.01-0.01--674b10b9c8/rhs/reference-000-000/torchinductor-cache/a5/ca577xafo6iwmrojul2nwaqk4xiurbm2u54sgjx2li5elvkmj3lj.py
# Topologically Sorted Source Nodes: [_to_copy], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float64), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, 2), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12*x1), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (2 + x0 + 12*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 12*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (6 + x0 + 12*x1), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (8 + x0 + 12*x1), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (10 + x0 + 12*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float64)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float64)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float64)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp15 = tmp14.to(tl.float64)
    tmp16 = triton_helpers.maximum(tmp13, tmp15)
    tmp17 = tmp1 - tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tmp3 - tmp16
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp6 - tmp16
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp9 - tmp16
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp12 - tmp16
    tmp29 = libdevice.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp15 - tmp16
    tmp32 = libdevice.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr0 + (x2), tmp16, xmask)
    tl.store(out_ptr1 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/softmax/test_softmax-shape7-dtype7-cuda-0.01-0.01--674b10b9c8/rhs/reference-000-000/torchinductor-cache/hv/chvihg5bvfoitkeo3v3renswcnxqbhurg7wsvtndz3zvr2g3l4cm.py
# Topologically Sorted Source Nodes: [_to_copy, _softmax], Original ATen: [aten._to_copy, aten._softmax]
# Source node to ATen node mapping:
#   _softmax => div
#   _to_copy => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float64), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type, 2), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 252
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 2)
    x2 = xindex // 12
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
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
    assert_size_stride(arg0_1, (7, 3, 6, 2), (36, 12, 2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((7, 3, 1, 2), (6, 2, 42, 1), torch.float64)
        buf1 = empty_strided_cuda((7, 3, 1, 2), (6, 2, 42, 1), torch.float64)
        # Topologically Sorted Source Nodes: [_to_copy], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, buf1, 42, stream=stream0)
        buf2 = empty_strided_cuda((7, 3, 6, 2), (36, 12, 2, 1), torch.float64)
        # Topologically Sorted Source Nodes: [_to_copy, _softmax], Original ATen: [aten._to_copy, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy_1.run(arg0_1, buf0, buf1, buf2, 252, stream=stream0)
        del arg0_1
        del buf0
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7, 3, 6, 2), (36, 12, 2, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
