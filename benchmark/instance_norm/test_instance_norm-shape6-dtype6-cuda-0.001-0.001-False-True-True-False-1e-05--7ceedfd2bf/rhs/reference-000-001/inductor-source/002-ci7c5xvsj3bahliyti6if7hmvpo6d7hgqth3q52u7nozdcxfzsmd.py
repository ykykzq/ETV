# AOT ID: ['1_inference']
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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape6-dtype6-cuda-0.001-0.001-False-True-True-False-1e-05--7ceedfd2bf/rhs/reference-000-001/torchinductor-cache/7h/c7hwkd32m5zrd3vyyk5nyox2ncr5vfn7exn67mexb5rgnrz2kq2j.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_stats => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused__native_batch_norm_legit_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 174
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 3.0
    tmp6 = (tmp4 / tmp5)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape6-dtype6-cuda-0.001-0.001-False-True-True-False-1e-05--7ceedfd2bf/rhs/reference-000-001/torchinductor-cache/hm/chm7biitpxmkafvdkngtlxgs6mnlyccdsnwhrgeer6nnbwj57z6j.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_stats => add, add_1, mul, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 3*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 3*x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (((x2 // 3) % 29)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp3 - tmp1
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 - tmp1
    tmp8 = tmp7 * tmp7
    tmp9 = tmp5 + tmp8
    tmp11 = tmp10 - tmp1
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 3.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp2 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (29, ), (1, ))
    assert_size_stride(arg1_1, (6, 29, 3, 1), (87, 3, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 174, 1, 1), (174, 1, 174, 174), torch.float32)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0.run(arg1_1, buf0, 174, stream=stream0)
        buf1 = empty_strided_cuda((1, 174, 3, 1), (522, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_1.run(arg1_1, buf0, arg0_1, buf1, 522, stream=stream0)
        del arg0_1
        del arg1_1
        del buf0
    return (reinterpret_tensor(buf1, (6, 29, 3, 1), (87, 3, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((6, 29, 3, 1), (87, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
