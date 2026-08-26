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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape7-dtype7-cuda-0.01-0.01-False-False-True-False-1e-05--233115369b/rhs/reference-000-000/torchinductor-cache/e2/ce2uqt4pomqwrnsuhf6ldesrslp6uhugbe4h7bdmw34wej4aelxx.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_training_default], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_training_default => add_1, convert_element_type_2, mul_1, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float16), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4032}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (((x2 // 56) % 4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (((x2 // 56) % 4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (((x2 // 56) % 4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = (tmp10 / tmp9)
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape7-dtype7-cuda-0.01-0.01-False-False-True-False-1e-05--233115369b/rhs/reference-000-000/torchinductor-cache/rp/crp7b42bxoxuuxdncvqsrn36yv7x4lrnvqo6dxoytsq32ddzcs4f.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_1, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %mean), kwargs = {})
triton_poi_fused_mean_1 = async_compile.triton('triton_poi_fused_mean_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 + tmp1
    tmp3 = tmp2 + tmp1
    tmp4 = 3.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, ), (1, ))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg2_1, (4, ), (1, ))
    assert_size_stride(arg3_1, (3, 4, 14, 4), (224, 56, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 12, 14, 4), (672, 56, 4, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_training_default], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_0.run(arg3_1, arg1_1, arg2_1, arg0_1, buf0, 672, stream=stream0)
        del arg0_1
        del arg3_1
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(arg1_1, arg1_1, 4, stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(arg2_1, arg2_1, 4, stream=stream0)
        del arg2_1
    return (reinterpret_tensor(buf0, (3, 4, 14, 4), (224, 56, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((3, 4, 14, 4), (224, 56, 4, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
