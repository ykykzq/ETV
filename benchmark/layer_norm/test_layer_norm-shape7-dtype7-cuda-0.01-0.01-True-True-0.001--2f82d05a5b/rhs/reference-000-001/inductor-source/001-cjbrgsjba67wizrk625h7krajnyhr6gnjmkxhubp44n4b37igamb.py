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


# kernel path: /data3/yelv.lh/ETV/benchmark/layer_norm/test_layer_norm-shape7-dtype7-cuda-0.01-0.01-True-True-0.001--2f82d05a5b/rhs/reference-000-001/torchinductor-cache/qt/cqtdfath4ggqkztpusnmxys7crbilzfyhokbmkxojlqivboh6xhq.py
# Topologically Sorted Source Nodes: [native_layer_norm_default], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   native_layer_norm_default => convert_element_type, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [3]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_native_layer_norm_0 = async_compile.triton('triton_poi_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 968}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = 3.0
    tmp9 = (tmp7 / tmp8)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/layer_norm/test_layer_norm-shape7-dtype7-cuda-0.01-0.01-True-True-0.001--2f82d05a5b/rhs/reference-000-001/torchinductor-cache/3n/c3nazqbxabj3vf7kcj6mzowgas42im6jtik5wuckuhftv2wjlvmw.py
# Topologically Sorted Source Nodes: [native_layer_norm_default], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   native_layer_norm_default => add, convert_element_type, convert_element_type_1, mul, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 0.001), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.float16), kwargs = {})
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_poi_fused_native_layer_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2178}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (1 + 3*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (2 + 3*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 - tmp2
    tmp7 = tmp6 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 - tmp2
    tmp11 = tmp10 * tmp10
    tmp12 = tmp7 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 - tmp2
    tmp16 = tmp15 * tmp15
    tmp17 = tmp12 + tmp16
    tmp18 = 3.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp3 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (11, 11, 1, 3), (33, 3, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((11, 11, 1, 1), (11, 1, 121, 121), torch.float32)
        # Topologically Sorted Source Nodes: [native_layer_norm_default], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_0.run(arg0_1, buf0, 121, stream=stream0)
        buf1 = empty_strided_cuda((11, 11, 1, 3), (33, 3, 3, 1), torch.float16)
        # Topologically Sorted Source Nodes: [native_layer_norm_default], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_1.run(arg0_1, buf0, buf1, 363, stream=stream0)
        del arg0_1
        del buf0
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((11, 11, 1, 3), (33, 3, 3, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
