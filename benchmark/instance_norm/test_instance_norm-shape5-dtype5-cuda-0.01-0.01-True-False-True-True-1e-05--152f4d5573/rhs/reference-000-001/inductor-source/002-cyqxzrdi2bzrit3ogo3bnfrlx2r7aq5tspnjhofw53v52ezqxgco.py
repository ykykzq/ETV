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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape5-dtype5-cuda-0.01-0.01-True-False-True-True-1e-05--152f4d5573/rhs/reference-000-001/torchinductor-cache/pu/cpuya7kxds2crrbf77pfuq24x443545nzdnglnvmyc4zohz5ia5h.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_training_default], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_training_default => convert_element_type_2, mul_1, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %unsqueeze), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.float16), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 176, 'x': 5544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 44
    xnumel = 21
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 21*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((y0 % 4)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + ((y0 % 4)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = tl.full([1, 1], 1, tl.int32)
    tmp11 = (tmp10 / tmp9)
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 21*y0), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape5-dtype5-cuda-0.01-0.01-True-False-True-True-1e-05--152f4d5573/rhs/reference-000-001/torchinductor-cache/6h/c6hqt5dynarhyspgwku5jthfphpf2cvxyjahf7dvmet4c7mwiisj.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_1, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %mean), kwargs = {})
triton_per_fused_mean_1 = async_compile.triton('triton_per_fused_mean_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_1', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mean_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 11
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
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 11.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, ), (1, ))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg2_1, (11, 4, 21), (84, 21, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, 44, 21), (924, 21, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_training_default], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_0.run(arg2_1, arg0_1, arg1_1, buf2, 44, 21, stream=stream0)
        del arg2_1
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_1.run(arg0_1, arg0_1, 4, 11, stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_1.run(arg1_1, arg1_1, 4, 11, stream=stream0)
        del arg1_1
    return (reinterpret_tensor(buf2, (11, 4, 21), (84, 21, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((11, 4, 21), (84, 21, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
