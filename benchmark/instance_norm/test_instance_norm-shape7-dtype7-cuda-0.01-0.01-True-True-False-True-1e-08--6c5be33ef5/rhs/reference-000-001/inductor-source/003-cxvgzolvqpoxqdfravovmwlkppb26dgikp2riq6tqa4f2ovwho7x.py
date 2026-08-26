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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape7-dtype7-cuda-0.01-0.01-True-True-False-True-1e-08--6c5be33ef5/rhs/reference-000-001/torchinductor-cache/mm/cmmmlajuaim5ghkco27nrijb5vcv26pditbjinyyj22om5yunpgs.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_default], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_default => add, convert_element_type, convert_element_type_1, mul, mul_6, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-08), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.float16), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 216, 'r0_': 4032}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12
    r0_numel = 56
    R0_BLOCK: tl.constexpr = 64
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 56*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + ((x0 % 4)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 56, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp1 - tmp11
    tmp19 = 56.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-08
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 56*x0), tmp28, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape7-dtype7-cuda-0.01-0.01-True-True-False-True-1e-08--6c5be33ef5/rhs/reference-000-001/torchinductor-cache/us/cusjckoo5hkzopiaulsxoupck7chlgvfwbkrzjmxy6lnhcwhuovw.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %mean_1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp22 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp1 = 56.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1.018181818181818
    tmp4 = tmp2 * tmp3
    tmp5 = 0.1
    tmp6 = tmp4 * tmp5
    tmp8 = 0.9
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = (tmp14 / tmp1)
    tmp16 = tmp15 * tmp3
    tmp17 = tmp16 * tmp5
    tmp18 = tmp17 + tmp10
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 + tmp20
    tmp23 = (tmp22 / tmp1)
    tmp24 = tmp23 * tmp3
    tmp25 = tmp24 * tmp5
    tmp26 = tmp25 + tmp10
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp21 + tmp28
    tmp30 = 3.0
    tmp31 = (tmp29 / tmp30)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape7-dtype7-cuda-0.01-0.01-True-True-False-True-1e-08--6c5be33ef5/rhs/reference-000-001/torchinductor-cache/vw/cvw5ac5uhficjopxvlrsxcexju26askom3n6wfqq2cjqxi4fel52.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %mean), kwargs = {})
triton_poi_fused_mean_2 = async_compile.triton('triton_poi_fused_mean_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_2', 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_2(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp16 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp1 = 0.1
    tmp2 = tmp0 * tmp1
    tmp4 = 0.9
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp6
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 + tmp6
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp15 + tmp20
    tmp22 = 3.0
    tmp23 = (tmp21 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
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
        buf0 = empty_strided_cuda((1, 12, 1, 1), (12, 1, 1, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 12, 1, 1), (12, 1, 12, 12), torch.float32)
        buf4 = empty_strided_cuda((1, 12, 14, 4), (672, 56, 4, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_default], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_0.run(arg3_1, arg0_1, buf0, buf1, buf4, 12, 56, stream=stream0)
        del arg0_1
        del arg3_1
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(buf1, arg2_1, arg2_1, 4, stream=stream0)
        del arg2_1
        del buf1
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_2.run(buf0, arg1_1, arg1_1, 4, stream=stream0)
        del arg1_1
        del buf0
    return (reinterpret_tensor(buf4, (3, 4, 14, 4), (224, 56, 4, 1), 0), )


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
