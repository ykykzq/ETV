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


# kernel path: /data3/yelv.lh/ETV/benchmark/batch_norm/test_batch_norm-shape7-dtype7-cuda-0.01-0.01-True-0.001--24386bd3a1/rhs/reference-000-001/torchinductor-cache/4h/c4heubjfg3k6o2yyurx4k54vytfuvpcwj5utym5xyoyvarfie4wb.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_no_stats => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 0.001), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_3), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float16), kwargs = {})
triton_per_fused__native_batch_norm_legit_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 5040}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 840
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp26 = tl.broadcast_to(tmp25, [R0_BLOCK])
    tmp29 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 840, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp1 - tmp11
    tmp19 = 840.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 0.001
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp24 * tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp33, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (5, 1, 24, 7), (168, 168, 7, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((5, 1, 24, 7), (168, 168, 7, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_no_stats], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0.run(arg2_1, arg1_1, arg0_1, buf3, 1, 840, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((5, 1, 24, 7), (168, 168, 7, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
