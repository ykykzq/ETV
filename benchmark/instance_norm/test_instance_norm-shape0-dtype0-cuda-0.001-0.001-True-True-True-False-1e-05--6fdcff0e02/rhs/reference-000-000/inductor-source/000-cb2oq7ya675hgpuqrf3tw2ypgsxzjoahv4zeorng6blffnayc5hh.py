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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape0-dtype0-cuda-0.001-0.001-True-True-True-False-1e-05--6fdcff0e02/rhs/reference-000-000/torchinductor-cache/ys/cys74wydaet5bys4jriceim2puzngwhp5shxe5qwvsuqf2rjdfvb.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_default => add, add_3, mul, rsqrt, sub, var_mean
#   mean => mean
#   mean_1 => mean_1
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg3_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %mean_1), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_mean_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_mean_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 4584}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 382
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [1])
    tmp39 = tl.load(in_ptr3 + (0))
    tmp40 = tl.broadcast_to(tmp39, [1])
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 382, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 382.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp26 = tmp23 + tmp25
    tmp27 = 0.1
    tmp28 = tmp10 * tmp27
    tmp31 = 0.9
    tmp32 = tmp30 * tmp31
    tmp33 = tmp28 + tmp32
    tmp34 = 1.0
    tmp35 = (tmp33 / tmp34)
    tmp36 = 1.0026246719160106
    tmp37 = tmp19 * tmp36
    tmp38 = tmp37 * tmp27
    tmp41 = tmp40 * tmp31
    tmp42 = tmp38 + tmp41
    tmp43 = (tmp42 / tmp34)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp26, r0_mask)
    tl.store(out_ptr4 + (tl.full([1], 0, tl.int32)), tmp35, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp43, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (1, ), (1, ))
    assert_size_stride(arg3_1, (1, 1, 382), (382, 382, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 1, 382), (382, 382, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_mean_0.run(arg3_1, arg0_1, arg1_1, arg2_1, buf3, arg1_1, arg2_1, 1, 382, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 382), (382, 382, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
