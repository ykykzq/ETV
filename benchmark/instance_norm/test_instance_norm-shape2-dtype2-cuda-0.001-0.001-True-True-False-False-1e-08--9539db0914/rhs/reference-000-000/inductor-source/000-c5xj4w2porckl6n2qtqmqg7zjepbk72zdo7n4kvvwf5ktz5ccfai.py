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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape2-dtype2-cuda-0.001-0.001-True-True-False-False-1e-08--9539db0914/rhs/reference-000-000/torchinductor-cache/gv/cgvxypd4hmbef6uziv6dlz3wlnvtvurffnnt2zdoonpu2545xamo.py
# Topologically Sorted Source Nodes: [cudnn_batch_norm_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
# Source node to ATen node mapping:
#   cudnn_batch_norm_default => add, add_3, mul, mul_6, rsqrt, sub, var_mean
#   mean => mean
#   mean_1 => mean_1
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg4_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg4_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-08), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %mean_1), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_mean_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_mean_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 480, 'r0_': 7920}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15
    r0_numel = 44
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 44*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 44, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 44.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-08
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = 0.1
    tmp29 = tmp10 * tmp28
    tmp31 = 0.9
    tmp32 = tmp30 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = 1.0
    tmp35 = (tmp33 / tmp34)
    tmp36 = 1.0232558139534884
    tmp37 = tmp19 * tmp36
    tmp38 = tmp37 * tmp28
    tmp40 = tmp39 * tmp31
    tmp41 = tmp38 + tmp40
    tmp42 = (tmp41 / tmp34)
    tl.store(out_ptr2 + (r0_1 + 44*x0), tmp27, r0_mask & xmask)
    tl.store(out_ptr4 + (x0), tmp35, xmask)
    tl.store(out_ptr6 + (x0), tmp42, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (15, ), (1, ))
    assert_size_stride(arg1_1, (15, ), (1, ))
    assert_size_stride(arg2_1, (15, ), (1, ))
    assert_size_stride(arg3_1, (15, ), (1, ))
    assert_size_stride(arg4_1, (1, 15, 44), (660, 44, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 15, 44), (660, 44, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cudnn_batch_norm_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_mean_0.run(arg4_1, arg0_1, arg1_1, arg2_1, arg3_1, buf3, arg2_1, arg3_1, 15, 44, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 15, 44), (660, 44, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
