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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape1-dtype1-cuda-0.01-0.01-True-True-False-False-1e-08--84e2024060/rhs/reference-000-000/torchinductor-cache/mx/cmxjww3dg7ooiot7wztz4rwb6q4soleyiohvu4rzk6s6lcvnh54p.py
# Topologically Sorted Source Nodes: [_native_batch_norm_legit_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
# Source node to ATen node mapping:
#   _native_batch_norm_legit_default => add, add_3, convert_element_type, convert_element_type_1, mul, mul_6, rsqrt, sub, var_mean
#   mean => mean
#   mean_1 => mean_1
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg4_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-08), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float16), kwargs = {})
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
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr2': '*fp16', 'out_ptr4': '*fp16', 'out_ptr6': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 6108}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 1018
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
    tmp36 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp37 = tl.broadcast_to(tmp36, [1])
    tmp50 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [1])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 1018, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp1 - tmp11
    tmp19 = 1018.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-08
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp24 * tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 0.1
    tmp35 = tmp11 * tmp34
    tmp38 = 0.9
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp35 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 1.0
    tmp45 = (tmp43 / tmp44)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = 1.0009832841691249
    tmp48 = tmp20 * tmp47
    tmp49 = tmp48 * tmp34
    tmp52 = tmp51 * tmp38
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp49 + tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = (tmp56 / tmp44)
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp33, r0_mask)
    tl.store(out_ptr4 + (tl.full([1], 0, tl.int32)), tmp46, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (1, ), (1, ))
    assert_size_stride(arg3_1, (1, ), (1, ))
    assert_size_stride(arg4_1, (1, 1, 1018), (1018, 1018, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 1, 1018), (1018, 1018, 1), torch.float16)
        # Topologically Sorted Source Nodes: [_native_batch_norm_legit_default, mean, mean_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_mean_0.run(arg4_1, arg0_1, arg1_1, arg2_1, arg3_1, buf3, arg2_1, arg3_1, 1, 1018, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((1, 1, 1018), (1018, 1018, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
