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


# kernel path: /data3/yelv.lh/ETV/benchmark/addmv/test_addmv-950-943-325-dtype0-cuda-0.001-0.001--5ff5029b24/rhs/reference-000-000/torchinductor-cache/em/cem5hfaryxtapmzs2oxqw3anhbqxi2iphkn4otoke6fvuvzt7zfs.py
# Topologically Sorted Source Nodes: [addmv], Original ATen: [aten.addmv]
# Source node to ATen node mapping:
#   addmv => add, mul, mul_1, mul_2, sum_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %arg0_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, -1.2815177460872722), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, -0.5311862414354188), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
triton_per_fused_addmv_0 = async_compile.triton('triton_per_fused_addmv_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_addmv_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11400, 'r0_': 1236300}}
)
@triton.jit
def triton_per_fused_addmv_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel):
    xnumel = 950
    XBLOCK: tl.constexpr = 1
    r0_numel = 325
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 325*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = -1.2815177460872722
    tmp8 = tmp6 * tmp7
    tmp10 = -0.5311862414354188
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (325, ), (1, ))
    assert_size_stride(arg1_1, (950, 325), (325, 1))
    assert_size_stride(arg2_1, (950, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((950, ), (1, ), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [addmv], Original ATen: [aten.addmv]
        stream0 = get_raw_stream(0)
        triton_per_fused_addmv_0.run(buf1, arg1_1, arg0_1, arg2_1, 950, 325, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((950, 325), (325, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((950, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
