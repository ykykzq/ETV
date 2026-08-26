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


# kernel path: /data3/yelv.lh/ETV/benchmark/lp_pool2d/test_lp_pool2d-False-False-False-3.0--976f7ef7e1/rhs/reference-000-001/torchinductor-cache/4i/c4itfvq66252eop7j2ewivclb727ufl33gbt6njiow5wokpejb7o.py
# Topologically Sorted Source Nodes: [pow_1, avg_pool2d, sign, abs_1, relu, mul, mul_1, pow_2], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   avg_pool2d => avg_pool2d
#   mul => mul
#   mul_1 => mul_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   relu => relu
#   sign => sign
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 3.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [4, 4]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 16), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.3333333333333333), kwargs = {})
triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0 = async_compile.triton('triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (6 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (7 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (8 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (9 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (12 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (13 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (14 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (15 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (18 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (19 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (20 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (21 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4 * tmp3
    tmp6 = tmp5 + tmp2
    tmp8 = tmp7 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 + tmp6
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tmp13 + tmp10
    tmp16 = tmp15 * tmp15
    tmp17 = tmp16 * tmp15
    tmp18 = tmp17 + tmp14
    tmp20 = tmp19 * tmp19
    tmp21 = tmp20 * tmp19
    tmp22 = tmp21 + tmp18
    tmp24 = tmp23 * tmp23
    tmp25 = tmp24 * tmp23
    tmp26 = tmp25 + tmp22
    tmp28 = tmp27 * tmp27
    tmp29 = tmp28 * tmp27
    tmp30 = tmp29 + tmp26
    tmp32 = tmp31 * tmp31
    tmp33 = tmp32 * tmp31
    tmp34 = tmp33 + tmp30
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 * tmp35
    tmp38 = tmp37 + tmp34
    tmp40 = tmp39 * tmp39
    tmp41 = tmp40 * tmp39
    tmp42 = tmp41 + tmp38
    tmp44 = tmp43 * tmp43
    tmp45 = tmp44 * tmp43
    tmp46 = tmp45 + tmp42
    tmp48 = tmp47 * tmp47
    tmp49 = tmp48 * tmp47
    tmp50 = tmp49 + tmp46
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tmp53 + tmp50
    tmp56 = tmp55 * tmp55
    tmp57 = tmp56 * tmp55
    tmp58 = tmp57 + tmp54
    tmp60 = tmp59 * tmp59
    tmp61 = tmp60 * tmp59
    tmp62 = tmp61 + tmp58
    tmp63 = 0.0625
    tmp64 = tmp62 * tmp63
    tmp65 = tl.full([1], 0, tl.int32)
    tmp66 = tmp65 < tmp64
    tmp67 = tmp66.to(tl.int8)
    tmp68 = tmp64 < tmp65
    tmp69 = tmp68.to(tl.int8)
    tmp70 = tmp67 - tmp69
    tmp71 = tmp70.to(tmp64.dtype)
    tmp72 = tl_math.abs(tmp64)
    tmp73 = triton_helpers.maximum(tmp65, tmp72)
    tmp74 = tmp71 * tmp73
    tmp75 = 16.0
    tmp76 = tmp74 * tmp75
    tmp77 = 0.3333333333333333
    tmp78 = libdevice.pow(tmp76, tmp77)
    tl.store(in_out_ptr0 + (x2), tmp78, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 23, 6), (414, 138, 6, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 3, 5, 1), (15, 5, 1, 30), torch.float32)
        buf1 = reinterpret_tensor(buf0, (2, 3, 5, 1), (15, 5, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_1, avg_pool2d, sign, abs_1, relu, mul, mul_1, pow_2], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0.run(buf1, arg0_1, 30, stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 23, 6), (414, 138, 6, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
