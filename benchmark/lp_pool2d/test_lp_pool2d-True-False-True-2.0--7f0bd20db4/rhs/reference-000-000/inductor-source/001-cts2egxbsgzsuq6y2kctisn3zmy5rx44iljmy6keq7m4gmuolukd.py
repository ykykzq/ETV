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


# kernel path: /data3/yelv.lh/ETV/benchmark/lp_pool2d/test_lp_pool2d-True-False-True-2.0--7f0bd20db4/rhs/reference-000-000/torchinductor-cache/a2/ca2veyu764n6o6q3bddsrrbklswb3xkme5zmbers4u53mevjo7v4.py
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
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [4, 4], [2, 2], [0, 0], True), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 16), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.5), kwargs = {})
triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0 = async_compile.triton('triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 9, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (2*x0 + 18*x1 + 90*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = 1 + 2*x0
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp16 < tmp8
    tmp19 = tmp17 & tmp18
    tmp20 = tmp5 & tmp19
    tmp21 = tl.load(in_ptr0 + (1 + 2*x0 + 18*x1 + 90*x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp20, tmp22, tmp23)
    tmp25 = tmp24 + tmp15
    tmp26 = 2 + 2*x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp8
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + (2 + 2*x0 + 18*x1 + 90*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tmp34 + tmp25
    tmp36 = 3 + 2*x0
    tmp37 = tmp36 >= tmp1
    tmp38 = tmp36 < tmp8
    tmp39 = tmp37 & tmp38
    tmp40 = tmp5 & tmp39
    tmp41 = tl.load(in_ptr0 + (3 + 2*x0 + 18*x1 + 90*x2), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp40, tmp42, tmp43)
    tmp45 = tmp44 + tmp35
    tmp46 = 1 + 2*x1
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp3
    tmp49 = tmp47 & tmp48
    tmp50 = tmp49 & tmp10
    tmp51 = tl.load(in_ptr0 + (9 + 2*x0 + 18*x1 + 90*x2), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 * tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp50, tmp52, tmp53)
    tmp55 = tmp54 + tmp45
    tmp56 = tmp49 & tmp19
    tmp57 = tl.load(in_ptr0 + (10 + 2*x0 + 18*x1 + 90*x2), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 * tmp57
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp56, tmp58, tmp59)
    tmp61 = tmp60 + tmp55
    tmp62 = tmp49 & tmp29
    tmp63 = tl.load(in_ptr0 + (11 + 2*x0 + 18*x1 + 90*x2), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp63 * tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp62, tmp64, tmp65)
    tmp67 = tmp66 + tmp61
    tmp68 = tmp49 & tmp39
    tmp69 = tl.load(in_ptr0 + (12 + 2*x0 + 18*x1 + 90*x2), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 * tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp68, tmp70, tmp71)
    tmp73 = tmp72 + tmp67
    tmp74 = 2 + 2*x1
    tmp75 = tmp74 >= tmp1
    tmp76 = tmp74 < tmp3
    tmp77 = tmp75 & tmp76
    tmp78 = tmp77 & tmp10
    tmp79 = tl.load(in_ptr0 + (18 + 2*x0 + 18*x1 + 90*x2), tmp78 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp79 * tmp79
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp78, tmp80, tmp81)
    tmp83 = tmp82 + tmp73
    tmp84 = tmp77 & tmp19
    tmp85 = tl.load(in_ptr0 + (19 + 2*x0 + 18*x1 + 90*x2), tmp84 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp85 * tmp85
    tmp87 = tl.full(tmp86.shape, 0.0, tmp86.dtype)
    tmp88 = tl.where(tmp84, tmp86, tmp87)
    tmp89 = tmp88 + tmp83
    tmp90 = tmp77 & tmp29
    tmp91 = tl.load(in_ptr0 + (20 + 2*x0 + 18*x1 + 90*x2), tmp90 & xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp91 * tmp91
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp90, tmp92, tmp93)
    tmp95 = tmp94 + tmp89
    tmp96 = tmp77 & tmp39
    tmp97 = tl.load(in_ptr0 + (21 + 2*x0 + 18*x1 + 90*x2), tmp96 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp97 * tmp97
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp96, tmp98, tmp99)
    tmp101 = tmp100 + tmp95
    tmp102 = 3 + 2*x1
    tmp103 = tmp102 >= tmp1
    tmp104 = tmp102 < tmp3
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp10
    tmp107 = tl.load(in_ptr0 + (27 + 2*x0 + 18*x1 + 90*x2), tmp106 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp107 * tmp107
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp101
    tmp112 = tmp105 & tmp19
    tmp113 = tl.load(in_ptr0 + (28 + 2*x0 + 18*x1 + 90*x2), tmp112 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp113 * tmp113
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp29
    tmp119 = tl.load(in_ptr0 + (29 + 2*x0 + 18*x1 + 90*x2), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp119 * tmp119
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp105 & tmp39
    tmp125 = tl.load(in_ptr0 + (30 + 2*x0 + 18*x1 + 90*x2), tmp124 & xmask, eviction_policy='evict_last', other=0.0)
    tmp126 = tmp125 * tmp125
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp124, tmp126, tmp127)
    tmp129 = tmp128 + tmp123
    tmp130 = ((9) * ((9) <= (4 + 2*x0)) + (4 + 2*x0) * ((4 + 2*x0) < (9)))*((10) * ((10) <= (4 + 2*x1)) + (4 + 2*x1) * ((4 + 2*x1) < (10))) + ((-2)*x0*((10) * ((10) <= (4 + 2*x1)) + (4 + 2*x1) * ((4 + 2*x1) < (10)))) + ((-2)*x1*((9) * ((9) <= (4 + 2*x0)) + (4 + 2*x0) * ((4 + 2*x0) < (9)))) + 4*x0*x1
    tmp131 = (tmp129 / tmp130)
    tmp132 = tl.full([1], 0, tl.int32)
    tmp133 = tmp132 < tmp131
    tmp134 = tmp133.to(tl.int8)
    tmp135 = tmp131 < tmp132
    tmp136 = tmp135.to(tl.int8)
    tmp137 = tmp134 - tmp136
    tmp138 = tmp137.to(tmp131.dtype)
    tmp139 = tl_math.abs(tmp131)
    tmp140 = triton_helpers.maximum(tmp132, tmp139)
    tmp141 = tmp138 * tmp140
    tmp142 = 16.0
    tmp143 = tmp141 * tmp142
    tmp144 = libdevice.sqrt(tmp143)
    tl.store(in_out_ptr0 + (x3), tmp144, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 10, 9), (270, 90, 9, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 3, 4, 4), (48, 16, 4, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_1, avg_pool2d, sign, abs_1, relu, mul, mul_1, pow_2], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0.run(buf1, arg0_1, 96, stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 10, 9), (270, 90, 9, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
