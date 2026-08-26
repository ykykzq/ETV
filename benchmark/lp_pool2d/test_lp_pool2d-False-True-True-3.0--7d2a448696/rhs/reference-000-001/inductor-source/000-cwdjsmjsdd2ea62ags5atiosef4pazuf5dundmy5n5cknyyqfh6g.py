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


# kernel path: /data3/yelv.lh/ETV/benchmark/lp_pool2d/test_lp_pool2d-False-True-True-3.0--7d2a448696/rhs/reference-000-001/torchinductor-cache/wq/cwql7lrxr37evz2opxvmgr7bs3shb57gwfrwzuehyv7bddzioc6a.py
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
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [5, 2], [], [0, 0], True), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 10), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.3333333333333333), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 5)
    x0 = (xindex % 8)
    x2 = xindex // 40
    x3 = xindex
    tmp0 = 5*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 15, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (2*x0 + 75*x1 + 360*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 * tmp12
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = 1 + 2*x0
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp8
    tmp20 = tmp18 & tmp19
    tmp21 = tmp5 & tmp20
    tmp22 = tl.load(in_ptr0 + (1 + 2*x0 + 75*x1 + 360*x2), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp26 + tmp16
    tmp28 = 1 + 5*x1
    tmp29 = tmp28 >= tmp1
    tmp30 = tmp28 < tmp3
    tmp31 = tmp29 & tmp30
    tmp32 = tmp31 & tmp10
    tmp33 = tl.load(in_ptr0 + (15 + 2*x0 + 75*x1 + 360*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33 * tmp33
    tmp35 = tmp34 * tmp33
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp32, tmp35, tmp36)
    tmp38 = tmp37 + tmp27
    tmp39 = tmp31 & tmp20
    tmp40 = tl.load(in_ptr0 + (16 + 2*x0 + 75*x1 + 360*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp38
    tmp46 = 2 + 5*x1
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp3
    tmp49 = tmp47 & tmp48
    tmp50 = tmp49 & tmp10
    tmp51 = tl.load(in_ptr0 + (30 + 2*x0 + 75*x1 + 360*x2), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp50, tmp53, tmp54)
    tmp56 = tmp55 + tmp45
    tmp57 = tmp49 & tmp20
    tmp58 = tl.load(in_ptr0 + (31 + 2*x0 + 75*x1 + 360*x2), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 * tmp58
    tmp60 = tmp59 * tmp58
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp57, tmp60, tmp61)
    tmp63 = tmp62 + tmp56
    tmp64 = 3 + 5*x1
    tmp65 = tmp64 >= tmp1
    tmp66 = tmp64 < tmp3
    tmp67 = tmp65 & tmp66
    tmp68 = tmp67 & tmp10
    tmp69 = tl.load(in_ptr0 + (45 + 2*x0 + 75*x1 + 360*x2), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 * tmp69
    tmp71 = tmp70 * tmp69
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp68, tmp71, tmp72)
    tmp74 = tmp73 + tmp63
    tmp75 = tmp67 & tmp20
    tmp76 = tl.load(in_ptr0 + (46 + 2*x0 + 75*x1 + 360*x2), tmp75 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76 * tmp76
    tmp78 = tmp77 * tmp76
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp75, tmp78, tmp79)
    tmp81 = tmp80 + tmp74
    tmp82 = 4 + 5*x1
    tmp83 = tmp82 >= tmp1
    tmp84 = tmp82 < tmp3
    tmp85 = tmp83 & tmp84
    tmp86 = tmp85 & tmp10
    tmp87 = tl.load(in_ptr0 + (60 + 2*x0 + 75*x1 + 360*x2), tmp86 & xmask, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp87 * tmp87
    tmp89 = tmp88 * tmp87
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp86, tmp89, tmp90)
    tmp92 = tmp91 + tmp81
    tmp93 = tmp85 & tmp20
    tmp94 = tl.load(in_ptr0 + (61 + 2*x0 + 75*x1 + 360*x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp94 * tmp94
    tmp96 = tmp95 * tmp94
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp93, tmp96, tmp97)
    tmp99 = tmp98 + tmp92
    tmp100 = ((15) * ((15) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (15)))*((24) * ((24) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (24))) + ((-5)*x1*((15) * ((15) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (15)))) + ((-2)*x0*((24) * ((24) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (24)))) + 10*x0*x1
    tmp101 = (tmp99 / tmp100)
    tmp102 = tl.full([1], 0, tl.int32)
    tmp103 = tmp102 < tmp101
    tmp104 = tmp103.to(tl.int8)
    tmp105 = tmp101 < tmp102
    tmp106 = tmp105.to(tl.int8)
    tmp107 = tmp104 - tmp106
    tmp108 = tmp107.to(tmp101.dtype)
    tmp109 = tl_math.abs(tmp101)
    tmp110 = triton_helpers.maximum(tmp102, tmp109)
    tmp111 = tmp108 * tmp110
    tmp112 = 10.0
    tmp113 = tmp111 * tmp112
    tmp114 = 0.3333333333333333
    tmp115 = libdevice.pow(tmp113, tmp114)
    tl.store(in_out_ptr0 + (x3), tmp115, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 1, 24, 15), (360, 360, 15, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 1, 5, 8), (40, 120, 8, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (3, 1, 5, 8), (40, 40, 8, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_1, avg_pool2d, sign, abs_1, relu, mul, mul_1, pow_2], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0.run(buf1, arg0_1, 120, stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 1, 24, 15), (360, 360, 15, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
