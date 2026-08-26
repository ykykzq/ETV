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


# kernel path: /data3/yelv.lh/ETV/benchmark/lp_pool2d/test_lp_pool2d-True-False-True-1.0--9ea77a959c/rhs/reference-000-000/torchinductor-cache/hd/chdkpz6ch5lx6wsayt5dfgc7yc3izjggzgmwm4qfbluhs3tgfmiw.py
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
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 1.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [4, 4], [4, 4], [0, 0], True), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 16), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 1.0), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 4)
    x0 = (xindex % 3)
    x2 = xindex // 12
    x3 = xindex
    tmp0 = 4*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 4*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 11, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (4*x0 + 44*x1 + 154*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1 + 4*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + 4*x0 + 44*x1 + 154*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 2 + 4*x0
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp8
    tmp23 = tmp21 & tmp22
    tmp24 = tmp5 & tmp23
    tmp25 = tl.load(in_ptr0 + (2 + 4*x0 + 44*x1 + 154*x2), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = 3 + 4*x0
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp8
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0 + 44*x1 + 154*x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = 1 + 4*x1
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp3
    tmp37 = tmp35 & tmp36
    tmp38 = tmp37 & tmp10
    tmp39 = tl.load(in_ptr0 + (11 + 4*x0 + 44*x1 + 154*x2), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39 + tmp33
    tmp41 = tmp37 & tmp16
    tmp42 = tl.load(in_ptr0 + (12 + 4*x0 + 44*x1 + 154*x2), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp42 + tmp40
    tmp44 = tmp37 & tmp23
    tmp45 = tl.load(in_ptr0 + (13 + 4*x0 + 44*x1 + 154*x2), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp43
    tmp47 = tmp37 & tmp30
    tmp48 = tl.load(in_ptr0 + (14 + 4*x0 + 44*x1 + 154*x2), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = 2 + 4*x1
    tmp51 = tmp50 >= tmp1
    tmp52 = tmp50 < tmp3
    tmp53 = tmp51 & tmp52
    tmp54 = tmp53 & tmp10
    tmp55 = tl.load(in_ptr0 + (22 + 4*x0 + 44*x1 + 154*x2), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp55 + tmp49
    tmp57 = tmp53 & tmp16
    tmp58 = tl.load(in_ptr0 + (23 + 4*x0 + 44*x1 + 154*x2), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp56
    tmp60 = tmp53 & tmp23
    tmp61 = tl.load(in_ptr0 + (24 + 4*x0 + 44*x1 + 154*x2), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp61 + tmp59
    tmp63 = tmp53 & tmp30
    tmp64 = tl.load(in_ptr0 + (25 + 4*x0 + 44*x1 + 154*x2), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64 + tmp62
    tmp66 = 3 + 4*x1
    tmp67 = tmp66 >= tmp1
    tmp68 = tmp66 < tmp3
    tmp69 = tmp67 & tmp68
    tmp70 = tmp69 & tmp10
    tmp71 = tl.load(in_ptr0 + (33 + 4*x0 + 44*x1 + 154*x2), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp71 + tmp65
    tmp73 = tmp69 & tmp16
    tmp74 = tl.load(in_ptr0 + (34 + 4*x0 + 44*x1 + 154*x2), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp74 + tmp72
    tmp76 = tmp69 & tmp23
    tmp77 = tl.load(in_ptr0 + (35 + 4*x0 + 44*x1 + 154*x2), tmp76 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp77 + tmp75
    tmp79 = tmp69 & tmp30
    tmp80 = tl.load(in_ptr0 + (36 + 4*x0 + 44*x1 + 154*x2), tmp79 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp78
    tmp82 = ((11) * ((11) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (11)))*((14) * ((14) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (14))) + ((-4)*x0*((14) * ((14) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (14)))) + ((-4)*x1*((11) * ((11) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (11)))) + 16*x0*x1
    tmp83 = (tmp81 / tmp82)
    tmp84 = tl.full([1], 0, tl.int32)
    tmp85 = tmp84 < tmp83
    tmp86 = tmp85.to(tl.int8)
    tmp87 = tmp83 < tmp84
    tmp88 = tmp87.to(tl.int8)
    tmp89 = tmp86 - tmp88
    tmp90 = tmp89.to(tmp83.dtype)
    tmp91 = tl_math.abs(tmp83)
    tmp92 = triton_helpers.maximum(tmp84, tmp91)
    tmp93 = tmp90 * tmp92
    tmp94 = 16.0
    tmp95 = tmp93 * tmp94
    tl.store(in_out_ptr0 + (x3), tmp95, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 1, 14, 11), (154, 154, 11, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 1, 4, 3), (12, 24, 3, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (2, 1, 4, 3), (12, 12, 3, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_1, avg_pool2d, sign, abs_1, relu, mul, mul_1, pow_2], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0.run(buf1, arg0_1, 24, stream=stream0)
        del arg0_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 1, 14, 11), (154, 154, 11, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
