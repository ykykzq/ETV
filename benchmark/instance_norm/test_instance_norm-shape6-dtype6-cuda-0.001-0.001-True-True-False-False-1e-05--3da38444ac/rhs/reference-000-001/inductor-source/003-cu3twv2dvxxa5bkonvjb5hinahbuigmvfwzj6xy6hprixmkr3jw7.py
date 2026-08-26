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


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape6-dtype6-cuda-0.001-0.001-True-True-False-False-1e-05--3da38444ac/rhs/reference-000-001/torchinductor-cache/j2/cj2sjt5vff3xa5f5jf7wvszuyw3txfwbnvqzricewqm6d7477du4.py
# Topologically Sorted Source Nodes: [cudnn_batch_norm_default], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   cudnn_batch_norm_default => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 174
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 3.0
    tmp6 = (tmp4 / tmp5)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape6-dtype6-cuda-0.001-0.001-True-True-False-False-1e-05--3da38444ac/rhs/reference-000-001/torchinductor-cache/gn/cgn6on3glijdfu5z5tqyaqbszbzxabwj63hztav3tq5kxnq3jv4o.py
# Topologically Sorted Source Nodes: [mean, mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   mean_1 => mean_1
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %mean_1), kwargs = {})
triton_poi_fused_mean_1 = async_compile.triton('triton_poi_fused_mean_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr2', 'out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (x0), xmask)
    tmp22 = tl.load(in_ptr0 + (87 + 3*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (29 + x0), xmask)
    tmp26 = tl.load(in_ptr0 + (88 + 3*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (89 + 3*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (174 + 3*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr1 + (58 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (175 + 3*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (176 + 3*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (261 + 3*x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (87 + x0), xmask)
    tmp60 = tl.load(in_ptr0 + (262 + 3*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr0 + (263 + 3*x0), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr0 + (348 + 3*x0), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr1 + (116 + x0), xmask)
    tmp77 = tl.load(in_ptr0 + (349 + 3*x0), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (350 + 3*x0), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (435 + 3*x0), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr1 + (145 + x0), xmask)
    tmp94 = tl.load(in_ptr0 + (436 + 3*x0), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr0 + (437 + 3*x0), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 - tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 - tmp1
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = 3.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.5
    tmp15 = tmp13 * tmp14
    tmp16 = 0.1
    tmp17 = tmp15 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp24
    tmp27 = tmp26 - tmp23
    tmp28 = tmp27 * tmp27
    tmp29 = tmp25 + tmp28
    tmp31 = tmp30 - tmp23
    tmp32 = tmp31 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = (tmp33 / tmp12)
    tmp35 = tmp34 * tmp14
    tmp36 = tmp35 * tmp16
    tmp37 = tmp36 + tmp20
    tmp38 = tmp21 + tmp37
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 * tmp41
    tmp44 = tmp43 - tmp40
    tmp45 = tmp44 * tmp44
    tmp46 = tmp42 + tmp45
    tmp48 = tmp47 - tmp40
    tmp49 = tmp48 * tmp48
    tmp50 = tmp46 + tmp49
    tmp51 = (tmp50 / tmp12)
    tmp52 = tmp51 * tmp14
    tmp53 = tmp52 * tmp16
    tmp54 = tmp53 + tmp20
    tmp55 = tmp38 + tmp54
    tmp58 = tmp56 - tmp57
    tmp59 = tmp58 * tmp58
    tmp61 = tmp60 - tmp57
    tmp62 = tmp61 * tmp61
    tmp63 = tmp59 + tmp62
    tmp65 = tmp64 - tmp57
    tmp66 = tmp65 * tmp65
    tmp67 = tmp63 + tmp66
    tmp68 = (tmp67 / tmp12)
    tmp69 = tmp68 * tmp14
    tmp70 = tmp69 * tmp16
    tmp71 = tmp70 + tmp20
    tmp72 = tmp55 + tmp71
    tmp75 = tmp73 - tmp74
    tmp76 = tmp75 * tmp75
    tmp78 = tmp77 - tmp74
    tmp79 = tmp78 * tmp78
    tmp80 = tmp76 + tmp79
    tmp82 = tmp81 - tmp74
    tmp83 = tmp82 * tmp82
    tmp84 = tmp80 + tmp83
    tmp85 = (tmp84 / tmp12)
    tmp86 = tmp85 * tmp14
    tmp87 = tmp86 * tmp16
    tmp88 = tmp87 + tmp20
    tmp89 = tmp72 + tmp88
    tmp92 = tmp90 - tmp91
    tmp93 = tmp92 * tmp92
    tmp95 = tmp94 - tmp91
    tmp96 = tmp95 * tmp95
    tmp97 = tmp93 + tmp96
    tmp99 = tmp98 - tmp91
    tmp100 = tmp99 * tmp99
    tmp101 = tmp97 + tmp100
    tmp102 = (tmp101 / tmp12)
    tmp103 = tmp102 * tmp14
    tmp104 = tmp103 * tmp16
    tmp105 = tmp104 + tmp20
    tmp106 = tmp89 + tmp105
    tmp107 = 6.0
    tmp108 = (tmp106 / tmp107)
    tmp109 = tmp1 * tmp16
    tmp111 = tmp110 * tmp19
    tmp112 = tmp109 + tmp111
    tmp113 = tmp23 * tmp16
    tmp114 = tmp113 + tmp111
    tmp115 = tmp112 + tmp114
    tmp116 = tmp40 * tmp16
    tmp117 = tmp116 + tmp111
    tmp118 = tmp115 + tmp117
    tmp119 = tmp57 * tmp16
    tmp120 = tmp119 + tmp111
    tmp121 = tmp118 + tmp120
    tmp122 = tmp74 * tmp16
    tmp123 = tmp122 + tmp111
    tmp124 = tmp121 + tmp123
    tmp125 = tmp91 * tmp16
    tmp126 = tmp125 + tmp111
    tmp127 = tmp124 + tmp126
    tmp128 = (tmp127 / tmp107)
    tl.store(out_ptr2 + (x0), tmp128, xmask)
    tl.store(out_ptr3 + (x0), tmp108, xmask)
''', device_str='cuda')


# kernel path: /data3/yelv.lh/ETV/benchmark/instance_norm/test_instance_norm-shape6-dtype6-cuda-0.001-0.001-True-True-False-False-1e-05--3da38444ac/rhs/reference-000-001/torchinductor-cache/gf/cgfg3cjnreue44qsoh2pr4p4a6ypgcsi56fgcixixj35ceuh2qpc.py
# Topologically Sorted Source Nodes: [cudnn_batch_norm_default], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   cudnn_batch_norm_default => add, add_3, mul, mul_6, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 3*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 3*x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (((x2 // 3) % 29)), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (((x2 // 3) % 29)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp3 - tmp1
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 - tmp1
    tmp8 = tmp7 * tmp7
    tmp9 = tmp5 + tmp8
    tmp11 = tmp10 - tmp1
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 3.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp2 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (x2), tmp23, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (29, ), (1, ))
    assert_size_stride(arg1_1, (29, ), (1, ))
    assert_size_stride(arg2_1, (29, ), (1, ))
    assert_size_stride(arg3_1, (29, ), (1, ))
    assert_size_stride(arg4_1, (6, 29, 3, 1), (87, 3, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 174, 1, 1), (174, 1, 174, 174), torch.float32)
        # Topologically Sorted Source Nodes: [cudnn_batch_norm_default], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_0.run(arg4_1, buf0, 174, stream=stream0)
        # Topologically Sorted Source Nodes: [mean, mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_1.run(arg4_1, buf0, arg3_1, arg2_1, arg2_1, arg3_1, 29, stream=stream0)
        del arg2_1
        del arg3_1
        buf2 = empty_strided_cuda((1, 174, 3, 1), (522, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cudnn_batch_norm_default], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_2.run(arg4_1, buf0, arg0_1, arg1_1, buf2, 522, stream=stream0)
        del arg0_1
        del arg1_1
        del arg4_1
        del buf0
    return (reinterpret_tensor(buf2, (6, 29, 3, 1), (87, 3, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((6, 29, 3, 1), (87, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
