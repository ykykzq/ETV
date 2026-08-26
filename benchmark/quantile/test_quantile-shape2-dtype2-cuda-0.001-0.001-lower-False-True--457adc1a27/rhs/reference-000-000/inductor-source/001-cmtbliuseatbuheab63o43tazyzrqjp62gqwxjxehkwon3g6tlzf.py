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


# kernel path: /data3/yelv.lh/ETV/benchmark/quantile/test_quantile-shape2-dtype2-cuda-0.001-0.001-lower-False-True--457adc1a27/rhs/reference-000-000/torchinductor-cache/72/c72gq36xwzj4hojpuqpa6jei2wc4q4yh5b3jsyejbsyhlce4ls3a.py
# Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, expand, floor_, _to_copy, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten.expand, aten.floor, aten._to_copy, aten.gather]
# Source node to ATen node mapping:
#   _to_copy => convert_element_type
#   any_1 => any_1
#   expand => full_default
#   floor_ => floor
#   gather => gather
#   isnan => isnan
#   masked_fill => full_default_1, where
# Graph fragment:
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%getitem,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%isnan, -1, True), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 988.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 936.4771728515625), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%any_1, %full_default_1, %full_default), kwargs = {})
#   %floor : [num_users=1] = call_function[target=torch.ops.aten.floor.default](args = (%where,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%getitem, -1, %convert_element_type), kwargs = {})
triton_per_fused__to_copy_any_expand_floor_gather_isnan_masked_fill_0 = async_compile.triton('triton_per_fused__to_copy_any_expand_floor_gather_isnan_masked_fill_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_any_expand_floor_gather_isnan_masked_fill_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_any_expand_floor_gather_isnan_masked_fill_0(in_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 989
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
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = (tmp2 != 0)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, False)
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp6, 0))
    tmp8 = 988.0
    tmp9 = 936.4771728515625
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = libdevice.floor(tmp10)
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tl.full([1], 989, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 989), "index out of bounds: 0 <= tmp16 < 989")
    tmp18 = tl.load(in_ptr0 + (tmp16), None, eviction_policy='evict_last')
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (43, 23), (23, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [sort_default], Original ATen: [aten.sort]
        buf0 = torch.ops.aten.sort.stable(reinterpret_tensor(arg0_1, (989, ), (1, ), 0), stable=False, dim=0, descending=False)
        del arg0_1
        buf1 = buf0[0]
        assert_size_stride(buf1, (989, ), (1, ), 'torch.ops.aten.sort.default')
        assert_alignment(buf1, 16, 'torch.ops.aten.sort.default')
        del buf0
        buf4 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [isnan, any_1, masked_fill, expand, floor_, _to_copy, gather], Original ATen: [aten.isnan, aten.any, aten.masked_fill, aten.expand, aten.floor, aten._to_copy, aten.gather]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_any_expand_floor_gather_isnan_masked_fill_0.run(buf1, buf4, 1, 989, stream=stream0)
        del buf1
    return (reinterpret_tensor(buf4, (), (), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((43, 23), (23, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
