
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = xindex // 7
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 21*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (7 + x0 + 21*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (14 + x0 + 21*x1), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tmp0 > tmp1
    tmp6 = tmp0 == tmp1
    tmp7 = tmp0 != tmp0
    tmp8 = tmp1 != tmp1
    tmp9 = tmp7 > tmp8
    tmp10 = tmp5 | tmp9
    tmp11 = tmp7 & tmp8
    tmp12 = tmp6 | tmp11
    tmp13 = tl.full([1], 0, tl.int64)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tmp12 & tmp15
    tmp17 = tmp10 | tmp16
    tmp18 = tl.where(tmp17, tmp0, tmp1)
    tmp19 = tl.where(tmp17, tmp13, tmp14)
    tmp20 = tmp18 > tmp3
    tmp21 = tmp18 == tmp3
    tmp22 = tmp18 != tmp18
    tmp23 = tmp3 != tmp3
    tmp24 = tmp22 > tmp23
    tmp25 = tmp20 | tmp24
    tmp26 = tmp22 & tmp23
    tmp27 = tmp21 | tmp26
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp19 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tmp25 | tmp30
    tmp32 = tl.where(tmp31, tmp18, tmp3)
    tmp33 = tl.where(tmp31, tmp19, tmp28)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp33, xmask)
