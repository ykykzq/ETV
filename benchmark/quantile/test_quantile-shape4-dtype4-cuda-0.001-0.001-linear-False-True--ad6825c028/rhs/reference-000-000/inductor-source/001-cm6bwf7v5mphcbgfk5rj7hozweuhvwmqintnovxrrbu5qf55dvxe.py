
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ceil_expand_gather_lerp_masked_fill_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ceil_expand_gather_lerp_masked_fill_sub_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = 399.0
    tmp3 = 222.51876831054688
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 - tmp6
    tmp8 = tl_math.abs(tmp7)
    tmp9 = 0.5
    tmp10 = tmp8 >= tmp9
    tmp11 = 1.0
    tmp12 = tmp7 - tmp11
    tmp13 = tl.where(tmp10, tmp12, tmp7)
    tmp14 = libdevice.ceil(tmp4)
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tl.full([XBLOCK], 400, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert((0 <= tmp19) & (tmp19 < 400), "index out of bounds: 0 <= tmp19 < 400")
    tmp21 = tl.load(in_ptr1 + (tmp19), None, eviction_policy='evict_last')
    tmp22 = tmp5 + tmp16
    tmp23 = tmp5 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp5)
    tl.device_assert((0 <= tmp24) & (tmp24 < 400), "index out of bounds: 0 <= tmp24 < 400")
    tmp26 = tl.load(in_ptr1 + (tmp24), None, eviction_policy='evict_last')
    tmp27 = tmp21 - tmp26
    tmp28 = tmp13 * tmp27
    tmp29 = tl.where(tmp10, tmp21, tmp26)
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp30, None)
