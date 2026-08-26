
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 648
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
    tmp8 = tl.load(in_ptr1 + (0))
    tmp9 = tl.broadcast_to(tmp8, [1])
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = (tmp2 != 0)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, False)
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp6, 0))
    tmp10 = 647.0
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp7, tmp10, tmp11)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 - tmp14
    tmp16 = tl_math.abs(tmp15)
    tmp17 = 0.5
    tmp18 = tmp16 >= tmp17
    tmp19 = 1.0
    tmp20 = tmp15 - tmp19
    tmp21 = tl.where(tmp18, tmp20, tmp15)
    tmp22 = libdevice.ceil(tmp12)
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tl.full([1], 648, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert((0 <= tmp27) & (tmp27 < 648), "index out of bounds: 0 <= tmp27 < 648")
    tmp29 = tl.load(in_ptr0 + (tmp27), None, eviction_policy='evict_last')
    tmp30 = tmp13 + tmp24
    tmp31 = tmp13 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp13)
    tl.device_assert((0 <= tmp32) & (tmp32 < 648), "index out of bounds: 0 <= tmp32 < 648")
    tmp34 = tl.load(in_ptr0 + (tmp32), None, eviction_policy='evict_last')
    tmp35 = tmp29 - tmp34
    tmp36 = tmp21 * tmp35
    tmp37 = tl.where(tmp18, tmp29, tmp34)
    tmp38 = tmp36 + tmp37
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp38, None)
