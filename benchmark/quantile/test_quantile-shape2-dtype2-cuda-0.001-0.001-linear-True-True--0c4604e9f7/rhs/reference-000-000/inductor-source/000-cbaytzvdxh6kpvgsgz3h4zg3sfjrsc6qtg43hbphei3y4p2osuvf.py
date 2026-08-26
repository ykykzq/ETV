
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_any_ceil_expand_gather_isnan_lerp_masked_fill_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_any_ceil_expand_gather_isnan_lerp_masked_fill_sub_0(in_ptr0, out_ptr1, xnumel, r0_numel):
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
    tmp9 = 143.46279907226562
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 - tmp12
    tmp14 = tl_math.abs(tmp13)
    tmp15 = 0.5
    tmp16 = tmp14 >= tmp15
    tmp17 = 1.0
    tmp18 = tmp13 - tmp17
    tmp19 = tl.where(tmp16, tmp18, tmp13)
    tmp20 = libdevice.ceil(tmp10)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tl.full([1], 989, tl.int32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp21 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp21)
    tl.device_assert((0 <= tmp25) & (tmp25 < 989), "index out of bounds: 0 <= tmp25 < 989")
    tmp27 = tl.load(in_ptr0 + (tmp25), None, eviction_policy='evict_last')
    tmp28 = tmp11 + tmp22
    tmp29 = tmp11 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp11)
    tl.device_assert((0 <= tmp30) & (tmp30 < 989), "index out of bounds: 0 <= tmp30 < 989")
    tmp32 = tl.load(in_ptr0 + (tmp30), None, eviction_policy='evict_last')
    tmp33 = tmp27 - tmp32
    tmp34 = tmp19 * tmp33
    tmp35 = tl.where(tmp16, tmp27, tmp32)
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp36, None)
