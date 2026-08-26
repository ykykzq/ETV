
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 360, 'r0_': 7920}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15
    r0_numel = 44
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 44*x0), r0_mask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 44, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 44.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-08
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp24 = 0.1
    tmp25 = tmp10 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 1.0
    tmp31 = (tmp29 / tmp30)
    tmp32 = 1.0232558139534884
    tmp33 = tmp19 * tmp32
    tmp34 = tmp33 * tmp24
    tmp36 = tmp35 * tmp27
    tmp37 = tmp34 + tmp36
    tmp38 = (tmp37 / tmp30)
    tl.store(out_ptr2 + (r0_1 + 44*x0), tmp23, r0_mask & xmask)
    tl.store(out_ptr4 + (x0), tmp31, xmask)
    tl.store(out_ptr6 + (x0), tmp38, xmask)
