
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'out_ptr4': '*fp16', 'out_ptr6': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 434, 'r0_': 2976}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 31
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp1 - tmp11
    tmp19 = 16.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 0.1
    tmp30 = tmp11 * tmp29
    tmp32 = 0.9
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp30 + tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 1.0
    tmp39 = (tmp37 / tmp38)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 1.0666666666666667
    tmp42 = tmp20 * tmp41
    tmp43 = tmp42 * tmp29
    tmp45 = tmp44 * tmp32
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48.to(tl.float32)
    tmp50 = (tmp49 / tmp38)
    tmp51 = tmp50.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 16*x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp40, xmask)
    tl.store(out_ptr6 + (x0), tmp51, xmask)
