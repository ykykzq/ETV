
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1897665}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1696
    r0_numel = 93
    R0_BLOCK: tl.constexpr = 128
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 53)
    x1 = xindex // 53
    tmp0 = tl.load(in_ptr0 + (r0_2 + 93*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 93*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp2 = 0.0
    tmp3 = float("-inf")
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tmp0 + tmp4
    tmp6 = tmp5 == tmp3
    tmp7 = tmp6 == 0
    tmp8 = tmp7.to(tl.int64)
    tmp9 = (tmp8 != 0)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, False)
    tmp13 = triton_helpers.any(tmp12, 1)[:, None]
    tmp14 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, float("-inf"))
    tmp19 = triton_helpers.max2(tmp18, 1)[:, None]
    tmp20 = tmp14 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp13 == 0
    tmp27 = tmp5 - tmp19
    tmp28 = tl_math.exp(tmp27)
    tmp29 = (tmp28 / tmp25)
    tmp30 = tl.where(tmp26, tmp2, tmp29)
    tl.store(out_ptr3 + (r0_2 + 93*x0 + 4960*x1), tmp30, r0_mask & xmask)
