
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34848, 'r0_': 3761406}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 1936
    XBLOCK: tl.constexpr = 1
    r0_numel = 471
    R0_BLOCK: tl.constexpr = 512
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 121)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 471*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 471*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = float("-inf")
    tmp5 = tmp3 == tmp4
    tmp6 = tmp5 == 0
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, False)
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp15 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, float("-inf"))
    tmp18 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp17, 0))
    tmp19 = tmp13 - tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp18, None)
    tl.store(out_ptr2 + (x3), tmp24, None)
