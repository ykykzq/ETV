
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 20873216}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_2(in_out_ptr0, in_ptr0, xnumel, r0_numel):
    xnumel = 6656
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 416)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (r0_2 + 256*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = float("-inf")
    tmp4 = tmp2 == tmp3
    tmp5 = tmp4 == 0
    tmp6 = tmp5.to(tl.int64)
    tmp7 = (tmp6 != 0)
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp13 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp15 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp13, 0))
    tmp16 = tmp11 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tmp10 == 0
    tmp22 = tmp2 - tmp15
    tmp23 = tl_math.exp(tmp22)
    tmp24 = (tmp23 / tmp20)
    tmp25 = 0.0
    tmp26 = tl.where(tmp21, tmp25, tmp24)
    tl.store(in_out_ptr0 + (r0_2 + 256*x3), tmp26, None)
