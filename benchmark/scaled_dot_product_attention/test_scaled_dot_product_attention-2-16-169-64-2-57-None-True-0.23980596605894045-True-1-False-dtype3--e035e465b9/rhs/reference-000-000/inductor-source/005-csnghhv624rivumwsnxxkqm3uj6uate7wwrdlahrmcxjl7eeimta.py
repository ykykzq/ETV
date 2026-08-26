
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 3699072}}
)
@triton.jit
def triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 5408
    r0_numel = 57
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
    r0_3 = r0_index
    x4 = xindex
    x0 = (xindex % 169)
    x2 = xindex // 2704
    x6 = (xindex % 2704)
    x5 = xindex // 169
    tmp0 = tl.load(in_ptr0 + (r0_3 + 57*x4), r0_mask & xmask, other=0.0)
    tmp1 = r0_3 + ((-1)*x0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = 0.0
    tmp7 = float("-inf")
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp9 == tmp7
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.int64)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, False)
    tmp17 = triton_helpers.any(tmp16, 1)[:, None]
    tmp18 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, float("-inf"))
    tmp23 = triton_helpers.max2(tmp22, 1)[:, None]
    tmp24 = tmp18 - tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp17 == 0
    tmp31 = tmp9 - tmp23
    tmp32 = tl_math.exp(tmp31)
    tmp33 = (tmp32 / tmp29)
    tmp34 = tl.where(tmp30, tmp6, tmp33)
    tl.store(out_ptr3 + (r0_3 + 57*x0 + 9664*x5), tmp34, r0_mask & xmask)
