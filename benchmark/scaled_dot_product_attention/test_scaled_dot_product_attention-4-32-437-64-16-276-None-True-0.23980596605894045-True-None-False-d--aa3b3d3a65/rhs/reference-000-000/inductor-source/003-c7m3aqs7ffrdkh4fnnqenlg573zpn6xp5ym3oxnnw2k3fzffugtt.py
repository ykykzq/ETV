
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 185260032}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 55936
    r0_numel = 276
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 437)
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 13984
    x6 = (xindex % 13984)
    _tmp18_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp18_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 276*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp16 = _tmp15 | tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp17 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])

        _tmp18_max_next, _tmp18_sum_next = triton_helpers.online_softmax_combine(
            _tmp18_max, _tmp18_sum, tmp17, False
        )

        _tmp18_max = tl.where(r0_mask & xmask, _tmp18_max_next, _tmp18_max)
        _tmp18_sum = tl.where(r0_mask & xmask, _tmp18_sum_next, _tmp18_sum)
    tmp15 = triton_helpers.any(_tmp15.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp20, tmp21 = triton_helpers.online_softmax_reduce(
        _tmp18_max, _tmp18_sum, 1, False)
    tmp20 = tmp20[:, None]
    tmp21 = tmp21[:, None]
    tmp18 = tmp20
    tmp19 = tmp21
    x4 = xindex // 437
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp23 = tl.load(in_ptr0 + (r0_3 + 276*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp15 == 0
        tmp24 = r0_3 + ((-1)*x0)
        tmp25 = tl.full([1, 1], 0, tl.int64)
        tmp26 = tmp24 <= tmp25
        tmp27 = tl.full([1, 1], True, tl.int1)
        tmp28 = tmp26 & tmp27
        tmp29 = 0.0
        tmp30 = float("-inf")
        tmp31 = tl.where(tmp28, tmp29, tmp30)
        tmp32 = tmp23 + tmp31
        tmp33 = tmp32 - tmp18
        tmp34 = tl_math.exp(tmp33)
        tmp35 = (tmp34 / tmp19)
        tmp36 = tl.where(tmp22, tmp29, tmp35)
        tl.store(out_ptr3 + (r0_3 + 276*x0 + 120640*x4), tmp36, r0_mask & xmask)
