
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 34466140}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6432
    r0_numel = 445
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 268)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 2144
    x6 = (xindex % 2144)
    _tmp15_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp15_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 445*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 445*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp2 = 0.0
        tmp3 = float("-inf")
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 + tmp5
        tmp7 = tmp6 == tmp3
        tmp8 = tmp7 == 0
        tmp9 = tmp8.to(tl.int64)
        tmp10 = (tmp9 != 0)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 | tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp15_max_next, _tmp15_sum_next = triton_helpers.online_softmax_combine(
            _tmp15_max, _tmp15_sum, tmp14, False
        )

        _tmp15_max = tl.where(r0_mask & xmask, _tmp15_max_next, _tmp15_max)
        _tmp15_sum = tl.where(r0_mask & xmask, _tmp15_sum_next, _tmp15_sum)
    tmp12 = triton_helpers.any(_tmp12.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp17, tmp18 = triton_helpers.online_softmax_reduce(
        _tmp15_max, _tmp15_sum, 1, False)
    tmp17 = tmp17[:, None]
    tmp18 = tmp18[:, None]
    tmp15 = tmp17
    tmp16 = tmp18
    x4 = xindex // 268
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp20 = tl.load(in_ptr0 + (r0_3 + 445*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr1 + (r0_3 + 445*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp19 = tmp12 == 0
        tmp22 = 0.0
        tmp23 = float("-inf")
        tmp24 = tl.where(tmp21, tmp22, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp20 + tmp25
        tmp27 = tmp26 - tmp15
        tmp28 = tl_math.exp(tmp27)
        tmp29 = (tmp28 / tmp16)
        tmp30 = tl.where(tmp19, tmp22, tmp29)
        tl.store(out_ptr3 + (r0_3 + 445*x0 + 119264*x4), tmp30, r0_mask & xmask)
