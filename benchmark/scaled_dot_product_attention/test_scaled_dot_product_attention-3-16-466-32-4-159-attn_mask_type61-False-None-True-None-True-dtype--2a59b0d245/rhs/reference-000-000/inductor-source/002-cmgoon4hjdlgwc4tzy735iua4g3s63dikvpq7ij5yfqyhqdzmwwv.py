
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 42974520}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 22368
    r0_numel = 159
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 466)
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 7456
    x6 = (xindex % 7456)
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 159*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 159*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = float("-inf")
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tmp6 = tmp5.to(tl.int64)
        tmp7 = (tmp6 != 0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 | tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])

        _tmp12_max_next, _tmp12_sum_next = triton_helpers.online_softmax_combine(
            _tmp12_max, _tmp12_sum, tmp11, False
        )

        _tmp12_max = tl.where(r0_mask & xmask, _tmp12_max_next, _tmp12_max)
        _tmp12_sum = tl.where(r0_mask & xmask, _tmp12_sum_next, _tmp12_sum)
    tmp9 = triton_helpers.any(_tmp9.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp14, tmp15 = triton_helpers.online_softmax_reduce(
        _tmp12_max, _tmp12_sum, 1, False)
    tmp14 = tmp14[:, None]
    tmp15 = tmp15[:, None]
    tmp12 = tmp14
    tmp13 = tmp15
    x4 = xindex // 466
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_3 + 159*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_3 + 159*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp9 == 0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19 - tmp12
        tmp21 = tl_math.exp(tmp20)
        tmp22 = (tmp21 / tmp13)
        tmp23 = 0.0
        tmp24 = tl.where(tmp16, tmp23, tmp22)
        tl.store(out_ptr3 + (r0_3 + 159*x0 + 74112*x4), tmp24, r0_mask & xmask)
