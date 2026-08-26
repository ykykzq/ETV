
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 36878440}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8992
    r0_numel = 340
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 281)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 4496
    x6 = (xindex % 4496)
    _tmp13_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp13_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 340*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 340*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = float("-inf")
        tmp5 = tmp3 == tmp4
        tmp6 = tmp5 == 0
        tmp7 = tmp6.to(tl.int64)
        tmp8 = (tmp7 != 0)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 | tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])

        _tmp13_max_next, _tmp13_sum_next = triton_helpers.online_softmax_combine(
            _tmp13_max, _tmp13_sum, tmp12, False
        )

        _tmp13_max = tl.where(r0_mask & xmask, _tmp13_max_next, _tmp13_max)
        _tmp13_sum = tl.where(r0_mask & xmask, _tmp13_sum_next, _tmp13_sum)
    tmp10 = triton_helpers.any(_tmp10.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp15, tmp16 = triton_helpers.online_softmax_reduce(
        _tmp13_max, _tmp13_sum, 1, False)
    tmp15 = tmp15[:, None]
    tmp16 = tmp16[:, None]
    tmp13 = tmp15
    tmp14 = tmp16
    x5 = xindex // 281
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp18 = tl.load(in_ptr0 + (r0_3 + 340*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r0_3 + 340*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp10 == 0
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21 - tmp13
        tmp23 = tl_math.exp(tmp22)
        tmp24 = (tmp23 / tmp14)
        tmp25 = 0.0
        tmp26 = tl.where(tmp17, tmp25, tmp24)
        tl.store(out_ptr3 + (r0_3 + 340*x0 + 95552*x5), tmp26, r0_mask & xmask)
