
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 134178196}}
)
@triton.jit
def triton_red_fused__safe_softmax_add_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 23168
    r0_numel = 482
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 362)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 11584
    x6 = (xindex % 11584)
    _tmp14_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp14_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 482*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 482*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp2 = 0.0
        tmp3 = float("-inf")
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp5 = tmp0 + tmp4
        tmp6 = tmp5 == tmp3
        tmp7 = tmp6 == 0
        tmp8 = tmp7.to(tl.int64)
        tmp9 = (tmp8 != 0)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 | tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp13 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])

        _tmp14_max_next, _tmp14_sum_next = triton_helpers.online_softmax_combine(
            _tmp14_max, _tmp14_sum, tmp13, False
        )

        _tmp14_max = tl.where(r0_mask & xmask, _tmp14_max_next, _tmp14_max)
        _tmp14_sum = tl.where(r0_mask & xmask, _tmp14_sum_next, _tmp14_sum)
    tmp11 = triton_helpers.any(_tmp11.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp16, tmp17 = triton_helpers.online_softmax_reduce(
        _tmp14_max, _tmp14_sum, 1, False)
    tmp16 = tmp16[:, None]
    tmp17 = tmp17[:, None]
    tmp14 = tmp16
    tmp15 = tmp17
    x4 = xindex // 362
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp19 = tl.load(in_ptr0 + (r0_3 + 482*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r0_3 + 482*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp18 = tmp11 == 0
        tmp21 = 0.0
        tmp22 = float("-inf")
        tmp23 = tl.where(tmp20, tmp21, tmp22)
        tmp24 = tmp19 + tmp23
        tmp25 = tmp24 - tmp14
        tmp26 = tl_math.exp(tmp25)
        tmp27 = (tmp26 / tmp15)
        tmp28 = tl.where(tmp18, tmp21, tmp27)
        tl.store(out_ptr3 + (r0_3 + 482*x0 + 174496*x4), tmp28, r0_mask & xmask)
