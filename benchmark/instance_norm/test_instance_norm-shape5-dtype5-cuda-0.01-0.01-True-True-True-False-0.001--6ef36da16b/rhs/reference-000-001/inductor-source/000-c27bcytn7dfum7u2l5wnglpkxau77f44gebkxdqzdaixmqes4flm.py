
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_2', 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mean_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 11
    R0_BLOCK: tl.constexpr = 16
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_1), r0_mask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 21.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1.05
    tmp4 = tmp2 * tmp3
    tmp5 = 0.1
    tmp6 = tmp4 * tmp5
    tmp8 = 0.9
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 11.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
