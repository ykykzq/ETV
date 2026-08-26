
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 1610}}
)
@triton.jit
def triton_per_fused_mean_0(in_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 805
    R0_BLOCK: tl.constexpr = 1024
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 805.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp8, None)
