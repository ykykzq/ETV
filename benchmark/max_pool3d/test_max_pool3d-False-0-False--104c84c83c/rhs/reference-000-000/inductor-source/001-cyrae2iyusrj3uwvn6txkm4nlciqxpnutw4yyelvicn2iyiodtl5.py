
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 7)
    x2 = xindex // 28
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (120 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (121 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (128 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (129 + 2*x0 + 16*x1 + 240*x2), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
