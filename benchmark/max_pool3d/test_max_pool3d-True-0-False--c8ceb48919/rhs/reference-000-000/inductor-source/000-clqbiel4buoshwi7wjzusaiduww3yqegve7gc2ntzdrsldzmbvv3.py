
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 6)
    x2 = xindex // 42
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (29 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (182 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (183 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (196 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (197 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (210 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (211 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (364 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (365 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (378 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (379 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (392 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (393 + 2*x0 + 28*x1 + 182*x2), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp32 = triton_helpers.maximum(tmp30, tmp31)
    tmp34 = triton_helpers.maximum(tmp32, tmp33)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
