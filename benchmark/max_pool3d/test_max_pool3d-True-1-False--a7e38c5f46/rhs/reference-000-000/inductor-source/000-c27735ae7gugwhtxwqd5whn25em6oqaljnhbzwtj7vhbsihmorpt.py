
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 40) % 10)
    x1 = ((xindex // 5) % 8)
    x0 = (xindex % 5)
    x3 = xindex // 400
    x7 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 9, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 15, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = (-1) + 2*x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tmp11 < tmp3
    tmp14 = tmp12 & tmp13
    tmp15 = tmp5 & tmp10
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-145) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = 2*x0
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp18 < tmp3
    tmp21 = tmp19 & tmp20
    tmp22 = tmp15 & tmp21
    tmp23 = tl.load(in_ptr0 + ((-144) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp22 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp24 = triton_helpers.maximum(tmp17, tmp23)
    tmp25 = 2*x1
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp8
    tmp28 = tmp26 & tmp27
    tmp29 = tmp5 & tmp28
    tmp30 = tmp29 & tmp14
    tmp31 = tl.load(in_ptr0 + ((-136) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp24, tmp31)
    tmp33 = tmp29 & tmp21
    tmp34 = tl.load(in_ptr0 + ((-135) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = x2
    tmp37 = tmp36 >= tmp1
    tmp38 = tmp36 < tmp3
    tmp39 = tmp37 & tmp38
    tmp40 = tmp39 & tmp10
    tmp41 = tmp40 & tmp14
    tmp42 = tl.load(in_ptr0 + ((-10) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp41 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp43 = triton_helpers.maximum(tmp35, tmp42)
    tmp44 = tmp40 & tmp21
    tmp45 = tl.load(in_ptr0 + ((-9) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp44 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp46 = triton_helpers.maximum(tmp43, tmp45)
    tmp47 = tmp39 & tmp28
    tmp48 = tmp47 & tmp14
    tmp49 = tl.load(in_ptr0 + ((-1) + 2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp48 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp50 = triton_helpers.maximum(tmp46, tmp49)
    tmp51 = tmp47 & tmp21
    tmp52 = tl.load(in_ptr0 + (2*x0 + 18*x1 + 135*x2 + 1215*x3), tmp51 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp50, tmp52)
    tl.store(out_ptr0 + (x7), tmp53, xmask)
