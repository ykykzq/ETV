
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 38) % 56)
    x0 = (xindex % 38)
    x3 = xindex // 38
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 3*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + 3*x0 + 224*x3), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp12 = 3*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + 3*x0 + 224*x3), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 3*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + 3*x0 + 224*x3), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 3*x0 + 224*x3), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (3*x0 + 224*x3), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 3*x0 + 224*x3), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + 3*x0 + 224*x3), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + 3*x0 + 224*x3), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + 3*x0 + 224*x3), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
