
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool3d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 171360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4284
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 84) % 17)
    x1 = ((xindex // 14) % 6)
    x0 = (xindex % 14)
    x3 = xindex // 1428
    x7 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 11, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = (-1) + x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tl.full([1], 13, tl.int64)
    tmp14 = tmp11 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 & tmp10
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + ((-157) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp17 & xmask, other=float("-inf"))
    tmp19 = x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp13
    tmp22 = tmp20 & tmp21
    tmp23 = tmp16 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-156) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp8
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tmp30 & tmp15
    tmp32 = tl.load(in_ptr0 + ((-144) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp25, tmp32)
    tmp34 = tmp30 & tmp22
    tmp35 = tl.load(in_ptr0 + ((-143) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp34 & xmask, other=float("-inf"))
    tmp36 = triton_helpers.maximum(tmp33, tmp35)
    tmp37 = x2
    tmp38 = tmp37 >= tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tmp38 & tmp39
    tmp41 = tmp40 & tmp10
    tmp42 = tmp41 & tmp15
    tmp43 = tl.load(in_ptr0 + ((-14) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp42 & xmask, other=float("-inf"))
    tmp44 = triton_helpers.maximum(tmp36, tmp43)
    tmp45 = tmp41 & tmp22
    tmp46 = tl.load(in_ptr0 + ((-13) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp45 & xmask, other=float("-inf"))
    tmp47 = triton_helpers.maximum(tmp44, tmp46)
    tmp48 = tmp40 & tmp29
    tmp49 = tmp48 & tmp15
    tmp50 = tl.load(in_ptr0 + ((-1) + x0 + 26*x1 + 143*x2 + 2288*x3), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp47, tmp50)
    tmp52 = tmp48 & tmp22
    tmp53 = tl.load(in_ptr0 + (x0 + 26*x1 + 143*x2 + 2288*x3), tmp52 & xmask, other=float("-inf"))
    tmp54 = triton_helpers.maximum(tmp51, tmp53)
    tl.store(out_ptr0 + (x7), tmp54, xmask)
