
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 96}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = 2*x0
    tmp6 = tmp5 >= tmp0
    tmp7 = tl.full([1], 12, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp4 & tmp9
    tmp11 = tl.load(in_ptr0 + (2*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 1 + 2*x0
    tmp17 = tmp16 >= tmp0
    tmp18 = tmp16 < tmp7
    tmp19 = tmp17 & tmp18
    tmp20 = tmp4 & tmp19
    tmp21 = tl.load(in_ptr0 + (1 + 2*x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp21
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = tmp25 + tmp15
    tmp27 = 2 + 2*x0
    tmp28 = tmp27 >= tmp0
    tmp29 = tmp27 < tmp7
    tmp30 = tmp28 & tmp29
    tmp31 = tmp4 & tmp30
    tmp32 = tl.load(in_ptr0 + (2 + 2*x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp33 * tmp32
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tmp36 + tmp26
    tmp38 = ((-2)*x0) + ((12) * ((12) <= (3 + 2*x0)) + (3 + 2*x0) * ((3 + 2*x0) < (12)))
    tmp39 = (tmp37 / tmp38)
    tmp40 = tl.full([1], 0, tl.int32)
    tmp41 = tmp40 < tmp39
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp39 < tmp40
    tmp44 = tmp43.to(tl.int8)
    tmp45 = tmp42 - tmp44
    tmp46 = tmp45.to(tmp39.dtype)
    tmp47 = tl_math.abs(tmp39)
    tmp48 = triton_helpers.maximum(tmp40, tmp47)
    tmp49 = tmp46 * tmp48
    tmp50 = 3.0
    tmp51 = tmp49 * tmp50
    tmp52 = 0.3333333333333333
    tmp53 = libdevice.pow(tmp51, tmp52)
    tl.store(in_out_ptr0 + (x2), tmp53, xmask)
