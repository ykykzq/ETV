
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 10) % 11)
    x0 = (xindex % 10)
    x4 = xindex // 10
    x3 = xindex
    tmp0 = 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 22, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 19, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (2*x0 + 38*x4), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 * tmp12
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = 1 + 2*x0
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp8
    tmp20 = tmp18 & tmp19
    tmp21 = tmp5 & tmp20
    tmp22 = tl.load(in_ptr0 + (1 + 2*x0 + 38*x4), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp26 + tmp16
    tmp28 = 1 + 2*x1
    tmp29 = tmp28 >= tmp1
    tmp30 = tmp28 < tmp3
    tmp31 = tmp29 & tmp30
    tmp32 = tmp31 & tmp10
    tmp33 = tl.load(in_ptr0 + (19 + 2*x0 + 38*x4), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33 * tmp33
    tmp35 = tmp34 * tmp33
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp32, tmp35, tmp36)
    tmp38 = tmp37 + tmp27
    tmp39 = tmp31 & tmp20
    tmp40 = tl.load(in_ptr0 + (20 + 2*x0 + 38*x4), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp38
    tmp46 = ((19) * ((19) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (19)))*((22) * ((22) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (22))) + ((-2)*x0*((22) * ((22) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (22)))) + ((-2)*x1*((19) * ((19) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (19)))) + 4*x0*x1
    tmp47 = (tmp45 / tmp46)
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = tmp48 < tmp47
    tmp50 = tmp49.to(tl.int8)
    tmp51 = tmp47 < tmp48
    tmp52 = tmp51.to(tl.int8)
    tmp53 = tmp50 - tmp52
    tmp54 = tmp53.to(tmp47.dtype)
    tmp55 = tl_math.abs(tmp47)
    tmp56 = triton_helpers.maximum(tmp48, tmp55)
    tmp57 = tmp54 * tmp56
    tmp58 = 4.0
    tmp59 = tmp57 * tmp58
    tmp60 = 0.3333333333333333
    tmp61 = libdevice.pow(tmp59, tmp60)
    tl.store(in_out_ptr0 + (x3), tmp61, xmask)
