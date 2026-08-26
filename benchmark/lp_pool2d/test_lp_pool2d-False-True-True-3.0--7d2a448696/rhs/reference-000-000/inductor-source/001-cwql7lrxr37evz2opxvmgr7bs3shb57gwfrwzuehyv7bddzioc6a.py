
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 5)
    x0 = (xindex % 8)
    x2 = xindex // 40
    x3 = xindex
    tmp0 = 5*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 15, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (2*x0 + 75*x1 + 360*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 * tmp12
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = 1 + 2*x0
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp8
    tmp20 = tmp18 & tmp19
    tmp21 = tmp5 & tmp20
    tmp22 = tl.load(in_ptr0 + (1 + 2*x0 + 75*x1 + 360*x2), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp26 + tmp16
    tmp28 = 1 + 5*x1
    tmp29 = tmp28 >= tmp1
    tmp30 = tmp28 < tmp3
    tmp31 = tmp29 & tmp30
    tmp32 = tmp31 & tmp10
    tmp33 = tl.load(in_ptr0 + (15 + 2*x0 + 75*x1 + 360*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33 * tmp33
    tmp35 = tmp34 * tmp33
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp32, tmp35, tmp36)
    tmp38 = tmp37 + tmp27
    tmp39 = tmp31 & tmp20
    tmp40 = tl.load(in_ptr0 + (16 + 2*x0 + 75*x1 + 360*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp38
    tmp46 = 2 + 5*x1
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp3
    tmp49 = tmp47 & tmp48
    tmp50 = tmp49 & tmp10
    tmp51 = tl.load(in_ptr0 + (30 + 2*x0 + 75*x1 + 360*x2), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp50, tmp53, tmp54)
    tmp56 = tmp55 + tmp45
    tmp57 = tmp49 & tmp20
    tmp58 = tl.load(in_ptr0 + (31 + 2*x0 + 75*x1 + 360*x2), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 * tmp58
    tmp60 = tmp59 * tmp58
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp57, tmp60, tmp61)
    tmp63 = tmp62 + tmp56
    tmp64 = 3 + 5*x1
    tmp65 = tmp64 >= tmp1
    tmp66 = tmp64 < tmp3
    tmp67 = tmp65 & tmp66
    tmp68 = tmp67 & tmp10
    tmp69 = tl.load(in_ptr0 + (45 + 2*x0 + 75*x1 + 360*x2), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 * tmp69
    tmp71 = tmp70 * tmp69
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp68, tmp71, tmp72)
    tmp74 = tmp73 + tmp63
    tmp75 = tmp67 & tmp20
    tmp76 = tl.load(in_ptr0 + (46 + 2*x0 + 75*x1 + 360*x2), tmp75 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76 * tmp76
    tmp78 = tmp77 * tmp76
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp75, tmp78, tmp79)
    tmp81 = tmp80 + tmp74
    tmp82 = 4 + 5*x1
    tmp83 = tmp82 >= tmp1
    tmp84 = tmp82 < tmp3
    tmp85 = tmp83 & tmp84
    tmp86 = tmp85 & tmp10
    tmp87 = tl.load(in_ptr0 + (60 + 2*x0 + 75*x1 + 360*x2), tmp86 & xmask, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp87 * tmp87
    tmp89 = tmp88 * tmp87
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp86, tmp89, tmp90)
    tmp92 = tmp91 + tmp81
    tmp93 = tmp85 & tmp20
    tmp94 = tl.load(in_ptr0 + (61 + 2*x0 + 75*x1 + 360*x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp94 * tmp94
    tmp96 = tmp95 * tmp94
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp93, tmp96, tmp97)
    tmp99 = tmp98 + tmp92
    tmp100 = ((15) * ((15) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (15)))*((24) * ((24) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (24))) + ((-5)*x1*((15) * ((15) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (15)))) + ((-2)*x0*((24) * ((24) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (24)))) + 10*x0*x1
    tmp101 = (tmp99 / tmp100)
    tmp102 = tl.full([1], 0, tl.int32)
    tmp103 = tmp102 < tmp101
    tmp104 = tmp103.to(tl.int8)
    tmp105 = tmp101 < tmp102
    tmp106 = tmp105.to(tl.int8)
    tmp107 = tmp104 - tmp106
    tmp108 = tmp107.to(tmp101.dtype)
    tmp109 = tl_math.abs(tmp101)
    tmp110 = triton_helpers.maximum(tmp102, tmp109)
    tmp111 = tmp108 * tmp110
    tmp112 = 10.0
    tmp113 = tmp111 * tmp112
    tmp114 = 0.3333333333333333
    tmp115 = libdevice.pow(tmp113, tmp114)
    tl.store(in_out_ptr0 + (x3), tmp115, xmask)
