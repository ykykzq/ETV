
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 28) % 8)
    x1 = ((xindex // 7) % 4)
    x0 = (xindex % 7)
    x3 = xindex // 224
    x4 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 15, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 4*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 16, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = 2*x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tl.full([1], 13, tl.int64)
    tmp14 = tmp11 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 & tmp10
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + (2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp19 * tmp18
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = 1 + 2*x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp13
    tmp26 = tmp24 & tmp25
    tmp27 = tmp16 & tmp26
    tmp28 = tl.load(in_ptr0 + (1 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp28 * tmp28
    tmp30 = tmp29 * tmp28
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tmp32 + tmp22
    tmp34 = 1 + 4*x1
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp8
    tmp37 = tmp35 & tmp36
    tmp38 = tmp5 & tmp37
    tmp39 = tmp38 & tmp15
    tmp40 = tl.load(in_ptr0 + (13 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp33
    tmp46 = tmp38 & tmp26
    tmp47 = tl.load(in_ptr0 + (14 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 * tmp47
    tmp49 = tmp48 * tmp47
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tmp51 + tmp45
    tmp53 = 2 + 4*x1
    tmp54 = tmp53 >= tmp1
    tmp55 = tmp53 < tmp8
    tmp56 = tmp54 & tmp55
    tmp57 = tmp5 & tmp56
    tmp58 = tmp57 & tmp15
    tmp59 = tl.load(in_ptr0 + (26 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp60 * tmp59
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tmp63 + tmp52
    tmp65 = tmp57 & tmp26
    tmp66 = tl.load(in_ptr0 + (27 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp66 * tmp66
    tmp68 = tmp67 * tmp66
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp65, tmp68, tmp69)
    tmp71 = tmp70 + tmp64
    tmp72 = 3 + 4*x1
    tmp73 = tmp72 >= tmp1
    tmp74 = tmp72 < tmp8
    tmp75 = tmp73 & tmp74
    tmp76 = tmp5 & tmp75
    tmp77 = tmp76 & tmp15
    tmp78 = tl.load(in_ptr0 + (39 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 * tmp78
    tmp80 = tmp79 * tmp78
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp77, tmp80, tmp81)
    tmp83 = tmp82 + tmp71
    tmp84 = tmp76 & tmp26
    tmp85 = tl.load(in_ptr0 + (40 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp84 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp85 * tmp85
    tmp87 = tmp86 * tmp85
    tmp88 = tl.full(tmp87.shape, 0.0, tmp87.dtype)
    tmp89 = tl.where(tmp84, tmp87, tmp88)
    tmp90 = tmp89 + tmp83
    tmp91 = 1 + 2*x2
    tmp92 = tmp91 >= tmp1
    tmp93 = tmp91 < tmp3
    tmp94 = tmp92 & tmp93
    tmp95 = tmp94 & tmp10
    tmp96 = tmp95 & tmp15
    tmp97 = tl.load(in_ptr0 + (208 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp96 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp97 * tmp97
    tmp99 = tmp98 * tmp97
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp96, tmp99, tmp100)
    tmp102 = tmp101 + tmp90
    tmp103 = tmp95 & tmp26
    tmp104 = tl.load(in_ptr0 + (209 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp103 & xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp104 * tmp104
    tmp106 = tmp105 * tmp104
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp103, tmp106, tmp107)
    tmp109 = tmp108 + tmp102
    tmp110 = tmp94 & tmp37
    tmp111 = tmp110 & tmp15
    tmp112 = tl.load(in_ptr0 + (221 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp111 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tmp112 * tmp112
    tmp114 = tmp113 * tmp112
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp111, tmp114, tmp115)
    tmp117 = tmp116 + tmp109
    tmp118 = tmp110 & tmp26
    tmp119 = tl.load(in_ptr0 + (222 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp118 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp119 * tmp119
    tmp121 = tmp120 * tmp119
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp118, tmp121, tmp122)
    tmp124 = tmp123 + tmp117
    tmp125 = tmp94 & tmp56
    tmp126 = tmp125 & tmp15
    tmp127 = tl.load(in_ptr0 + (234 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp126 & xmask, eviction_policy='evict_last', other=0.0)
    tmp128 = tmp127 * tmp127
    tmp129 = tmp128 * tmp127
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp126, tmp129, tmp130)
    tmp132 = tmp131 + tmp124
    tmp133 = tmp125 & tmp26
    tmp134 = tl.load(in_ptr0 + (235 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp133 & xmask, eviction_policy='evict_last', other=0.0)
    tmp135 = tmp134 * tmp134
    tmp136 = tmp135 * tmp134
    tmp137 = tl.full(tmp136.shape, 0.0, tmp136.dtype)
    tmp138 = tl.where(tmp133, tmp136, tmp137)
    tmp139 = tmp138 + tmp132
    tmp140 = tmp94 & tmp75
    tmp141 = tmp140 & tmp15
    tmp142 = tl.load(in_ptr0 + (247 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp141 & xmask, eviction_policy='evict_last', other=0.0)
    tmp143 = tmp142 * tmp142
    tmp144 = tmp143 * tmp142
    tmp145 = tl.full(tmp144.shape, 0.0, tmp144.dtype)
    tmp146 = tl.where(tmp141, tmp144, tmp145)
    tmp147 = tmp146 + tmp139
    tmp148 = tmp140 & tmp26
    tmp149 = tl.load(in_ptr0 + (248 + 2*x0 + 52*x1 + 416*x2 + 3120*x3), tmp148 & xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp149 * tmp149
    tmp151 = tmp150 * tmp149
    tmp152 = tl.full(tmp151.shape, 0.0, tmp151.dtype)
    tmp153 = tl.where(tmp148, tmp151, tmp152)
    tmp154 = tmp153 + tmp147
    tmp155 = ((13) * ((13) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (13)))*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15)))*((16) * ((16) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (16))) + ((-16)*x0*x1*x2) + ((-4)*x1*((13) * ((13) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (13)))*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15)))) + ((-2)*x0*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15)))*((16) * ((16) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (16)))) + ((-2)*x2*((13) * ((13) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (13)))*((16) * ((16) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (16)))) + 4*x0*x2*((16) * ((16) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (16))) + 8*x0*x1*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15))) + 8*x1*x2*((13) * ((13) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (13)))
    tmp156 = (tmp154 / tmp155)
    tmp157 = tl.full([1], 0, tl.int32)
    tmp158 = tmp157 < tmp156
    tmp159 = tmp158.to(tl.int8)
    tmp160 = tmp156 < tmp157
    tmp161 = tmp160.to(tl.int8)
    tmp162 = tmp159 - tmp161
    tmp163 = tmp162.to(tmp156.dtype)
    tmp164 = tl_math.abs(tmp156)
    tmp165 = triton_helpers.maximum(tmp157, tmp164)
    tmp166 = tmp163 * tmp165
    tmp167 = 16.0
    tmp168 = tmp166 * tmp167
    tmp169 = 0.3333333333333333
    tmp170 = libdevice.pow(tmp168, tmp169)
    tl.store(in_out_ptr0 + (x4), tmp170, xmask)
