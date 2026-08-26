
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = 5*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 9, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 5*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (5*x0 + 45*x1 + 81*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = 1 + 5*x0
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tmp16 & tmp17
    tmp19 = tmp5 & tmp18
    tmp20 = tl.load(in_ptr0 + (1 + 5*x0 + 45*x1 + 81*x2), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 * tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp19, tmp21, tmp22)
    tmp24 = tmp23 + tmp14
    tmp25 = 2 + 5*x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp3
    tmp28 = tmp26 & tmp27
    tmp29 = tmp5 & tmp28
    tmp30 = tl.load(in_ptr0 + (2 + 5*x0 + 45*x1 + 81*x2), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp29, tmp31, tmp32)
    tmp34 = tmp33 + tmp24
    tmp35 = 3 + 5*x0
    tmp36 = tmp35 >= tmp1
    tmp37 = tmp35 < tmp3
    tmp38 = tmp36 & tmp37
    tmp39 = tmp5 & tmp38
    tmp40 = tl.load(in_ptr0 + (3 + 5*x0 + 45*x1 + 81*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp39, tmp41, tmp42)
    tmp44 = tmp43 + tmp34
    tmp45 = 4 + 5*x0
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp5 & tmp48
    tmp50 = tl.load(in_ptr0 + (4 + 5*x0 + 45*x1 + 81*x2), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 * tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp49, tmp51, tmp52)
    tmp54 = tmp53 + tmp44
    tmp55 = 1 + 5*x1
    tmp56 = tmp55 >= tmp1
    tmp57 = tmp55 < tmp3
    tmp58 = tmp56 & tmp57
    tmp59 = tmp58 & tmp9
    tmp60 = tl.load(in_ptr0 + (9 + 5*x0 + 45*x1 + 81*x2), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 * tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp59, tmp61, tmp62)
    tmp64 = tmp63 + tmp54
    tmp65 = tmp58 & tmp18
    tmp66 = tl.load(in_ptr0 + (10 + 5*x0 + 45*x1 + 81*x2), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp66 * tmp66
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp65, tmp67, tmp68)
    tmp70 = tmp69 + tmp64
    tmp71 = tmp58 & tmp28
    tmp72 = tl.load(in_ptr0 + (11 + 5*x0 + 45*x1 + 81*x2), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp72 * tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp71, tmp73, tmp74)
    tmp76 = tmp75 + tmp70
    tmp77 = tmp58 & tmp38
    tmp78 = tl.load(in_ptr0 + (12 + 5*x0 + 45*x1 + 81*x2), tmp77 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 * tmp78
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp77, tmp79, tmp80)
    tmp82 = tmp81 + tmp76
    tmp83 = tmp58 & tmp48
    tmp84 = tl.load(in_ptr0 + (13 + 5*x0 + 45*x1 + 81*x2), tmp83 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 * tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp83, tmp85, tmp86)
    tmp88 = tmp87 + tmp82
    tmp89 = 2 + 5*x1
    tmp90 = tmp89 >= tmp1
    tmp91 = tmp89 < tmp3
    tmp92 = tmp90 & tmp91
    tmp93 = tmp92 & tmp9
    tmp94 = tl.load(in_ptr0 + (18 + 5*x0 + 45*x1 + 81*x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp94 * tmp94
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp93, tmp95, tmp96)
    tmp98 = tmp97 + tmp88
    tmp99 = tmp92 & tmp18
    tmp100 = tl.load(in_ptr0 + (19 + 5*x0 + 45*x1 + 81*x2), tmp99 & xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp100 * tmp100
    tmp102 = tl.full(tmp101.shape, 0.0, tmp101.dtype)
    tmp103 = tl.where(tmp99, tmp101, tmp102)
    tmp104 = tmp103 + tmp98
    tmp105 = tmp92 & tmp28
    tmp106 = tl.load(in_ptr0 + (20 + 5*x0 + 45*x1 + 81*x2), tmp105 & xmask, eviction_policy='evict_last', other=0.0)
    tmp107 = tmp106 * tmp106
    tmp108 = tl.full(tmp107.shape, 0.0, tmp107.dtype)
    tmp109 = tl.where(tmp105, tmp107, tmp108)
    tmp110 = tmp109 + tmp104
    tmp111 = tmp92 & tmp38
    tmp112 = tl.load(in_ptr0 + (21 + 5*x0 + 45*x1 + 81*x2), tmp111 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tmp112 * tmp112
    tmp114 = tl.full(tmp113.shape, 0.0, tmp113.dtype)
    tmp115 = tl.where(tmp111, tmp113, tmp114)
    tmp116 = tmp115 + tmp110
    tmp117 = tmp92 & tmp48
    tmp118 = tl.load(in_ptr0 + (22 + 5*x0 + 45*x1 + 81*x2), tmp117 & xmask, eviction_policy='evict_last', other=0.0)
    tmp119 = tmp118 * tmp118
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp117, tmp119, tmp120)
    tmp122 = tmp121 + tmp116
    tmp123 = 3 + 5*x1
    tmp124 = tmp123 >= tmp1
    tmp125 = tmp123 < tmp3
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp9
    tmp128 = tl.load(in_ptr0 + (27 + 5*x0 + 45*x1 + 81*x2), tmp127 & xmask, eviction_policy='evict_last', other=0.0)
    tmp129 = tmp128 * tmp128
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp122
    tmp133 = tmp126 & tmp18
    tmp134 = tl.load(in_ptr0 + (28 + 5*x0 + 45*x1 + 81*x2), tmp133 & xmask, eviction_policy='evict_last', other=0.0)
    tmp135 = tmp134 * tmp134
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp28
    tmp140 = tl.load(in_ptr0 + (29 + 5*x0 + 45*x1 + 81*x2), tmp139 & xmask, eviction_policy='evict_last', other=0.0)
    tmp141 = tmp140 * tmp140
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp126 & tmp38
    tmp146 = tl.load(in_ptr0 + (30 + 5*x0 + 45*x1 + 81*x2), tmp145 & xmask, eviction_policy='evict_last', other=0.0)
    tmp147 = tmp146 * tmp146
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp145, tmp147, tmp148)
    tmp150 = tmp149 + tmp144
    tmp151 = tmp126 & tmp48
    tmp152 = tl.load(in_ptr0 + (31 + 5*x0 + 45*x1 + 81*x2), tmp151 & xmask, eviction_policy='evict_last', other=0.0)
    tmp153 = tmp152 * tmp152
    tmp154 = tl.full(tmp153.shape, 0.0, tmp153.dtype)
    tmp155 = tl.where(tmp151, tmp153, tmp154)
    tmp156 = tmp155 + tmp150
    tmp157 = 4 + 5*x1
    tmp158 = tmp157 >= tmp1
    tmp159 = tmp157 < tmp3
    tmp160 = tmp158 & tmp159
    tmp161 = tmp160 & tmp9
    tmp162 = tl.load(in_ptr0 + (36 + 5*x0 + 45*x1 + 81*x2), tmp161 & xmask, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp162 * tmp162
    tmp164 = tl.full(tmp163.shape, 0.0, tmp163.dtype)
    tmp165 = tl.where(tmp161, tmp163, tmp164)
    tmp166 = tmp165 + tmp156
    tmp167 = tmp160 & tmp18
    tmp168 = tl.load(in_ptr0 + (37 + 5*x0 + 45*x1 + 81*x2), tmp167 & xmask, eviction_policy='evict_last', other=0.0)
    tmp169 = tmp168 * tmp168
    tmp170 = tl.full(tmp169.shape, 0.0, tmp169.dtype)
    tmp171 = tl.where(tmp167, tmp169, tmp170)
    tmp172 = tmp171 + tmp166
    tmp173 = tmp160 & tmp28
    tmp174 = tl.load(in_ptr0 + (38 + 5*x0 + 45*x1 + 81*x2), tmp173 & xmask, eviction_policy='evict_last', other=0.0)
    tmp175 = tmp174 * tmp174
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp173, tmp175, tmp176)
    tmp178 = tmp177 + tmp172
    tmp179 = tmp160 & tmp38
    tmp180 = tl.load(in_ptr0 + (39 + 5*x0 + 45*x1 + 81*x2), tmp179 & xmask, eviction_policy='evict_last', other=0.0)
    tmp181 = tmp180 * tmp180
    tmp182 = tl.full(tmp181.shape, 0.0, tmp181.dtype)
    tmp183 = tl.where(tmp179, tmp181, tmp182)
    tmp184 = tmp183 + tmp178
    tmp185 = tmp160 & tmp48
    tmp186 = tl.load(in_ptr0 + (40 + 5*x0 + 45*x1 + 81*x2), tmp185 & xmask, eviction_policy='evict_last', other=0.0)
    tmp187 = tmp186 * tmp186
    tmp188 = tl.full(tmp187.shape, 0.0, tmp187.dtype)
    tmp189 = tl.where(tmp185, tmp187, tmp188)
    tmp190 = tmp189 + tmp184
    tmp191 = ((9) * ((9) <= (5 + 5*x0)) + (5 + 5*x0) * ((5 + 5*x0) < (9)))*((9) * ((9) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (9))) + ((-5)*x0*((9) * ((9) <= (5 + 5*x1)) + (5 + 5*x1) * ((5 + 5*x1) < (9)))) + ((-5)*x1*((9) * ((9) <= (5 + 5*x0)) + (5 + 5*x0) * ((5 + 5*x0) < (9)))) + 25*x0*x1
    tmp192 = (tmp190 / tmp191)
    tmp193 = tl.full([1], 0, tl.int32)
    tmp194 = tmp193 < tmp192
    tmp195 = tmp194.to(tl.int8)
    tmp196 = tmp192 < tmp193
    tmp197 = tmp196.to(tl.int8)
    tmp198 = tmp195 - tmp197
    tmp199 = tmp198.to(tmp192.dtype)
    tmp200 = tl_math.abs(tmp192)
    tmp201 = triton_helpers.maximum(tmp193, tmp200)
    tmp202 = tmp199 * tmp201
    tmp203 = 25.0
    tmp204 = tmp202 * tmp203
    tmp205 = libdevice.sqrt(tmp204)
    tl.store(in_out_ptr0 + (x3), tmp205, xmask)
