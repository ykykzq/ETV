
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 14) % 3)
    x1 = ((xindex // 7) % 2)
    x0 = (xindex % 7)
    x3 = xindex // 42
    x4 = xindex
    tmp0 = 3*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 3*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 7, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tmp11 < tmp8
    tmp14 = tmp12 & tmp13
    tmp15 = tmp5 & tmp10
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + (x0 + 21*x1 + 147*x2 + 392*x3), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 * tmp17
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = 1 + 3*x1
    tmp23 = tmp22 >= tmp1
    tmp24 = tmp22 < tmp8
    tmp25 = tmp23 & tmp24
    tmp26 = tmp5 & tmp25
    tmp27 = tmp26 & tmp14
    tmp28 = tl.load(in_ptr0 + (7 + x0 + 21*x1 + 147*x2 + 392*x3), tmp27 & xmask, other=0.0)
    tmp29 = tmp28 * tmp28
    tmp30 = tmp29 * tmp28
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tmp32 + tmp21
    tmp34 = 2 + 3*x1
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp8
    tmp37 = tmp35 & tmp36
    tmp38 = tmp5 & tmp37
    tmp39 = tmp38 & tmp14
    tmp40 = tl.load(in_ptr0 + (14 + x0 + 21*x1 + 147*x2 + 392*x3), tmp39 & xmask, other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp33
    tmp46 = 3 + 3*x1
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp8
    tmp49 = tmp47 & tmp48
    tmp50 = tmp5 & tmp49
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (21 + x0 + 21*x1 + 147*x2 + 392*x3), tmp51 & xmask, other=0.0)
    tmp53 = tmp52 * tmp52
    tmp54 = tmp53 * tmp52
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp51, tmp54, tmp55)
    tmp57 = tmp56 + tmp45
    tmp58 = 1 + 3*x2
    tmp59 = tmp58 >= tmp1
    tmp60 = tmp58 < tmp3
    tmp61 = tmp59 & tmp60
    tmp62 = tmp61 & tmp10
    tmp63 = tmp62 & tmp14
    tmp64 = tl.load(in_ptr0 + (49 + x0 + 21*x1 + 147*x2 + 392*x3), tmp63 & xmask, other=0.0)
    tmp65 = tmp64 * tmp64
    tmp66 = tmp65 * tmp64
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp63, tmp66, tmp67)
    tmp69 = tmp68 + tmp57
    tmp70 = tmp61 & tmp25
    tmp71 = tmp70 & tmp14
    tmp72 = tl.load(in_ptr0 + (56 + x0 + 21*x1 + 147*x2 + 392*x3), tmp71 & xmask, other=0.0)
    tmp73 = tmp72 * tmp72
    tmp74 = tmp73 * tmp72
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp71, tmp74, tmp75)
    tmp77 = tmp76 + tmp69
    tmp78 = tmp61 & tmp37
    tmp79 = tmp78 & tmp14
    tmp80 = tl.load(in_ptr0 + (63 + x0 + 21*x1 + 147*x2 + 392*x3), tmp79 & xmask, other=0.0)
    tmp81 = tmp80 * tmp80
    tmp82 = tmp81 * tmp80
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp79, tmp82, tmp83)
    tmp85 = tmp84 + tmp77
    tmp86 = tmp61 & tmp49
    tmp87 = tmp86 & tmp14
    tmp88 = tl.load(in_ptr0 + (70 + x0 + 21*x1 + 147*x2 + 392*x3), tmp87 & xmask, other=0.0)
    tmp89 = tmp88 * tmp88
    tmp90 = tmp89 * tmp88
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp87, tmp90, tmp91)
    tmp93 = tmp92 + tmp85
    tmp94 = 2 + 3*x2
    tmp95 = tmp94 >= tmp1
    tmp96 = tmp94 < tmp3
    tmp97 = tmp95 & tmp96
    tmp98 = tmp97 & tmp10
    tmp99 = tmp98 & tmp14
    tmp100 = tl.load(in_ptr0 + (98 + x0 + 21*x1 + 147*x2 + 392*x3), tmp99 & xmask, other=0.0)
    tmp101 = tmp100 * tmp100
    tmp102 = tmp101 * tmp100
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp99, tmp102, tmp103)
    tmp105 = tmp104 + tmp93
    tmp106 = tmp97 & tmp25
    tmp107 = tmp106 & tmp14
    tmp108 = tl.load(in_ptr0 + (105 + x0 + 21*x1 + 147*x2 + 392*x3), tmp107 & xmask, other=0.0)
    tmp109 = tmp108 * tmp108
    tmp110 = tmp109 * tmp108
    tmp111 = tl.full(tmp110.shape, 0.0, tmp110.dtype)
    tmp112 = tl.where(tmp107, tmp110, tmp111)
    tmp113 = tmp112 + tmp105
    tmp114 = tmp97 & tmp37
    tmp115 = tmp114 & tmp14
    tmp116 = tl.load(in_ptr0 + (112 + x0 + 21*x1 + 147*x2 + 392*x3), tmp115 & xmask, other=0.0)
    tmp117 = tmp116 * tmp116
    tmp118 = tmp117 * tmp116
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp115, tmp118, tmp119)
    tmp121 = tmp120 + tmp113
    tmp122 = tmp97 & tmp49
    tmp123 = tmp122 & tmp14
    tmp124 = tl.load(in_ptr0 + (119 + x0 + 21*x1 + 147*x2 + 392*x3), tmp123 & xmask, other=0.0)
    tmp125 = tmp124 * tmp124
    tmp126 = tmp125 * tmp124
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp123, tmp126, tmp127)
    tmp129 = tmp128 + tmp121
    tmp130 = ((7) * ((7) <= (1 + x0)) + (1 + x0) * ((1 + x0) < (7)))*((7) * ((7) <= (4 + 3*x1)) + (4 + 3*x1) * ((4 + 3*x1) < (7)))*((8) * ((8) <= (3 + 3*x2)) + (3 + 3*x2) * ((3 + 3*x2) < (8))) + ((-1)*x0*((7) * ((7) <= (4 + 3*x1)) + (4 + 3*x1) * ((4 + 3*x1) < (7)))*((8) * ((8) <= (3 + 3*x2)) + (3 + 3*x2) * ((3 + 3*x2) < (8)))) + ((-9)*x0*x1*x2) + ((-3)*x1*((7) * ((7) <= (1 + x0)) + (1 + x0) * ((1 + x0) < (7)))*((8) * ((8) <= (3 + 3*x2)) + (3 + 3*x2) * ((3 + 3*x2) < (8)))) + ((-3)*x2*((7) * ((7) <= (1 + x0)) + (1 + x0) * ((1 + x0) < (7)))*((7) * ((7) <= (4 + 3*x1)) + (4 + 3*x1) * ((4 + 3*x1) < (7)))) + 3*x0*x1*((8) * ((8) <= (3 + 3*x2)) + (3 + 3*x2) * ((3 + 3*x2) < (8))) + 3*x0*x2*((7) * ((7) <= (4 + 3*x1)) + (4 + 3*x1) * ((4 + 3*x1) < (7))) + 9*x1*x2*((7) * ((7) <= (1 + x0)) + (1 + x0) * ((1 + x0) < (7)))
    tmp131 = (tmp129 / tmp130)
    tmp132 = tl.full([1], 0, tl.int32)
    tmp133 = tmp132 < tmp131
    tmp134 = tmp133.to(tl.int8)
    tmp135 = tmp131 < tmp132
    tmp136 = tmp135.to(tl.int8)
    tmp137 = tmp134 - tmp136
    tmp138 = tmp137.to(tmp131.dtype)
    tmp139 = tl_math.abs(tmp131)
    tmp140 = triton_helpers.maximum(tmp132, tmp139)
    tmp141 = tmp138 * tmp140
    tmp142 = 12.0
    tmp143 = tmp141 * tmp142
    tmp144 = 0.3333333333333333
    tmp145 = libdevice.pow(tmp143, tmp144)
    tl.store(in_out_ptr0 + (x4), tmp145, xmask)
