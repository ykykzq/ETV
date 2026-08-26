
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 28
    x1 = ((xindex // 7) % 4)
    x0 = (xindex % 7)
    x3 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 15, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 7, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = 2*x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tl.full([1], 14, tl.int64)
    tmp14 = tmp11 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 & tmp10
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + (2*x0 + 28*x1 + 196*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp19 * tmp18
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = 1 + 2*x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp13
    tmp26 = tmp24 & tmp25
    tmp27 = tmp16 & tmp26
    tmp28 = tl.load(in_ptr0 + (1 + 2*x0 + 28*x1 + 196*x2), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp28 * tmp28
    tmp30 = tmp29 * tmp28
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tmp32 + tmp22
    tmp34 = 1 + 2*x1
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp8
    tmp37 = tmp35 & tmp36
    tmp38 = tmp5 & tmp37
    tmp39 = tmp38 & tmp15
    tmp40 = tl.load(in_ptr0 + (14 + 2*x0 + 28*x1 + 196*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp40
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp39, tmp42, tmp43)
    tmp45 = tmp44 + tmp33
    tmp46 = tmp38 & tmp26
    tmp47 = tl.load(in_ptr0 + (15 + 2*x0 + 28*x1 + 196*x2), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 * tmp47
    tmp49 = tmp48 * tmp47
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tmp51 + tmp45
    tmp53 = 1 + 2*x2
    tmp54 = tmp53 >= tmp1
    tmp55 = tmp53 < tmp3
    tmp56 = tmp54 & tmp55
    tmp57 = tmp56 & tmp10
    tmp58 = tmp57 & tmp15
    tmp59 = tl.load(in_ptr0 + (98 + 2*x0 + 28*x1 + 196*x2), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp60 * tmp59
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tmp63 + tmp52
    tmp65 = tmp57 & tmp26
    tmp66 = tl.load(in_ptr0 + (99 + 2*x0 + 28*x1 + 196*x2), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp66 * tmp66
    tmp68 = tmp67 * tmp66
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp65, tmp68, tmp69)
    tmp71 = tmp70 + tmp64
    tmp72 = tmp56 & tmp37
    tmp73 = tmp72 & tmp15
    tmp74 = tl.load(in_ptr0 + (112 + 2*x0 + 28*x1 + 196*x2), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp74 * tmp74
    tmp76 = tmp75 * tmp74
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp73, tmp76, tmp77)
    tmp79 = tmp78 + tmp71
    tmp80 = tmp72 & tmp26
    tmp81 = tl.load(in_ptr0 + (113 + 2*x0 + 28*x1 + 196*x2), tmp80 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tmp81 * tmp81
    tmp83 = tmp82 * tmp81
    tmp84 = tl.full(tmp83.shape, 0.0, tmp83.dtype)
    tmp85 = tl.where(tmp80, tmp83, tmp84)
    tmp86 = tmp85 + tmp79
    tmp87 = ((7) * ((7) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (7)))*((14) * ((14) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (14)))*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15))) + ((-8)*x0*x1*x2) + ((-2)*x0*((7) * ((7) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (7)))*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15)))) + ((-2)*x1*((14) * ((14) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (14)))*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15)))) + ((-2)*x2*((7) * ((7) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (7)))*((14) * ((14) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (14)))) + 4*x0*x1*((15) * ((15) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (15))) + 4*x0*x2*((7) * ((7) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (7))) + 4*x1*x2*((14) * ((14) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (14)))
    tmp88 = (tmp86 / tmp87)
    tmp89 = tl.full([1], 0, tl.int32)
    tmp90 = tmp89 < tmp88
    tmp91 = tmp90.to(tl.int8)
    tmp92 = tmp88 < tmp89
    tmp93 = tmp92.to(tl.int8)
    tmp94 = tmp91 - tmp93
    tmp95 = tmp94.to(tmp88.dtype)
    tmp96 = tl_math.abs(tmp88)
    tmp97 = triton_helpers.maximum(tmp89, tmp96)
    tmp98 = tmp95 * tmp97
    tmp99 = 8.0
    tmp100 = tmp98 * tmp99
    tmp101 = 0.3333333333333333
    tmp102 = libdevice.pow(tmp100, tmp101)
    tl.store(in_out_ptr0 + (x3), tmp102, xmask)
