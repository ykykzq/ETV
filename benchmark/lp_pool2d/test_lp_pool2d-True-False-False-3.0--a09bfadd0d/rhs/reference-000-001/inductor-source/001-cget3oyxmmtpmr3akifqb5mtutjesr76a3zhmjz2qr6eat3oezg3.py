
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 7)
    x2 = xindex // 28
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (16 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (17 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (18 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (19 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (20 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (32 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (33 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (34 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (35 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (36 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (48 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (49 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr0 + (50 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (51 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (52 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr0 + (64 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr0 + (65 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (66 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (67 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr0 + (68 + 3*x0 + 48*x1 + 384*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4 * tmp3
    tmp6 = tmp5 + tmp2
    tmp8 = tmp7 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 + tmp6
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tmp13 + tmp10
    tmp16 = tmp15 * tmp15
    tmp17 = tmp16 * tmp15
    tmp18 = tmp17 + tmp14
    tmp20 = tmp19 * tmp19
    tmp21 = tmp20 * tmp19
    tmp22 = tmp21 + tmp18
    tmp24 = tmp23 * tmp23
    tmp25 = tmp24 * tmp23
    tmp26 = tmp25 + tmp22
    tmp28 = tmp27 * tmp27
    tmp29 = tmp28 * tmp27
    tmp30 = tmp29 + tmp26
    tmp32 = tmp31 * tmp31
    tmp33 = tmp32 * tmp31
    tmp34 = tmp33 + tmp30
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 * tmp35
    tmp38 = tmp37 + tmp34
    tmp40 = tmp39 * tmp39
    tmp41 = tmp40 * tmp39
    tmp42 = tmp41 + tmp38
    tmp44 = tmp43 * tmp43
    tmp45 = tmp44 * tmp43
    tmp46 = tmp45 + tmp42
    tmp48 = tmp47 * tmp47
    tmp49 = tmp48 * tmp47
    tmp50 = tmp49 + tmp46
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tmp53 + tmp50
    tmp56 = tmp55 * tmp55
    tmp57 = tmp56 * tmp55
    tmp58 = tmp57 + tmp54
    tmp60 = tmp59 * tmp59
    tmp61 = tmp60 * tmp59
    tmp62 = tmp61 + tmp58
    tmp64 = tmp63 * tmp63
    tmp65 = tmp64 * tmp63
    tmp66 = tmp65 + tmp62
    tmp68 = tmp67 * tmp67
    tmp69 = tmp68 * tmp67
    tmp70 = tmp69 + tmp66
    tmp72 = tmp71 * tmp71
    tmp73 = tmp72 * tmp71
    tmp74 = tmp73 + tmp70
    tmp76 = tmp75 * tmp75
    tmp77 = tmp76 * tmp75
    tmp78 = tmp77 + tmp74
    tmp80 = tmp79 * tmp79
    tmp81 = tmp80 * tmp79
    tmp82 = tmp81 + tmp78
    tmp84 = tmp83 * tmp83
    tmp85 = tmp84 * tmp83
    tmp86 = tmp85 + tmp82
    tmp88 = tmp87 * tmp87
    tmp89 = tmp88 * tmp87
    tmp90 = tmp89 + tmp86
    tmp92 = tmp91 * tmp91
    tmp93 = tmp92 * tmp91
    tmp94 = tmp93 + tmp90
    tmp96 = tmp95 * tmp95
    tmp97 = tmp96 * tmp95
    tmp98 = tmp97 + tmp94
    tmp99 = 0.04
    tmp100 = tmp98 * tmp99
    tmp101 = tl.full([1], 0, tl.int32)
    tmp102 = tmp101 < tmp100
    tmp103 = tmp102.to(tl.int8)
    tmp104 = tmp100 < tmp101
    tmp105 = tmp104.to(tl.int8)
    tmp106 = tmp103 - tmp105
    tmp107 = tmp106.to(tmp100.dtype)
    tmp108 = tl_math.abs(tmp100)
    tmp109 = triton_helpers.maximum(tmp101, tmp108)
    tmp110 = tmp107 * tmp109
    tmp111 = 25.0
    tmp112 = tmp110 * tmp111
    tmp113 = 0.3333333333333333
    tmp114 = libdevice.pow(tmp112, tmp113)
    tl.store(in_out_ptr0 + (x3), tmp114, xmask)
