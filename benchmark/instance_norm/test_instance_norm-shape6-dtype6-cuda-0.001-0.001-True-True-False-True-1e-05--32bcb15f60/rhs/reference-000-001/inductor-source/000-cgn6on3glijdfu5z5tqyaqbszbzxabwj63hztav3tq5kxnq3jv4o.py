
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr2', 'out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (x0), xmask)
    tmp22 = tl.load(in_ptr0 + (87 + 3*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (29 + x0), xmask)
    tmp26 = tl.load(in_ptr0 + (88 + 3*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (89 + 3*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (174 + 3*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr1 + (58 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (175 + 3*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (176 + 3*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (261 + 3*x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (87 + x0), xmask)
    tmp60 = tl.load(in_ptr0 + (262 + 3*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr0 + (263 + 3*x0), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr0 + (348 + 3*x0), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr1 + (116 + x0), xmask)
    tmp77 = tl.load(in_ptr0 + (349 + 3*x0), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (350 + 3*x0), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (435 + 3*x0), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr1 + (145 + x0), xmask)
    tmp94 = tl.load(in_ptr0 + (436 + 3*x0), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr0 + (437 + 3*x0), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 - tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 - tmp1
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = 3.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.5
    tmp15 = tmp13 * tmp14
    tmp16 = 0.1
    tmp17 = tmp15 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp24
    tmp27 = tmp26 - tmp23
    tmp28 = tmp27 * tmp27
    tmp29 = tmp25 + tmp28
    tmp31 = tmp30 - tmp23
    tmp32 = tmp31 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = (tmp33 / tmp12)
    tmp35 = tmp34 * tmp14
    tmp36 = tmp35 * tmp16
    tmp37 = tmp36 + tmp20
    tmp38 = tmp21 + tmp37
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 * tmp41
    tmp44 = tmp43 - tmp40
    tmp45 = tmp44 * tmp44
    tmp46 = tmp42 + tmp45
    tmp48 = tmp47 - tmp40
    tmp49 = tmp48 * tmp48
    tmp50 = tmp46 + tmp49
    tmp51 = (tmp50 / tmp12)
    tmp52 = tmp51 * tmp14
    tmp53 = tmp52 * tmp16
    tmp54 = tmp53 + tmp20
    tmp55 = tmp38 + tmp54
    tmp58 = tmp56 - tmp57
    tmp59 = tmp58 * tmp58
    tmp61 = tmp60 - tmp57
    tmp62 = tmp61 * tmp61
    tmp63 = tmp59 + tmp62
    tmp65 = tmp64 - tmp57
    tmp66 = tmp65 * tmp65
    tmp67 = tmp63 + tmp66
    tmp68 = (tmp67 / tmp12)
    tmp69 = tmp68 * tmp14
    tmp70 = tmp69 * tmp16
    tmp71 = tmp70 + tmp20
    tmp72 = tmp55 + tmp71
    tmp75 = tmp73 - tmp74
    tmp76 = tmp75 * tmp75
    tmp78 = tmp77 - tmp74
    tmp79 = tmp78 * tmp78
    tmp80 = tmp76 + tmp79
    tmp82 = tmp81 - tmp74
    tmp83 = tmp82 * tmp82
    tmp84 = tmp80 + tmp83
    tmp85 = (tmp84 / tmp12)
    tmp86 = tmp85 * tmp14
    tmp87 = tmp86 * tmp16
    tmp88 = tmp87 + tmp20
    tmp89 = tmp72 + tmp88
    tmp92 = tmp90 - tmp91
    tmp93 = tmp92 * tmp92
    tmp95 = tmp94 - tmp91
    tmp96 = tmp95 * tmp95
    tmp97 = tmp93 + tmp96
    tmp99 = tmp98 - tmp91
    tmp100 = tmp99 * tmp99
    tmp101 = tmp97 + tmp100
    tmp102 = (tmp101 / tmp12)
    tmp103 = tmp102 * tmp14
    tmp104 = tmp103 * tmp16
    tmp105 = tmp104 + tmp20
    tmp106 = tmp89 + tmp105
    tmp107 = 6.0
    tmp108 = (tmp106 / tmp107)
    tmp109 = tmp1 * tmp16
    tmp111 = tmp110 * tmp19
    tmp112 = tmp109 + tmp111
    tmp113 = tmp23 * tmp16
    tmp114 = tmp113 + tmp111
    tmp115 = tmp112 + tmp114
    tmp116 = tmp40 * tmp16
    tmp117 = tmp116 + tmp111
    tmp118 = tmp115 + tmp117
    tmp119 = tmp57 * tmp16
    tmp120 = tmp119 + tmp111
    tmp121 = tmp118 + tmp120
    tmp122 = tmp74 * tmp16
    tmp123 = tmp122 + tmp111
    tmp124 = tmp121 + tmp123
    tmp125 = tmp91 * tmp16
    tmp126 = tmp125 + tmp111
    tmp127 = tmp124 + tmp126
    tmp128 = (tmp127 / tmp107)
    tl.store(out_ptr2 + (x0), tmp128, xmask)
    tl.store(out_ptr3 + (x0), tmp108, xmask)
