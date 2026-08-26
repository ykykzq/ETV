
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 4)
    x2 = xindex // 8
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (12 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (13 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (14 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (15 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (16 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (24 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (25 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (26 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (27 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (28 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr0 + (36 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (37 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (38 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (39 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (40 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (48 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (49 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (50 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr0 + (51 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (52 + 5*x0 + 60*x1 + 252*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 + tmp7
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14 * tmp14
    tmp16 = tmp15 + tmp13
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 + tmp16
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 + tmp19
    tmp24 = tmp23 * tmp23
    tmp25 = tmp24 + tmp22
    tmp27 = tmp26 * tmp26
    tmp28 = tmp27 + tmp25
    tmp30 = tmp29 * tmp29
    tmp31 = tmp30 + tmp28
    tmp33 = tmp32 * tmp32
    tmp34 = tmp33 + tmp31
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 + tmp34
    tmp39 = tmp38 * tmp38
    tmp40 = tmp39 + tmp37
    tmp42 = tmp41 * tmp41
    tmp43 = tmp42 + tmp40
    tmp45 = tmp44 * tmp44
    tmp46 = tmp45 + tmp43
    tmp48 = tmp47 * tmp47
    tmp49 = tmp48 + tmp46
    tmp51 = tmp50 * tmp50
    tmp52 = tmp51 + tmp49
    tmp54 = tmp53 * tmp53
    tmp55 = tmp54 + tmp52
    tmp57 = tmp56 * tmp56
    tmp58 = tmp57 + tmp55
    tmp60 = tmp59 * tmp59
    tmp61 = tmp60 + tmp58
    tmp63 = tmp62 * tmp62
    tmp64 = tmp63 + tmp61
    tmp66 = tmp65 * tmp65
    tmp67 = tmp66 + tmp64
    tmp69 = tmp68 * tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp71 * tmp71
    tmp73 = tmp72 + tmp70
    tmp74 = 0.04
    tmp75 = tmp73 * tmp74
    tmp76 = tl.full([1], 0, tl.int32)
    tmp77 = tmp76 < tmp75
    tmp78 = tmp77.to(tl.int8)
    tmp79 = tmp75 < tmp76
    tmp80 = tmp79.to(tl.int8)
    tmp81 = tmp78 - tmp80
    tmp82 = tmp81.to(tmp75.dtype)
    tmp83 = tl_math.abs(tmp75)
    tmp84 = triton_helpers.maximum(tmp76, tmp83)
    tmp85 = tmp82 * tmp84
    tmp86 = 25.0
    tmp87 = tmp85 * tmp86
    tmp88 = libdevice.sqrt(tmp87)
    tl.store(in_out_ptr0 + (x3), tmp88, xmask)
