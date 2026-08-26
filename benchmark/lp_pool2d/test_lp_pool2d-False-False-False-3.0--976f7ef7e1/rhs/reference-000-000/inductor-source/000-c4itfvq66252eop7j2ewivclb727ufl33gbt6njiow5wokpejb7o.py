
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (6 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (7 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (8 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (9 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (12 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (13 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (14 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (15 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (18 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (19 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (20 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr0 + (21 + 24*x0 + 138*x1), xmask, eviction_policy='evict_last')
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
    tmp63 = 0.0625
    tmp64 = tmp62 * tmp63
    tmp65 = tl.full([1], 0, tl.int32)
    tmp66 = tmp65 < tmp64
    tmp67 = tmp66.to(tl.int8)
    tmp68 = tmp64 < tmp65
    tmp69 = tmp68.to(tl.int8)
    tmp70 = tmp67 - tmp69
    tmp71 = tmp70.to(tmp64.dtype)
    tmp72 = tl_math.abs(tmp64)
    tmp73 = triton_helpers.maximum(tmp65, tmp72)
    tmp74 = tmp71 * tmp73
    tmp75 = 16.0
    tmp76 = tmp74 * tmp75
    tmp77 = 0.3333333333333333
    tmp78 = libdevice.pow(tmp76, tmp77)
    tl.store(in_out_ptr0 + (x2), tmp78, xmask)
