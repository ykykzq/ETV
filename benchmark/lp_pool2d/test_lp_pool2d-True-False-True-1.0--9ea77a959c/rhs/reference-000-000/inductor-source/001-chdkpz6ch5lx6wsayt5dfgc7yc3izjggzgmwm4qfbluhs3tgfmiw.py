
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 4)
    x0 = (xindex % 3)
    x2 = xindex // 12
    x3 = xindex
    tmp0 = 4*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 4*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 11, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (4*x0 + 44*x1 + 154*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1 + 4*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + 4*x0 + 44*x1 + 154*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 2 + 4*x0
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp8
    tmp23 = tmp21 & tmp22
    tmp24 = tmp5 & tmp23
    tmp25 = tl.load(in_ptr0 + (2 + 4*x0 + 44*x1 + 154*x2), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = 3 + 4*x0
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp8
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0 + 44*x1 + 154*x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = 1 + 4*x1
    tmp35 = tmp34 >= tmp1
    tmp36 = tmp34 < tmp3
    tmp37 = tmp35 & tmp36
    tmp38 = tmp37 & tmp10
    tmp39 = tl.load(in_ptr0 + (11 + 4*x0 + 44*x1 + 154*x2), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39 + tmp33
    tmp41 = tmp37 & tmp16
    tmp42 = tl.load(in_ptr0 + (12 + 4*x0 + 44*x1 + 154*x2), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp42 + tmp40
    tmp44 = tmp37 & tmp23
    tmp45 = tl.load(in_ptr0 + (13 + 4*x0 + 44*x1 + 154*x2), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp43
    tmp47 = tmp37 & tmp30
    tmp48 = tl.load(in_ptr0 + (14 + 4*x0 + 44*x1 + 154*x2), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = 2 + 4*x1
    tmp51 = tmp50 >= tmp1
    tmp52 = tmp50 < tmp3
    tmp53 = tmp51 & tmp52
    tmp54 = tmp53 & tmp10
    tmp55 = tl.load(in_ptr0 + (22 + 4*x0 + 44*x1 + 154*x2), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp55 + tmp49
    tmp57 = tmp53 & tmp16
    tmp58 = tl.load(in_ptr0 + (23 + 4*x0 + 44*x1 + 154*x2), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp56
    tmp60 = tmp53 & tmp23
    tmp61 = tl.load(in_ptr0 + (24 + 4*x0 + 44*x1 + 154*x2), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp61 + tmp59
    tmp63 = tmp53 & tmp30
    tmp64 = tl.load(in_ptr0 + (25 + 4*x0 + 44*x1 + 154*x2), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64 + tmp62
    tmp66 = 3 + 4*x1
    tmp67 = tmp66 >= tmp1
    tmp68 = tmp66 < tmp3
    tmp69 = tmp67 & tmp68
    tmp70 = tmp69 & tmp10
    tmp71 = tl.load(in_ptr0 + (33 + 4*x0 + 44*x1 + 154*x2), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp71 + tmp65
    tmp73 = tmp69 & tmp16
    tmp74 = tl.load(in_ptr0 + (34 + 4*x0 + 44*x1 + 154*x2), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp74 + tmp72
    tmp76 = tmp69 & tmp23
    tmp77 = tl.load(in_ptr0 + (35 + 4*x0 + 44*x1 + 154*x2), tmp76 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp77 + tmp75
    tmp79 = tmp69 & tmp30
    tmp80 = tl.load(in_ptr0 + (36 + 4*x0 + 44*x1 + 154*x2), tmp79 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp78
    tmp82 = ((11) * ((11) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (11)))*((14) * ((14) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (14))) + ((-4)*x0*((14) * ((14) <= (4 + 4*x1)) + (4 + 4*x1) * ((4 + 4*x1) < (14)))) + ((-4)*x1*((11) * ((11) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (11)))) + 16*x0*x1
    tmp83 = (tmp81 / tmp82)
    tmp84 = tl.full([1], 0, tl.int32)
    tmp85 = tmp84 < tmp83
    tmp86 = tmp85.to(tl.int8)
    tmp87 = tmp83 < tmp84
    tmp88 = tmp87.to(tl.int8)
    tmp89 = tmp86 - tmp88
    tmp90 = tmp89.to(tmp83.dtype)
    tmp91 = tl_math.abs(tmp83)
    tmp92 = triton_helpers.maximum(tmp84, tmp91)
    tmp93 = tmp90 * tmp92
    tmp94 = 16.0
    tmp95 = tmp93 * tmp94
    tl.store(in_out_ptr0 + (x3), tmp95, xmask)
