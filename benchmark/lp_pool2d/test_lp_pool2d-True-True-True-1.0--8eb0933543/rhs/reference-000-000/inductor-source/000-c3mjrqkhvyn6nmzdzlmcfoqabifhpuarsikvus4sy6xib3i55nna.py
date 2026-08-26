
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 5) % 10)
    x0 = (xindex % 5)
    x4 = xindex // 5
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 4*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 17, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + (4*x0 + 17*x4), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1 + 4*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (1 + 4*x0 + 17*x4), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 2 + 4*x0
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp8
    tmp23 = tmp21 & tmp22
    tmp24 = tmp5 & tmp23
    tmp25 = tl.load(in_ptr0 + (2 + 4*x0 + 17*x4), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = 3 + 4*x0
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp8
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0 + 17*x4), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = ((10) * ((10) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (10)))*((17) * ((17) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (17))) + ((-1)*x1*((17) * ((17) <= (4 + 4*x0)) + (4 + 4*x0) * ((4 + 4*x0) < (17)))) + ((-4)*x0*((10) * ((10) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (10)))) + 4*x0*x1
    tmp35 = (tmp33 / tmp34)
    tmp36 = tl.full([1], 0, tl.int32)
    tmp37 = tmp36 < tmp35
    tmp38 = tmp37.to(tl.int8)
    tmp39 = tmp35 < tmp36
    tmp40 = tmp39.to(tl.int8)
    tmp41 = tmp38 - tmp40
    tmp42 = tmp41.to(tmp35.dtype)
    tmp43 = tl_math.abs(tmp35)
    tmp44 = triton_helpers.maximum(tmp36, tmp43)
    tmp45 = tmp42 * tmp44
    tmp46 = 4.0
    tmp47 = tmp45 * tmp46
    tl.store(in_out_ptr0 + (x3), tmp47, xmask)
