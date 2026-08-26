
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17496}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 729
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 81) % 3)
    x1 = ((xindex // 9) % 9)
    x0 = (xindex % 9)
    x3 = xindex // 243
    x4 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 9, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = x0
    tmp12 = tmp11 >= tmp1
    tmp13 = tl.full([1], 10, tl.int64)
    tmp14 = tmp11 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 & tmp10
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + (x0 + 10*x1 + 180*x2 + 450*x3), tmp17 & xmask, other=0.0)
    tmp19 = tmp18 * tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = 1 + x0
    tmp23 = tmp22 >= tmp1
    tmp24 = tmp22 < tmp13
    tmp25 = tmp23 & tmp24
    tmp26 = tmp16 & tmp25
    tmp27 = tl.load(in_ptr0 + (1 + x0 + 10*x1 + 180*x2 + 450*x3), tmp26 & xmask, other=0.0)
    tmp28 = tmp27 * tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp26, tmp28, tmp29)
    tmp31 = tmp30 + tmp21
    tmp32 = 1 + 2*x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp10
    tmp37 = tmp36 & tmp15
    tmp38 = tl.load(in_ptr0 + (90 + x0 + 10*x1 + 180*x2 + 450*x3), tmp37 & xmask, other=0.0)
    tmp39 = tmp38 * tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tmp41 + tmp31
    tmp43 = tmp36 & tmp25
    tmp44 = tl.load(in_ptr0 + (91 + x0 + 10*x1 + 180*x2 + 450*x3), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 * tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp43, tmp45, tmp46)
    tmp48 = tmp47 + tmp42
    tmp49 = x0*x1*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5))) + ((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))*((9) * ((9) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (9)))*((10) * ((10) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (10))) + ((-1)*x0*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))*((9) * ((9) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (9)))) + ((-1)*x1*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))*((10) * ((10) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (10)))) + ((-2)*x0*x1*x2) + ((-2)*x2*((9) * ((9) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (9)))*((10) * ((10) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (10)))) + 2*x0*x2*((9) * ((9) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (9))) + 2*x1*x2*((10) * ((10) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (10)))
    tmp50 = (tmp48 / tmp49)
    tmp51 = tl.full([1], 0, tl.int32)
    tmp52 = tmp51 < tmp50
    tmp53 = tmp52.to(tl.int8)
    tmp54 = tmp50 < tmp51
    tmp55 = tmp54.to(tl.int8)
    tmp56 = tmp53 - tmp55
    tmp57 = tmp56.to(tmp50.dtype)
    tmp58 = tl_math.abs(tmp50)
    tmp59 = triton_helpers.maximum(tmp51, tmp58)
    tmp60 = tmp57 * tmp59
    tmp61 = 4.0
    tmp62 = tmp60 * tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tl.store(in_out_ptr0 + (x4), tmp63, xmask)
