
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 208000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 13)
    x2 = ((xindex // 130) % 10)
    x3 = xindex // 1300
    x4 = (xindex % 1300)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (1 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (11 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (12 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (154 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp19 = tl.load(in_ptr0 + (155 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp23 = tl.load(in_ptr0 + (165 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
    tmp27 = tl.load(in_ptr0 + (166 + x0 + 11*x1 + 154*x2 + 1694*x3), xmask)
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
    tmp31 = 0.125
    tmp32 = tmp30 * tmp31
    tmp33 = tl.full([1], 0, tl.int32)
    tmp34 = tmp33 < tmp32
    tmp35 = tmp34.to(tl.int8)
    tmp36 = tmp32 < tmp33
    tmp37 = tmp36.to(tl.int8)
    tmp38 = tmp35 - tmp37
    tmp39 = tmp38.to(tmp32.dtype)
    tmp40 = tl_math.abs(tmp32)
    tmp41 = triton_helpers.maximum(tmp33, tmp40)
    tmp42 = tmp39 * tmp41
    tmp43 = 8.0
    tmp44 = tmp42 * tmp43
    tmp45 = 0.3333333333333333
    tmp46 = libdevice.pow(tmp44, tmp45)
    tl.store(out_ptr1 + (x5), tmp46, xmask)
