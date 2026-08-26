
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30492}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 693
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 11)
    x1 = ((xindex // 11) % 7)
    x2 = xindex // 77
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 13*x1 + 117*x2), xmask)
    tmp2 = tl.load(in_ptr0 + (1 + x0 + 13*x1 + 117*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (2 + x0 + 13*x1 + 117*x2), xmask)
    tmp8 = tl.load(in_ptr0 + (13 + x0 + 13*x1 + 117*x2), xmask)
    tmp11 = tl.load(in_ptr0 + (14 + x0 + 13*x1 + 117*x2), xmask)
    tmp14 = tl.load(in_ptr0 + (15 + x0 + 13*x1 + 117*x2), xmask)
    tmp17 = tl.load(in_ptr0 + (26 + x0 + 13*x1 + 117*x2), xmask)
    tmp20 = tl.load(in_ptr0 + (27 + x0 + 13*x1 + 117*x2), xmask)
    tmp23 = tl.load(in_ptr0 + (28 + x0 + 13*x1 + 117*x2), xmask)
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
    tmp26 = 0.1111111111111111
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = tmp28 < tmp27
    tmp30 = tmp29.to(tl.int8)
    tmp31 = tmp27 < tmp28
    tmp32 = tmp31.to(tl.int8)
    tmp33 = tmp30 - tmp32
    tmp34 = tmp33.to(tmp27.dtype)
    tmp35 = tl_math.abs(tmp27)
    tmp36 = triton_helpers.maximum(tmp28, tmp35)
    tmp37 = tmp34 * tmp36
    tmp38 = 9.0
    tmp39 = tmp37 * tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tl.store(in_out_ptr0 + (x3), tmp40, xmask)
