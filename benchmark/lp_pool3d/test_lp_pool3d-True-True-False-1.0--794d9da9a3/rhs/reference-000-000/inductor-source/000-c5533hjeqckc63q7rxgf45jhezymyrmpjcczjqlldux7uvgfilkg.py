
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 7)
    x2 = xindex // 35
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (13 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (15 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (16 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (182 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (183 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (184 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (185 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (195 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (196 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (197 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (198 + 2*x0 + 26*x1 + 364*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
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
    tmp43 = 16.0
    tmp44 = tmp42 * tmp43
    tl.store(in_out_ptr0 + (x3), tmp44, xmask)
