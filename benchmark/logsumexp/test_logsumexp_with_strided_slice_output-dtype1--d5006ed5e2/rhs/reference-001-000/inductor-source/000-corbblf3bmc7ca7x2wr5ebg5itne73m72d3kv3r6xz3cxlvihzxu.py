
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 12*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 12*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 12*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 12*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl_math.abs(tmp10)
    tmp12 = float("inf")
    tmp13 = tmp11 == tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = tmp0 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp1 - tmp15
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp3 - tmp15
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp5 - tmp15
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp7 - tmp15
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp9 - tmp15
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tmp33 = tl_math.log(tmp32)
    tmp34 = tmp33 + tmp15
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
