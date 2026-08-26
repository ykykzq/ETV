
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 484}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_0(in_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = triton_helpers.maximum(tmp13, tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = triton_helpers.maximum(tmp16, tmp18)
    tmp20 = tl_math.abs(tmp19)
    tmp21 = float("inf")
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp19)
    tmp25 = tmp1 - tmp24
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp3 - tmp24
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp6 - tmp24
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp9 - tmp24
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp32 + tmp34
    tmp36 = tmp12 - tmp24
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp15 - tmp24
    tmp40 = tl_math.exp(tmp39)
    tmp41 = tmp38 + tmp40
    tmp42 = tmp18 - tmp24
    tmp43 = tl_math.exp(tmp42)
    tmp44 = tmp41 + tmp43
    tmp45 = tl_math.log(tmp44)
    tmp46 = tmp45 + tmp24
    tmp47 = tmp46.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp47, xmask)
