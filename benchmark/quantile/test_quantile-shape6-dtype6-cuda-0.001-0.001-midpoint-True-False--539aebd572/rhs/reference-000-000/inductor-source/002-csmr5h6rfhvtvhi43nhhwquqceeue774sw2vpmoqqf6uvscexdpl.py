
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_any_isnan_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_any_isnan_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 108
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 6*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 6*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 6*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (4 + 6*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (5 + 6*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = (tmp2 != 0)
    tmp5 = libdevice.isnan(tmp4).to(tl.int1)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = (tmp6 != 0)
    tmp8 = tmp3 | tmp7
    tmp10 = libdevice.isnan(tmp9).to(tl.int1)
    tmp11 = tmp10.to(tl.int64)
    tmp12 = (tmp11 != 0)
    tmp13 = tmp8 | tmp12
    tmp15 = libdevice.isnan(tmp14).to(tl.int1)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = (tmp16 != 0)
    tmp18 = tmp13 | tmp17
    tmp20 = libdevice.isnan(tmp19).to(tl.int1)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = (tmp21 != 0)
    tmp23 = tmp18 | tmp22
    tmp25 = libdevice.isnan(tmp24).to(tl.int1)
    tmp26 = tmp25.to(tl.int64)
    tmp27 = (tmp26 != 0)
    tmp28 = tmp23 | tmp27
    tl.store(out_ptr0 + (x0), tmp28, xmask)
