
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_any_gather_isnan_masked_fill_round_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_any_gather_isnan_masked_fill_round_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = (tmp2 != 0)
    tmp5 = libdevice.isnan(tmp4).to(tl.int1)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = (tmp6 != 0)
    tmp8 = tmp3 | tmp7
    tmp9 = 1.0
    tmp10 = 0.41425836241315184
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = libdevice.nearbyint(tmp11)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([XBLOCK], 2, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tmp17) & (tmp17 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 2")
    tmp19 = tl.load(in_ptr0 + (tmp17 + 2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp19, xmask)
