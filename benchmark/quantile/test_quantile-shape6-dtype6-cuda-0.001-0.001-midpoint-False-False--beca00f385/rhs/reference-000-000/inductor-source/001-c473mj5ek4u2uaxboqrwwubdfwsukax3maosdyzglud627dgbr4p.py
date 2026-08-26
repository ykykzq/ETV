
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 8.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl.where(tmp0, tmp2, tmp3)
    tmp5 = libdevice.ceil(tmp4)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.full([XBLOCK], 9, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 9")
    tmp12 = tl.load(in_ptr2 + (tmp10 + 9*x1), xmask, eviction_policy='evict_last')
    tmp13 = tmp4.to(tl.int64)
    tmp14 = tmp13 + tmp7
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 9)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 9")
    tmp18 = tl.load(in_ptr2 + (tmp16 + 9*x1), xmask, eviction_policy='evict_last')
    tmp19 = tmp12 - tmp18
    tmp20 = -0.5
    tmp21 = tmp20 * tmp19
    tmp22 = tl.full([1], True, tl.int1)
    tmp23 = tl.where(tmp22, tmp12, tmp18)
    tmp24 = tmp21 + tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
