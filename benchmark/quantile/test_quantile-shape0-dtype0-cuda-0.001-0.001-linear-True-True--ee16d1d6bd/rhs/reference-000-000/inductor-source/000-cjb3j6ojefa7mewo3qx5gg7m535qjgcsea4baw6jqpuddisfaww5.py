
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = 450.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.where(tmp1, tmp3, tmp4)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = 0.5
    tmp11 = tmp9 >= tmp10
    tmp12 = 1.0
    tmp13 = tmp8 - tmp12
    tmp14 = tl.where(tmp11, tmp13, tmp8)
    tmp15 = libdevice.ceil(tmp5)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = tl.full([XBLOCK], 451, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tl.device_assert(((0 <= tmp20) & (tmp20 < 451)) | ~(xmask), "index out of bounds: 0 <= tmp20 < 451")
    tmp22 = tl.load(in_ptr2 + (tmp20), xmask, eviction_policy='evict_last')
    tmp23 = tmp6 + tmp17
    tmp24 = tmp6 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp6)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 451)) | ~(xmask), "index out of bounds: 0 <= tmp25 < 451")
    tmp27 = tl.load(in_ptr2 + (tmp25), xmask, eviction_policy='evict_last')
    tmp28 = tmp22 - tmp27
    tmp29 = tmp14 * tmp28
    tmp30 = tl.where(tmp11, tmp22, tmp27)
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x0), tmp31, xmask)
