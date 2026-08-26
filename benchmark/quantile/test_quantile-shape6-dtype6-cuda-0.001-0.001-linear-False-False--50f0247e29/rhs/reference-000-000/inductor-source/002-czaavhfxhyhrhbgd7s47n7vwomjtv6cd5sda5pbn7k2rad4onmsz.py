
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_any_ceil_gather_isnan_lerp_masked_fill_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 3*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 3*x0), xmask, eviction_policy='evict_last')
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
    tmp14 = 2.0
    tmp15 = 1.3159825876209328
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 - tmp18
    tmp20 = tl_math.abs(tmp19)
    tmp21 = 0.5
    tmp22 = tmp20 >= tmp21
    tmp23 = 1.0
    tmp24 = tmp19 - tmp23
    tmp25 = tl.where(tmp22, tmp24, tmp19)
    tmp26 = libdevice.ceil(tmp16)
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tl.full([XBLOCK], 3, tl.int32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 3)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 3")
    tmp33 = tl.load(in_ptr0 + (tmp31 + 3*x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp17 + tmp28
    tmp35 = tmp17 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp17)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 3)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 3")
    tmp38 = tl.load(in_ptr0 + (tmp36 + 3*x0), xmask, eviction_policy='evict_last')
    tmp39 = tmp33 - tmp38
    tmp40 = tmp25 * tmp39
    tmp41 = tl.where(tmp22, tmp33, tmp38)
    tmp42 = tmp40 + tmp41
    tl.store(in_out_ptr0 + (x0), tmp42, xmask)
