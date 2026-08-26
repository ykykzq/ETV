
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_ceil_gather_lerp_masked_fill_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 486
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 3
    x0 = (xindex % 3)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = 3.0
    tmp21 = tmp19 * tmp20
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 - tmp24
    tmp26 = tl_math.abs(tmp25)
    tmp27 = 0.5
    tmp28 = tmp26 >= tmp27
    tmp29 = 1.0
    tmp30 = tmp25 - tmp29
    tmp31 = tl.where(tmp28, tmp30, tmp25)
    tmp32 = libdevice.ceil(tmp22)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.full([XBLOCK], 4, tl.int32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp33 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp33)
    tl.device_assert(((0 <= tmp37) & (tmp37 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp37 < 4")
    tmp39 = tl.load(in_ptr0 + (tmp37 + 4*x1), xmask, eviction_policy='evict_last')
    tmp40 = tmp23 + tmp34
    tmp41 = tmp23 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp23)
    tl.device_assert(((0 <= tmp42) & (tmp42 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp42 < 4")
    tmp44 = tl.load(in_ptr0 + (tmp42 + 4*x1), xmask, eviction_policy='evict_last')
    tmp45 = tmp39 - tmp44
    tmp46 = tmp31 * tmp45
    tmp47 = tl.where(tmp28, tmp39, tmp44)
    tmp48 = tmp46 + tmp47
    tl.store(in_out_ptr0 + (x2), tmp48, xmask)
