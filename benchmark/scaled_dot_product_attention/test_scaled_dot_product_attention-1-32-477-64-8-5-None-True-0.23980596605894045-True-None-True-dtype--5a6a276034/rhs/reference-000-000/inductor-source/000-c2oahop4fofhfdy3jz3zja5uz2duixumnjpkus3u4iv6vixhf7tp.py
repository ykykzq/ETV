
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 274752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 477)
    tmp0 = tl.load(in_ptr0 + (5*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 5*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (2 + 5*x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (3 + 5*x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (4 + 5*x2), xmask, eviction_policy='evict_last')
    tmp1 = (-1)*x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = 0.0
    tmp7 = float("-inf")
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp9 == tmp7
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.int64)
    tmp13 = (tmp12 != 0)
    tmp15 = 1 + ((-1)*x0)
    tmp16 = tmp15 <= tmp2
    tmp17 = tmp16 & tmp4
    tmp18 = tl.where(tmp17, tmp6, tmp7)
    tmp19 = tmp14 + tmp18
    tmp20 = tmp19 == tmp7
    tmp21 = tmp20 == 0
    tmp22 = tmp21.to(tl.int64)
    tmp23 = (tmp22 != 0)
    tmp24 = tmp13 | tmp23
    tmp26 = 2 + ((-1)*x0)
    tmp27 = tmp26 <= tmp2
    tmp28 = tmp27 & tmp4
    tmp29 = tl.where(tmp28, tmp6, tmp7)
    tmp30 = tmp25 + tmp29
    tmp31 = tmp30 == tmp7
    tmp32 = tmp31 == 0
    tmp33 = tmp32.to(tl.int64)
    tmp34 = (tmp33 != 0)
    tmp35 = tmp24 | tmp34
    tmp37 = 3 + ((-1)*x0)
    tmp38 = tmp37 <= tmp2
    tmp39 = tmp38 & tmp4
    tmp40 = tl.where(tmp39, tmp6, tmp7)
    tmp41 = tmp36 + tmp40
    tmp42 = tmp41 == tmp7
    tmp43 = tmp42 == 0
    tmp44 = tmp43.to(tl.int64)
    tmp45 = (tmp44 != 0)
    tmp46 = tmp35 | tmp45
    tmp48 = 4 + ((-1)*x0)
    tmp49 = tmp48 <= tmp2
    tmp50 = tmp49 & tmp4
    tmp51 = tl.where(tmp50, tmp6, tmp7)
    tmp52 = tmp47 + tmp51
    tmp53 = tmp52 == tmp7
    tmp54 = tmp53 == 0
    tmp55 = tmp54.to(tl.int64)
    tmp56 = (tmp55 != 0)
    tmp57 = tmp46 | tmp56
    tmp58 = triton_helpers.maximum(tmp9, tmp19)
    tmp59 = triton_helpers.maximum(tmp58, tmp30)
    tmp60 = triton_helpers.maximum(tmp59, tmp41)
    tmp61 = triton_helpers.maximum(tmp60, tmp52)
    tmp62 = tmp9 - tmp61
    tmp63 = tl_math.exp(tmp62)
    tmp64 = tmp19 - tmp61
    tmp65 = tl_math.exp(tmp64)
    tmp66 = tmp63 + tmp65
    tmp67 = tmp30 - tmp61
    tmp68 = tl_math.exp(tmp67)
    tmp69 = tmp66 + tmp68
    tmp70 = tmp41 - tmp61
    tmp71 = tl_math.exp(tmp70)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp52 - tmp61
    tmp74 = tl_math.exp(tmp73)
    tmp75 = tmp72 + tmp74
    tl.store(out_ptr0 + (x2), tmp57, xmask)
    tl.store(out_ptr1 + (x2), tmp61, xmask)
    tl.store(out_ptr2 + (x2), tmp75, xmask)
