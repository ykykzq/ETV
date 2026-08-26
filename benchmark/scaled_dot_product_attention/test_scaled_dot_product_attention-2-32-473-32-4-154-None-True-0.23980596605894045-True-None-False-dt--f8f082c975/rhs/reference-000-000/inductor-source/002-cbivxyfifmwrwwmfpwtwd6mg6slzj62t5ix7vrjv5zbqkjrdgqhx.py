
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 57028608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4752384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 156) % 476)
    x0 = (xindex % 156)
    x2 = xindex // 74256
    x3 = (xindex % 74256)
    tmp0 = x1
    tmp1 = tl.full([1], 473, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 154, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (x1 + 473*((x2 % 32)) + 15232*(x2 // 32)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tmp7 == 0
    tmp9 = tl.load(in_ptr1 + (x0 + 154*x1 + 72842*x2), tmp6 & xmask, other=0.0)
    tmp10 = x0 + ((-1)*x1)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 <= tmp11
    tmp13 = tl.full([1], True, tl.int1)
    tmp14 = tmp12 & tmp13
    tmp15 = 0.0
    tmp16 = float("-inf")
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tmp9 + tmp17
    tmp19 = tl.load(in_ptr2 + (x1 + 473*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tl.load(in_ptr3 + (x1 + 473*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = (tmp21 / tmp22)
    tmp24 = tl.where(tmp8, tmp15, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tl.store(out_ptr0 + (x3 + 74272*x2), tmp26, xmask)
