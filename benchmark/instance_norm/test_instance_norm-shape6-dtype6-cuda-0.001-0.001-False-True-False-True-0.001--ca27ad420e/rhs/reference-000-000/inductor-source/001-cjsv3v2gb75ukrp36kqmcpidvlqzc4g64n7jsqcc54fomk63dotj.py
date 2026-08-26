
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 3*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 3*x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (((x2 // 3) % 29)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp3 - tmp1
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 - tmp1
    tmp8 = tmp7 * tmp7
    tmp9 = tmp5 + tmp8
    tmp11 = tmp10 - tmp1
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 3.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 0.001
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp2 * tmp18
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
