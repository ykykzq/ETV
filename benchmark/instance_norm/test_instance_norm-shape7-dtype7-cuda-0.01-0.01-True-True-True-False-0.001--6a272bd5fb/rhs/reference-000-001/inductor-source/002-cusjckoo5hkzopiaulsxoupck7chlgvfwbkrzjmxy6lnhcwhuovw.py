
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_1', 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_1(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp22 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp1 = 56.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1.018181818181818
    tmp4 = tmp2 * tmp3
    tmp5 = 0.1
    tmp6 = tmp4 * tmp5
    tmp8 = 0.9
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = (tmp14 / tmp1)
    tmp16 = tmp15 * tmp3
    tmp17 = tmp16 * tmp5
    tmp18 = tmp17 + tmp10
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 + tmp20
    tmp23 = (tmp22 / tmp1)
    tmp24 = tmp23 * tmp3
    tmp25 = tmp24 * tmp5
    tmp26 = tmp25 + tmp10
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp21 + tmp28
    tmp30 = 3.0
    tmp31 = (tmp29 / tmp30)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp32, xmask)
