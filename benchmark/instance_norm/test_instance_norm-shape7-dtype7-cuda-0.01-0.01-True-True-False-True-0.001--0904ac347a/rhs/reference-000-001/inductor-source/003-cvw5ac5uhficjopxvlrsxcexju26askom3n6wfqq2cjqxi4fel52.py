
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_2', 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_2(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp16 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp1 = 0.1
    tmp2 = tmp0 * tmp1
    tmp4 = 0.9
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp6
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 + tmp6
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp15 + tmp20
    tmp22 = 3.0
    tmp23 = (tmp21 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
