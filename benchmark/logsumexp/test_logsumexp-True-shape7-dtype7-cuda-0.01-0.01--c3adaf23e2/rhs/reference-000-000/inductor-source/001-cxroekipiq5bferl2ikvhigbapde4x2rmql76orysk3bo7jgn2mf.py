
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (192 + x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (384 + x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (576 + x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp11 = tl_math.abs(tmp10)
    tmp12 = float("inf")
    tmp13 = tmp11 == tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = tmp1 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp3 - tmp15
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp6 - tmp15
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp9 - tmp15
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tl_math.log(tmp26)
    tmp28 = tmp27 + tmp15
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp29, xmask)
