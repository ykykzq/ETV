
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12*x1), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (2 + x0 + 12*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 12*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (6 + x0 + 12*x1), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (8 + x0 + 12*x1), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (10 + x0 + 12*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float64)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float64)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float64)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp15 = tmp14.to(tl.float64)
    tmp16 = triton_helpers.maximum(tmp13, tmp15)
    tmp17 = tmp1 - tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tmp3 - tmp16
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp6 - tmp16
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp9 - tmp16
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp12 - tmp16
    tmp29 = libdevice.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp15 - tmp16
    tmp32 = libdevice.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr0 + (x2), tmp16, xmask)
    tl.store(out_ptr1 + (x2), tmp33, xmask)
