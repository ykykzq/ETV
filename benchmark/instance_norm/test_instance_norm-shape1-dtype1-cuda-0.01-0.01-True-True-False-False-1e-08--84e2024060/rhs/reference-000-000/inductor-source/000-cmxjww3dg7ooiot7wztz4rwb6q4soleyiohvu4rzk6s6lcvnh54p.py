
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr2': '*fp16', 'out_ptr4': '*fp16', 'out_ptr6': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_0', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '58367EC428ADC15B85CB9CF138B580A95422950F92C44A257A91C53B1E76C9F7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 6108}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 1018
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp26 = tl.broadcast_to(tmp25, [R0_BLOCK])
    tmp29 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [R0_BLOCK])
    tmp36 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp37 = tl.broadcast_to(tmp36, [1])
    tmp50 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [1])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 1018, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp1 - tmp11
    tmp19 = 1018.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-08
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp24 * tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 0.1
    tmp35 = tmp11 * tmp34
    tmp38 = 0.9
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp35 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 1.0
    tmp45 = (tmp43 / tmp44)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = 1.0009832841691249
    tmp48 = tmp20 * tmp47
    tmp49 = tmp48 * tmp34
    tmp52 = tmp51 * tmp38
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp49 + tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = (tmp56 / tmp44)
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp33, r0_mask)
    tl.store(out_ptr4 + (tl.full([1], 0, tl.int32)), tmp46, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp58, None)
