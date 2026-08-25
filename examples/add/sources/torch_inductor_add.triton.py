import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
