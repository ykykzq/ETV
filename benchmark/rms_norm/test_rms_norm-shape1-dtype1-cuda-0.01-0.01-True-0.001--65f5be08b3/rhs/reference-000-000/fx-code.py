


def forward(self, arg0_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [0], True);  pow_1 = None
    add_ = torch.ops.aten.add_.Scalar(mean, 0.0010000000474974513);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_);  add_ = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(mul, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0));  mul = None
    return _to_copy_1
    
