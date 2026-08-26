


def forward(self, arg0_1, arg1_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [3, 2, 1], True);  pow_1 = None
    add_ = torch.ops.aten.add_.Scalar(mean, 0.0010000000474974513);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_);  add_ = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, rsqrt);  arg0_1 = rsqrt = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
    return mul_1
    
