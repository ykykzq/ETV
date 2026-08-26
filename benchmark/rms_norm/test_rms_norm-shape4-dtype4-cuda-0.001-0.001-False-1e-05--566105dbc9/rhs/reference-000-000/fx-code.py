


def forward(self, arg0_1, arg1_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [2, 1, 0], True);  pow_1 = None
    add_ = torch.ops.aten.add_.Scalar(mean, 9.999999747378752e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_);  add_ = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, rsqrt);  arg0_1 = rsqrt = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
    return mul_1
    
