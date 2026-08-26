


def forward(self, arg0_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 1.0);  arg0_1 = None
    avg_pool2d = torch.ops.aten.avg_pool2d.default(pow_1, [4, 3]);  pow_1 = None
    sign = torch.ops.aten.sign.default(avg_pool2d)
    abs_1 = torch.ops.aten.abs.default(avg_pool2d);  avg_pool2d = None
    relu = torch.ops.aten.relu.default(abs_1);  abs_1 = None
    mul = torch.ops.aten.mul.Tensor(sign, relu);  sign = relu = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 12);  mul = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(mul_1, 1.0);  mul_1 = None
    return pow_2
    
