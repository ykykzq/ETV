


def forward(self, arg0_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2.0);  arg0_1 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(pow_1, -2);  pow_1 = None
    avg_pool2d = torch.ops.aten.avg_pool2d.default(unsqueeze, [1, 2], [1, 2]);  unsqueeze = None
    squeeze = torch.ops.aten.squeeze.dim(avg_pool2d, -2);  avg_pool2d = None
    sign = torch.ops.aten.sign.default(squeeze)
    abs_1 = torch.ops.aten.abs.default(squeeze);  squeeze = None
    relu = torch.ops.aten.relu.default(abs_1);  abs_1 = None
    mul = torch.ops.aten.mul.Tensor(sign, relu);  sign = relu = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 2);  mul = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(mul_1, 0.5);  mul_1 = None
    return pow_2
    
