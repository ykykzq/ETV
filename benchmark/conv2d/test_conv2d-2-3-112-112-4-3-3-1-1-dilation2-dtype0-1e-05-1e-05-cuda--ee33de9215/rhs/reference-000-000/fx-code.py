


def forward(self, arg0_1, arg1_1, arg2_1):
    convolution = torch.ops.aten.convolution.default(arg1_1, arg2_1, arg0_1, [1, 1], [1, 1], [2, 3], False, [0, 0], 1);  arg1_1 = arg2_1 = arg0_1 = None
    return convolution
    
