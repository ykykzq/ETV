


def forward(self, arg0_1, arg1_1, arg2_1):
    clamp = torch.ops.aten.clamp.Tensor(arg0_1, arg2_1, arg1_1);  arg0_1 = arg2_1 = arg1_1 = None
    return clamp
    
