


def forward(self, arg0_1, arg1_1):
    bincount = torch.ops.aten.bincount.default(arg0_1, arg1_1, 10);  arg0_1 = arg1_1 = None
    return bincount
    
