


def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return mm
    
