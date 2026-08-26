


def forward(self, arg0_1, arg1_1):
    bmm = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return bmm
    
