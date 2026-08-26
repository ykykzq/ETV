


def forward(self, arg0_1, arg1_1, arg2_1):
    addmm = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1, beta = -1.539877162998668, alpha = -0.8044050935608485);  arg0_1 = arg1_1 = arg2_1 = None
    return addmm
    
