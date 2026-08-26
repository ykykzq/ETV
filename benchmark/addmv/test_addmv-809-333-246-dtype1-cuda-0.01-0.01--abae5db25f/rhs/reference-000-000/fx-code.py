


def forward(self, arg0_1, arg1_1, arg2_1):
    addmv = torch.ops.aten.addmv.default(arg0_1, arg1_1, arg2_1, beta = 0.37707535476743975, alpha = -1.7830420547213741);  arg0_1 = arg1_1 = arg2_1 = None
    return addmv
    
