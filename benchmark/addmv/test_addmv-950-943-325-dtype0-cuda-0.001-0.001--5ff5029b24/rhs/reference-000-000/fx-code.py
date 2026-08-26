


def forward(self, arg0_1, arg1_1, arg2_1):
    addmv = torch.ops.aten.addmv.default(arg0_1, arg1_1, arg2_1, beta = -0.5311862414354188, alpha = -1.2815177460872722);  arg0_1 = arg1_1 = arg2_1 = None
    return addmv
    
