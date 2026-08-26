


def forward(self, arg0_1, arg1_1, arg2_1):
    addmm = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1, beta = 0.40419496995615106, alpha = 1.0129277330529662);  arg0_1 = arg1_1 = arg2_1 = None
    return addmm
    
