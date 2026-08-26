


def forward(self, arg0_1):
    logsumexp = torch.ops.aten.logsumexp.default(arg0_1, [1], True);  arg0_1 = None
    return logsumexp
    
