


def forward(self, arg0_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
    mean = torch.ops.aten.mean.default(_to_copy);  _to_copy = None
    return mean
    
