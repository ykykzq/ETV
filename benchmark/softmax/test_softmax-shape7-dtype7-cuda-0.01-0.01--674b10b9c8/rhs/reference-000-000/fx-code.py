


def forward(self, arg0_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0));  arg0_1 = None
    _softmax = torch.ops.aten._softmax.default(_to_copy, 2, False);  _to_copy = None
    return _softmax
    
