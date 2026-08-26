


def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0));  empty = None
    _native_batch_norm_legit = torch.ops.aten._native_batch_norm_legit.no_stats(arg0_1, None, None, True, 0.1, 1e-05);  arg0_1 = None
    getitem = _native_batch_norm_legit[0]
    getitem_1 = _native_batch_norm_legit[1];  getitem_1 = None
    getitem_2 = _native_batch_norm_legit[2];  _native_batch_norm_legit = getitem_2 = None
    return getitem
    
