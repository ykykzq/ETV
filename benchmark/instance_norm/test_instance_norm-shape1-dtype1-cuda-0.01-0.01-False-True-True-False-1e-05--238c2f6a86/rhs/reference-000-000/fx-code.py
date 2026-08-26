


def forward(self, arg0_1, arg1_1):
    repeat = torch.ops.aten.repeat.default(arg0_1, [1]);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [1, 1, 1018]);  arg1_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0));  empty = None
    _native_batch_norm_legit = torch.ops.aten._native_batch_norm_legit.no_stats(view, None, repeat, True, 0.1, 1e-05);  view = repeat = None
    getitem = _native_batch_norm_legit[0]
    getitem_1 = _native_batch_norm_legit[1];  getitem_1 = None
    getitem_2 = _native_batch_norm_legit[2];  _native_batch_norm_legit = getitem_2 = None
    view_1 = torch.ops.aten.view.default(getitem, [1, 1, 1018]);  getitem = None
    return view_1
    
