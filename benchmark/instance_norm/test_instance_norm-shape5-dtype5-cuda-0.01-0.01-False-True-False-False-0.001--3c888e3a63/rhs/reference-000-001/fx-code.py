


def forward(self, arg0_1, arg1_1, arg2_1):
    repeat = torch.ops.aten.repeat.default(arg2_1, [11]);  arg2_1 = None
    repeat_1 = torch.ops.aten.repeat.default(arg0_1, [11]);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [1, 44, 21]);  arg1_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0));  empty = None
    _native_batch_norm_legit = torch.ops.aten._native_batch_norm_legit.no_stats(view, repeat, repeat_1, True, 0.1, 0.001);  view = repeat = repeat_1 = None
    getitem = _native_batch_norm_legit[0]
    getitem_1 = _native_batch_norm_legit[1];  getitem_1 = None
    getitem_2 = _native_batch_norm_legit[2];  _native_batch_norm_legit = getitem_2 = None
    view_1 = torch.ops.aten.view.default(getitem, [11, 4, 21]);  getitem = None
    return view_1
    
