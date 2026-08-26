


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    repeat = torch.ops.aten.repeat.default(arg0_1, [11]);  arg0_1 = None
    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [11])
    repeat_2 = torch.ops.aten.repeat.default(arg3_1, [11])
    view = torch.ops.aten.view.default(arg1_1, [1, 44, 21]);  arg1_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0));  empty = None
    _native_batch_norm_legit = torch.ops.aten._native_batch_norm_legit.default(view, None, repeat, repeat_1, repeat_2, True, 0.1, 0.001);  view = repeat = None
    getitem = _native_batch_norm_legit[0]
    getitem_1 = _native_batch_norm_legit[1];  getitem_1 = None
    getitem_2 = _native_batch_norm_legit[2];  _native_batch_norm_legit = getitem_2 = None
    alias = torch.ops.aten.alias.default(arg2_1);  arg2_1 = None
    view_1 = torch.ops.aten.view.default(repeat_1, [11, 4]);  repeat_1 = None
    mean = torch.ops.aten.mean.dim(view_1, [0]);  view_1 = None
    copy_ = torch.ops.aten.copy_.default(alias, mean);  alias = mean = copy_ = None
    alias_1 = torch.ops.aten.alias.default(arg3_1);  arg3_1 = None
    view_2 = torch.ops.aten.view.default(repeat_2, [11, 4]);  repeat_2 = None
    mean_1 = torch.ops.aten.mean.dim(view_2, [0]);  view_2 = None
    copy__1 = torch.ops.aten.copy_.default(alias_1, mean_1);  alias_1 = mean_1 = copy__1 = None
    view_3 = torch.ops.aten.view.default(getitem, [11, 4, 21]);  getitem = None
    return view_3
    
