


def forward(self, arg0_1, arg1_1):
    unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    transpose = torch.ops.aten.transpose.int(unsqueeze, 0, -1);  unsqueeze = None
    sort = torch.ops.aten.sort.default(transpose);  transpose = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view = torch.ops.aten.view.default(getitem, [23, 43]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(arg1_1, 42);  arg1_1 = None
    isnan = torch.ops.aten.isnan.default(view)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [23, 3]);  mul = None
    expand_1 = torch.ops.aten.expand.default(any_1, [23, 3]);  any_1 = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, expand_1, 42);  expand = expand_1 = None
    round_ = torch.ops.aten.round_.default(masked_fill);  masked_fill = None
    _to_copy = torch.ops.aten._to_copy.default(round_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  round_ = None
    gather = torch.ops.aten.gather.default(view, -1, _to_copy);  view = _to_copy = None
    unsqueeze_ = torch.ops.aten.unsqueeze_.default(gather, 0);  gather = None
    transpose_ = torch.ops.aten.transpose_.default(unsqueeze_, 0, -1);  unsqueeze_ = None
    squeeze_ = torch.ops.aten.squeeze_.dim(transpose_, -1);  transpose_ = None
    return squeeze_
    
