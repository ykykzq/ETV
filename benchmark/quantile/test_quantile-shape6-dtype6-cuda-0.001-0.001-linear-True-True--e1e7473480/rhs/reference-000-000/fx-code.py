


def forward(self, arg0_1, arg1_1):
    view = torch.ops.aten.view.default(arg0_1, [648]);  arg0_1 = None
    sort = torch.ops.aten.sort.default(view);  view = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view_1 = torch.ops.aten.view.default(getitem, [1, 1, 1, 1, 648]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(arg1_1, 647);  arg1_1 = None
    isnan = torch.ops.aten.isnan.default(view_1)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [1, 1, 1, 1, 1]);  mul = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, any_1, 647);  expand = any_1 = None
    _to_copy = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    gather = torch.ops.aten.gather.default(view_1, -1, _to_copy)
    sub = torch.ops.aten.sub.Tensor(masked_fill, _to_copy);  _to_copy = None
    ceil_ = torch.ops.aten.ceil_.default(masked_fill);  masked_fill = None
    _to_copy_1 = torch.ops.aten._to_copy.default(ceil_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  ceil_ = None
    gather_1 = torch.ops.aten.gather.default(view_1, -1, _to_copy_1);  view_1 = _to_copy_1 = None
    lerp_ = torch.ops.aten.lerp_.Tensor(gather, gather_1, sub);  gather = gather_1 = sub = None
    unsqueeze_ = torch.ops.aten.unsqueeze_.default(lerp_, 0);  lerp_ = None
    transpose_ = torch.ops.aten.transpose_.default(unsqueeze_, 0, -1);  unsqueeze_ = None
    squeeze_ = torch.ops.aten.squeeze_.dim(transpose_, -1);  transpose_ = None
    return squeeze_
    
