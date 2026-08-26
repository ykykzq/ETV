


def forward(self, arg0_1, arg1_1):
    unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    transpose = torch.ops.aten.transpose.int(unsqueeze, 0, -1);  unsqueeze = None
    sort = torch.ops.aten.sort.default(transpose);  transpose = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view = torch.ops.aten.view.default(getitem, [8, 25, 2]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(arg1_1, 1);  arg1_1 = None
    isnan = torch.ops.aten.isnan.default(view)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [8, 25, 1]);  mul = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, any_1, 1);  expand = any_1 = None
    _to_copy = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    gather = torch.ops.aten.gather.default(view, -1, _to_copy);  _to_copy = None
    full_like = torch.ops.aten.full_like.default(masked_fill, 0.5)
    ceil_ = torch.ops.aten.ceil_.default(masked_fill);  masked_fill = None
    _to_copy_1 = torch.ops.aten._to_copy.default(ceil_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  ceil_ = None
    gather_1 = torch.ops.aten.gather.default(view, -1, _to_copy_1);  view = _to_copy_1 = None
    lerp_ = torch.ops.aten.lerp_.Tensor(gather, gather_1, full_like);  gather = gather_1 = full_like = None
    unsqueeze_ = torch.ops.aten.unsqueeze_.default(lerp_, 0);  lerp_ = None
    transpose_ = torch.ops.aten.transpose_.default(unsqueeze_, 0, -1);  unsqueeze_ = None
    squeeze_ = torch.ops.aten.squeeze_.dim(transpose_, -1);  transpose_ = None
    return squeeze_
    
