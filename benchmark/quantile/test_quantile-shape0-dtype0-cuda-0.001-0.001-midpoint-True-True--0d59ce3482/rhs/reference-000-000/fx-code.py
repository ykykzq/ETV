


def forward(self, arg0_1):
    scalar_tensor = torch.ops.aten.scalar_tensor.default(0.9709430115908664, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    sort = torch.ops.aten.sort.default(arg0_1);  arg0_1 = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view = torch.ops.aten.view.default(getitem, [1, 451]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(scalar_tensor, 450);  scalar_tensor = None
    isnan = torch.ops.aten.isnan.default(view)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [1, 1]);  mul = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, any_1, 450);  expand = any_1 = None
    _to_copy = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    gather = torch.ops.aten.gather.default(view, -1, _to_copy);  _to_copy = None
    full_like = torch.ops.aten.full_like.default(masked_fill, 0.5)
    ceil_ = torch.ops.aten.ceil_.default(masked_fill);  masked_fill = None
    _to_copy_1 = torch.ops.aten._to_copy.default(ceil_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  ceil_ = None
    gather_1 = torch.ops.aten.gather.default(view, -1, _to_copy_1);  view = _to_copy_1 = None
    lerp_ = torch.ops.aten.lerp_.Tensor(gather, gather_1, full_like);  gather = gather_1 = full_like = None
    squeeze_ = torch.ops.aten.squeeze_.dim(lerp_, -1);  lerp_ = None
    return squeeze_
    
