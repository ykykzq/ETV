


def forward(self, arg0_1):
    scalar_tensor = torch.ops.aten.scalar_tensor.default(0.02919232317430631, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    view = torch.ops.aten.view.default(arg0_1, [989]);  arg0_1 = None
    sort = torch.ops.aten.sort.default(view);  view = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view_1 = torch.ops.aten.view.default(getitem, [989]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(scalar_tensor, 988);  scalar_tensor = None
    isnan = torch.ops.aten.isnan.default(view_1)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [1]);  mul = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, any_1, 988);  expand = any_1 = None
    _to_copy = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    gather = torch.ops.aten.gather.default(view_1, -1, _to_copy)
    sub = torch.ops.aten.sub.Tensor(masked_fill, _to_copy);  _to_copy = None
    ceil_ = torch.ops.aten.ceil_.default(masked_fill);  masked_fill = None
    _to_copy_1 = torch.ops.aten._to_copy.default(ceil_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  ceil_ = None
    gather_1 = torch.ops.aten.gather.default(view_1, -1, _to_copy_1);  view_1 = _to_copy_1 = None
    lerp_ = torch.ops.aten.lerp_.Tensor(gather, gather_1, sub);  gather = gather_1 = sub = None
    squeeze_ = torch.ops.aten.squeeze_.dim(lerp_, -1);  lerp_ = None
    return squeeze_
    
