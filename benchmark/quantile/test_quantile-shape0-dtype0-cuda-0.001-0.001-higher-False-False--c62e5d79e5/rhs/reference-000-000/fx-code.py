


def forward(self, arg0_1):
    scalar_tensor = torch.ops.aten.scalar_tensor.default(0.723417723970258, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    sort = torch.ops.aten.sort.default(arg0_1);  arg0_1 = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = getitem_1 = None
    view = torch.ops.aten.view.default(getitem, [451]);  getitem = None
    mul = torch.ops.aten.mul.Scalar(scalar_tensor, 450);  scalar_tensor = None
    isnan = torch.ops.aten.isnan.default(view)
    any_1 = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand = torch.ops.aten.expand.default(mul, [1]);  mul = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(expand, any_1, 450);  expand = any_1 = None
    ceil_ = torch.ops.aten.ceil_.default(masked_fill);  masked_fill = None
    _to_copy = torch.ops.aten._to_copy.default(ceil_, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  ceil_ = None
    gather = torch.ops.aten.gather.default(view, -1, _to_copy);  view = _to_copy = None
    squeeze_ = torch.ops.aten.squeeze_.dim(gather, -1);  gather = None
    return squeeze_
    
