


def forward(self, arg0_1, arg1_1, arg2_1):
    mul = torch.ops.aten.mul.Scalar(arg1_1, 0.42044820762685725);  arg1_1 = None
    ones = torch.ops.aten.ones.default([185, 71], dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0))
    tril = torch.ops.aten.tril.default(ones);  ones = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.float32, device = device(type='cuda', index=0))
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(tril, scalar_tensor_1, scalar_tensor);  tril = scalar_tensor_1 = scalar_tensor = None
    unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [1, 16, 2, 71, 32]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [1, 32, 71, 32]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [1, 16, 2, 71, 32]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [1, 32, 71, 32]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.42044820762685725);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [1, 32, 185, 32]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [32, 185, 32]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [1, 32, 32, 71]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [32, 32, 71]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [1, 32, 185, 71]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, where);  _unsafe_view = where = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [1, 32, 185, 71]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [32, 185, 71]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [1, 32, 71, 32]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [32, 71, 32]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [1, 32, 185, 32]);  bmm_1 = None
    return _unsafe_view_1
    
