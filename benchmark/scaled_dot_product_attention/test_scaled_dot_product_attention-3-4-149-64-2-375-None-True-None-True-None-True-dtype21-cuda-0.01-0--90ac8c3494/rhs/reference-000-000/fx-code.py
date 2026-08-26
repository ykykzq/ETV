


def forward(self, arg0_1, arg1_1, arg2_1):
    mul = torch.ops.aten.mul.Scalar(arg1_1, 0.3535533905932738);  arg1_1 = None
    ones = torch.ops.aten.ones.default([149, 375], dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0))
    tril = torch.ops.aten.tril.default(ones);  ones = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.float32, device = device(type='cuda', index=0))
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(tril, scalar_tensor_1, scalar_tensor);  tril = scalar_tensor_1 = scalar_tensor = None
    unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [3, 2, 2, 375, 64]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [3, 4, 375, 64]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [3, 2, 2, 375, 64]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [3, 4, 375, 64]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.3535533905932738);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [3, 4, 149, 64]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [12, 149, 64]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [3, 4, 64, 375]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [12, 64, 375]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [3, 4, 149, 375]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, where);  _unsafe_view = where = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [3, 4, 149, 375]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [12, 149, 375]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [3, 4, 375, 64]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [12, 375, 64]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [3, 4, 149, 64]);  bmm_1 = None
    return _unsafe_view_1
    
