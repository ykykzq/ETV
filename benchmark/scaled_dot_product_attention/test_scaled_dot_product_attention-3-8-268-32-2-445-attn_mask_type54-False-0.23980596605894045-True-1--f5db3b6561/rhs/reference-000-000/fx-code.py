


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.float16, device = device(type='cuda', index=0))
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(arg0_1, scalar_tensor_1, scalar_tensor);  arg0_1 = scalar_tensor_1 = scalar_tensor = None
    _to_copy = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
    mul = torch.ops.aten.mul.Scalar(_to_copy, 0.48969987345203636);  _to_copy = None
    unsqueeze = torch.ops.aten.unsqueeze.default(_to_copy_1, 2);  _to_copy_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [3, 2, 4, 445, 32]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [3, 8, 445, 32]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(_to_copy_2, 2);  _to_copy_2 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [3, 2, 4, 445, 32]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [3, 8, 445, 32]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.48969987345203636);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [3, 8, 268, 32]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [24, 268, 32]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [3, 8, 32, 445]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [24, 32, 445]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [3, 8, 268, 445]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, where);  _unsafe_view = where = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    _to_copy_3 = torch.ops.aten._to_copy.default(_safe_softmax, dtype = torch.float16);  _to_copy_3 = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [3, 8, 268, 445]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [24, 268, 445]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [3, 8, 445, 32]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [24, 445, 32]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [3, 8, 268, 32]);  bmm_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(_unsafe_view_1, dtype = torch.float16);  _unsafe_view_1 = None
    return _to_copy_4
    
