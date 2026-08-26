


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    _to_copy = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
    mul = torch.ops.aten.mul.Scalar(_to_copy, 0.48969987345203636);  _to_copy = None
    unsqueeze = torch.ops.aten.unsqueeze.default(_to_copy_1, 2);  _to_copy_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [2, 4, 4, 340, 64]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [2, 16, 340, 64]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(_to_copy_2, 2);  _to_copy_2 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [2, 4, 4, 340, 64]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [2, 16, 340, 64]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.48969987345203636);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [2, 16, 281, 64]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [32, 281, 64]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [2, 16, 64, 340]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [32, 64, 340]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [2, 16, 281, 340]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, arg0_1);  _unsafe_view = arg0_1 = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    _to_copy_3 = torch.ops.aten._to_copy.default(_safe_softmax, dtype = torch.float16);  _to_copy_3 = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [2, 16, 281, 340]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [32, 281, 340]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [2, 16, 340, 64]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [32, 340, 64]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [2, 16, 281, 64]);  bmm_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(_unsafe_view_1, dtype = torch.float16);  _unsafe_view_1 = None
    return _to_copy_4
    
