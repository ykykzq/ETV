


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    mul = torch.ops.aten.mul.Scalar(arg2_1, 0.48969987345203636);  arg2_1 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [4, 8, 2, 308, 32]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [4, 16, 308, 32]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [4, 8, 2, 308, 32]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [4, 16, 308, 32]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.48969987345203636);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [4, 16, 117, 32]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [64, 117, 32]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [4, 16, 32, 308]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [64, 32, 308]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [4, 16, 117, 308]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, arg0_1);  _unsafe_view = arg0_1 = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [4, 16, 117, 308]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [64, 117, 308]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [4, 16, 308, 32]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [64, 308, 32]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [4, 16, 117, 32]);  bmm_1 = None
    return _unsafe_view_1
    
