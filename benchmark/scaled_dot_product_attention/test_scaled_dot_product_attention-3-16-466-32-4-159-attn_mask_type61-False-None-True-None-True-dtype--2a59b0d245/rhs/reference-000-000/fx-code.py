


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    mul = torch.ops.aten.mul.Scalar(arg2_1, 0.42044820762685725);  arg2_1 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
    expand = torch.ops.aten.expand.default(unsqueeze, [3, 4, 4, 159, 32]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [3, 16, 159, 32]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [3, 4, 4, 159, 32]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [3, 16, 159, 32]);  clone_1 = None
    transpose = torch.ops.aten.transpose.int(view, -2, -1);  view = None
    mul_1 = torch.ops.aten.mul.Scalar(transpose, 0.42044820762685725);  transpose = None
    expand_2 = torch.ops.aten.expand.default(mul, [3, 16, 466, 32]);  mul = None
    view_2 = torch.ops.aten.view.default(expand_2, [48, 466, 32]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(mul_1, [3, 16, 32, 159]);  mul_1 = None
    view_3 = torch.ops.aten.view.default(expand_3, [48, 32, 159]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [3, 16, 466, 159]);  bmm = None
    add = torch.ops.aten.add.Tensor(_unsafe_view, arg0_1);  _unsafe_view = arg0_1 = None
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1);  add = None
    expand_4 = torch.ops.aten.expand.default(_safe_softmax, [3, 16, 466, 159]);  _safe_softmax = None
    view_4 = torch.ops.aten.view.default(expand_4, [48, 466, 159]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_1, [3, 16, 159, 32]);  view_1 = None
    view_5 = torch.ops.aten.view.default(expand_5, [48, 159, 32]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm_1, [3, 16, 466, 32]);  bmm_1 = None
    return _unsafe_view_1
    
