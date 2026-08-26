


def forward(self, arg0_1, arg1_1, arg2_1):
    unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
    slice_1 = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 9223372036854775807);  unsqueeze = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(slice_1, 2);  slice_1 = None
    slice_2 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
    slice_3 = torch.ops.aten.slice.Tensor(unsqueeze_2, 1, 0, 9223372036854775807);  unsqueeze_2 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(slice_3, 2);  slice_3 = None
    slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_3, 3, 0, 9223372036854775807);  unsqueeze_3 = None
    view = torch.ops.aten.view.default(arg1_1, [1, 1, 1, 32, 2]);  arg1_1 = None
    select = torch.ops.aten.select.int(view, 4, 0)
    select_1 = torch.ops.aten.select.int(view, 4, 1);  view = None
    mul = torch.ops.aten.mul.Tensor(select, slice_4)
    mul_1 = torch.ops.aten.mul.Tensor(select_1, slice_2)
    sub = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(select, slice_2);  select = slice_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(select_1, slice_4);  select_1 = slice_4 = None
    add = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
    stack = torch.ops.aten.stack.default([sub, add], -1);  sub = add = None
    view_1 = torch.ops.aten.view.default(stack, [1, 1, 1, 64]);  stack = None
    return view_1
    
