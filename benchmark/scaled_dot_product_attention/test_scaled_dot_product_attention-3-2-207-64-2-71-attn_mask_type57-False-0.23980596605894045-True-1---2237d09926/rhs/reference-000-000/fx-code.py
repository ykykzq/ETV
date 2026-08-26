


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.float16, device = device(type='cuda', index=0))
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(arg0_1, scalar_tensor_1, scalar_tensor);  arg0_1 = scalar_tensor_1 = scalar_tensor = None
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(where, [0, 1], 0.0);  where = None
    slice_1 = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 71);  constant_pad_nd = None
    expand = torch.ops.aten.expand.default(slice_1, [3, 2, 207, 71]);  slice_1 = None
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(arg2_1, arg1_1, arg3_1, expand, False, scale = 0.23980596605894045);  arg2_1 = arg1_1 = arg3_1 = expand = None
    getitem = _scaled_dot_product_efficient_attention[0]
    getitem_1 = _scaled_dot_product_efficient_attention[1];  getitem_1 = None
    getitem_2 = _scaled_dot_product_efficient_attention[2];  getitem_2 = None
    getitem_3 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = getitem_3 = None
    return getitem
    
