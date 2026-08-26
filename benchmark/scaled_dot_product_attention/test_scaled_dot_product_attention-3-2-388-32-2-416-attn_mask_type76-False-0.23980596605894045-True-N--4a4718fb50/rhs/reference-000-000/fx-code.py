


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    expand = torch.ops.aten.expand.default(arg0_1, [3, 2, 388, 416]);  arg0_1 = None
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(arg2_1, arg1_1, arg3_1, expand, False, scale = 0.23980596605894045);  arg2_1 = arg1_1 = arg3_1 = expand = None
    getitem = _scaled_dot_product_efficient_attention[0]
    getitem_1 = _scaled_dot_product_efficient_attention[1];  getitem_1 = None
    getitem_2 = _scaled_dot_product_efficient_attention[2];  getitem_2 = None
    getitem_3 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = getitem_3 = None
    return getitem
    
