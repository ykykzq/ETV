


def forward(self, arg0_1, arg1_1):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(arg0_1, [11, 1, 3], arg1_1, None, 1e-08);  arg0_1 = arg1_1 = None
    getitem = native_layer_norm[0]
    getitem_1 = native_layer_norm[1];  getitem_1 = None
    getitem_2 = native_layer_norm[2];  native_layer_norm = getitem_2 = None
    return getitem
    
