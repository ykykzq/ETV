


def forward(self, arg0_1, arg1_1):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(arg1_1, [22], None, arg0_1, 1e-08);  arg1_1 = arg0_1 = None
    getitem = native_layer_norm[0]
    getitem_1 = native_layer_norm[1];  getitem_1 = None
    getitem_2 = native_layer_norm[2];  native_layer_norm = getitem_2 = None
    return getitem
    
