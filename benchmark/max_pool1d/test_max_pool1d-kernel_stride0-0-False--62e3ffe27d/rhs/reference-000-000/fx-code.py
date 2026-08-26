


def forward(self, arg0_1):
    unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, -2);  arg0_1 = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(unsqueeze, [1, 2], [1, 2]);  unsqueeze = None
    getitem = max_pool2d_with_indices[0]
    getitem_1 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    squeeze = torch.ops.aten.squeeze.dim(getitem, -2);  getitem = None
    squeeze_1 = torch.ops.aten.squeeze.dim(getitem_1, -2);  getitem_1 = squeeze_1 = None
    return squeeze
    
