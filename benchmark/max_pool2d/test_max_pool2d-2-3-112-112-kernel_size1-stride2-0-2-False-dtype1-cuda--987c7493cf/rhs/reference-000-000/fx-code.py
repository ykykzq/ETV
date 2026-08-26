


def forward(self, arg0_1):
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg0_1, [3, 3], [2, 3], [0, 0], [2, 2]);  arg0_1 = None
    getitem = max_pool2d_with_indices[0]
    getitem_1 = max_pool2d_with_indices[1];  max_pool2d_with_indices = getitem_1 = None
    return getitem
    
