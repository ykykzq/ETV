


def forward(self, arg0_1):
    max_pool3d_with_indices = torch.ops.aten.max_pool3d_with_indices.default(arg0_1, [3, 3, 3], [2, 2, 1], [0, 0, 0], [1, 1, 1], True);  arg0_1 = None
    getitem = max_pool3d_with_indices[0]
    getitem_1 = max_pool3d_with_indices[1];  max_pool3d_with_indices = getitem_1 = None
    return getitem
    
