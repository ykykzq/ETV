


def forward(self, arg0_1):
    adaptive_max_pool2d = torch.ops.aten.adaptive_max_pool2d.default(arg0_1, [5, 7]);  arg0_1 = None
    getitem = adaptive_max_pool2d[0]
    getitem_1 = adaptive_max_pool2d[1];  adaptive_max_pool2d = getitem_1 = None
    return getitem
    
