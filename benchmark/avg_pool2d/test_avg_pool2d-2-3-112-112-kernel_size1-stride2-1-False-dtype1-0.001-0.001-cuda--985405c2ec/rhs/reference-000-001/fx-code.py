


def forward(self, arg0_1):
    avg_pool2d = torch.ops.aten.avg_pool2d.default(arg0_1, [3, 3], [2, 3], [1, 1]);  arg0_1 = None
    return avg_pool2d
    
