


def forward(self, arg0_1):
    median = torch.ops.aten.median.dim(arg0_1, 1);  arg0_1 = None
    getitem = median[0]
    getitem_1 = median[1];  median = None
    return (getitem, getitem_1)
    
