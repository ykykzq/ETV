


def forward(self, arg0_1):
    max_1 = torch.ops.aten.max.dim(arg0_1, -1, True);  arg0_1 = None
    getitem = max_1[0]
    getitem_1 = max_1[1];  max_1 = None
    return (getitem, getitem_1)
    
