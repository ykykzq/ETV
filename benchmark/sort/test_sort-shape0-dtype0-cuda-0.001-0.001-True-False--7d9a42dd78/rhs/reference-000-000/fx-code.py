


def forward(self, arg0_1):
    sort = torch.ops.aten.sort.stable(arg0_1, stable = False, descending = True);  arg0_1 = None
    getitem = sort[0]
    getitem_1 = sort[1];  sort = None
    return (getitem, getitem_1)
    
