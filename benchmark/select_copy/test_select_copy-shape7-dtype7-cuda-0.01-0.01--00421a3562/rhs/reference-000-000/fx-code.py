


def forward(self, arg0_1):
    select_copy = torch.ops.aten.select_copy.int(arg0_1, 1, 0);  arg0_1 = None
    return select_copy
    
