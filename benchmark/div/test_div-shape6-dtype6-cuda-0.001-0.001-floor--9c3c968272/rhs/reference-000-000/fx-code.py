


def forward(self, arg0_1, arg1_1):
    div = torch.ops.aten.div.Tensor_mode(arg0_1, arg1_1, rounding_mode = 'floor');  arg0_1 = arg1_1 = None
    return div
    
