


def forward(self, arg0_1, arg1_1):
    expand = torch.ops.aten.expand.default(arg0_1, [2, 394, 724]);  arg0_1 = None
    view = torch.ops.aten.view.default(expand, [2, 394, 724]);  expand = None
    expand_1 = torch.ops.aten.expand.default(arg1_1, [2, 724, 388]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(expand_1, [2, 724, 388]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(bmm, [2, 394, 388]);  bmm = None
    return _unsafe_view
    
