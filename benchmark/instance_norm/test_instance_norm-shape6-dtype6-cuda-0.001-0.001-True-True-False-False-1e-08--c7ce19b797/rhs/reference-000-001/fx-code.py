


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    repeat = torch.ops.aten.repeat.default(arg4_1, [6]);  arg4_1 = None
    repeat_1 = torch.ops.aten.repeat.default(arg0_1, [6]);  arg0_1 = None
    repeat_2 = torch.ops.aten.repeat.default(arg2_1, [6])
    repeat_3 = torch.ops.aten.repeat.default(arg3_1, [6])
    view = torch.ops.aten.view.default(arg1_1, [1, 174, 3, 1]);  arg1_1 = None
    cudnn_batch_norm = torch.ops.aten.cudnn_batch_norm.default(view, repeat, repeat_1, repeat_2, repeat_3, True, 0.1, 1e-08);  view = repeat = repeat_1 = None
    getitem = cudnn_batch_norm[0]
    getitem_1 = cudnn_batch_norm[1];  getitem_1 = None
    getitem_2 = cudnn_batch_norm[2];  getitem_2 = None
    getitem_3 = cudnn_batch_norm[3];  cudnn_batch_norm = getitem_3 = None
    alias = torch.ops.aten.alias.default(arg2_1);  arg2_1 = None
    view_1 = torch.ops.aten.view.default(repeat_2, [6, 29]);  repeat_2 = None
    mean = torch.ops.aten.mean.dim(view_1, [0]);  view_1 = None
    copy_ = torch.ops.aten.copy_.default(alias, mean);  alias = mean = copy_ = None
    alias_1 = torch.ops.aten.alias.default(arg3_1);  arg3_1 = None
    view_2 = torch.ops.aten.view.default(repeat_3, [6, 29]);  repeat_3 = None
    mean_1 = torch.ops.aten.mean.dim(view_2, [0]);  view_2 = None
    copy__1 = torch.ops.aten.copy_.default(alias_1, mean_1);  alias_1 = mean_1 = copy__1 = None
    view_3 = torch.ops.aten.view.default(getitem, [6, 29, 3, 1]);  getitem = None
    return view_3
    
