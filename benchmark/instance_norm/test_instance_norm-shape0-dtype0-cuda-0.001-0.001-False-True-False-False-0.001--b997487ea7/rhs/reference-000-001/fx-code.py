


def forward(self, arg0_1, arg1_1, arg2_1):
    repeat = torch.ops.aten.repeat.default(arg2_1, [1]);  arg2_1 = None
    repeat_1 = torch.ops.aten.repeat.default(arg0_1, [1]);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [1, 1, 382]);  arg1_1 = None
    cudnn_batch_norm = torch.ops.aten.cudnn_batch_norm.default(view, repeat, repeat_1, None, None, True, 0.1, 0.001);  view = repeat = repeat_1 = None
    getitem = cudnn_batch_norm[0]
    getitem_1 = cudnn_batch_norm[1];  getitem_1 = None
    getitem_2 = cudnn_batch_norm[2];  getitem_2 = None
    getitem_3 = cudnn_batch_norm[3];  cudnn_batch_norm = getitem_3 = None
    view_1 = torch.ops.aten.view.default(getitem, [1, 1, 382]);  getitem = None
    return view_1
    
