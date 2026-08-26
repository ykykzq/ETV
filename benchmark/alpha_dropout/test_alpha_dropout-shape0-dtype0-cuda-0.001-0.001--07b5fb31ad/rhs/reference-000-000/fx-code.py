


def forward(self, arg0_1):
    empty_like = torch.ops.aten.empty_like.default(arg0_1)
    bernoulli_ = torch.ops.aten.bernoulli_.float(empty_like, 0.5876426048708127);  empty_like = None
    add = torch.ops.aten.add.Scalar(bernoulli_, -1)
    mul_ = torch.ops.aten.mul_.Scalar(add, 1.5206805237999195);  add = None
    add_ = torch.ops.aten.add_.Scalar(mul_, 0.627063859617823);  mul_ = None
    mul__1 = torch.ops.aten.mul_.Scalar(bernoulli_, 0.8649571093445578);  bernoulli_ = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, mul__1);  arg0_1 = mul__1 = None
    add__1 = torch.ops.aten.add_.Tensor(mul, add_);  mul = add_ = None
    return add__1
    
