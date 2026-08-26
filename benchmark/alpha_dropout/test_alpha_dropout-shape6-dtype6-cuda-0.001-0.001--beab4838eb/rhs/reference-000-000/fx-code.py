


def forward(self, arg0_1):
    empty_like = torch.ops.aten.empty_like.default(arg0_1)
    bernoulli_ = torch.ops.aten.bernoulli_.float(empty_like, 0.7632635256421605);  empty_like = None
    add = torch.ops.aten.add.Scalar(bernoulli_, -1)
    mul_ = torch.ops.aten.mul_.Scalar(add, 1.5292054110318856);  add = None
    add_ = torch.ops.aten.add_.Scalar(mul_, 0.3620186975766194);  mul_ = None
    mul__1 = torch.ops.aten.mul_.Scalar(bernoulli_, 0.8698060317199324);  bernoulli_ = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, mul__1);  arg0_1 = mul__1 = None
    add__1 = torch.ops.aten.add_.Tensor(mul, add_);  mul = add_ = None
    return add__1
    
