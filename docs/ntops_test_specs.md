# ntops 75 个独立 Kernel 的测试派生 Spec

## 范围

本目录只依据 `InfiniTensor/ntops` commit
`9ae4166ad342e4745f0eed13a5a20d069e994fc0` 的 `tests/` 归纳，不引用 ETV 的实现能力，
也不把 PyTorch 文档中未被测试覆盖的行为补进来。
上游测试源码采用 Apache-2.0 许可证；机器证据同时记录了仓库、commit 和许可证文件，
许可证副本位于 `specs/ntops_LICENSE`。

该 checkout 有 76 个 `test_*.py`，但只有 75 个独立 kernel。`matmul` 没有对应的 kernel，
只是根据 rank 调用 `mm` 或 `bmm` 的 API 包装器，因此本目录按 75 个 kernel 统计。
75 个模块共有 108 个测试函数。逐函数的参数化表达式、输入构造、候选调用、Torch 调用和
断言完整保存在 `specs/ntops_test_evidence.json`。

下面的“Spec”严格表示测试所观察的合约。除非另有说明，输入和输出均在 CUDA 上；测试
没有覆盖的非法输入、异常、广播方式、特殊浮点值或副作用，不应从表中自行推出。

在对应 ntops checkout 存在时，可重复生成证据：

```bash
.venv/bin/python tools/extract_ntops_test_specs.py \
  --ntops-repo build/upstream/ntops \
  --output specs/ntops_test_evidence.json
.venv/bin/python tools/materialize_ntops_operator_specs.py
```

抽取器读取 Python AST 和原始 source segment，不导入 ntops、不执行测试，也不消耗随机数；
因此记录的是生成规则和断言，而不是某次随机运行的取值。
对应的 75 份独立、仅描述输入前提的 YAML 位于 `specs/ntops/operators/`，入口为
`specs/ntops/index.yaml`。本页继续保留测试后置条件作为调研记录，但这些后置条件不写入
算子 YAML。

## 公共记号

`F` 表示共享浮点生成器：rank 分别取 1、2、3、4；每个 rank 各生成一个 float32 和
float16 shape。shape 的所有维度为正，生成器先取 `[256, 1024]` 的元素预算，再经逐次
整数除法构造维度；每个维度均落在 `[1,1024]`，但预算不等同于最终 `numel`。普通输入
由 `torch.randn` 生成，测试没有为这些样本声明更窄的数学取值范围。
float32 的 `rtol=atol=1e-3`，float16 的 `rtol=atol=1e-2`。

`I` 使用相同 shape 生成器，dtype 为 bool、int8、int16、int32。bool 由均匀随机数与
0.5 比较得到，整数逐元素取 `[-10, 10)`。`M` 表示 `m,n,k` 各自独立取 `[1,1024]`，
dtype、device 和容差与 `F` 相同。`N(S,T)` 表示 shape 为 `S`、dtype 为 `T` 的
`randn` 张量。

“近似等于”指测试调用 `torch.allclose`；“精确等于”指 `torch.equal` 或指针/对象恒等
断言。表中写“同 Torch”时，后置条件就是与该测试实际调用的 Torch reference 在相同
参数下满足对应比较关系，并不表示已经给出或证明了该算子的独立数学定义。
除广播、归约、排序的 indices 和显式形状变换外，逐元素行的输出 shape 与输入 shape
相同；输出 dtype 以对应 Torch reference 为准。

## 逐元素与激活算子

| 算子 | 测试输入域和参数 | 测试后置条件 |
| --- | --- | --- |
| `abs` | 一个 `F` 张量。 | 输出与 `torch.abs` 近似等于。 |
| `acosh` | 基本域为 `F` shape、float16/float32、逐元素 `[1,10]`；另测 float16/32/64 的 `[1,1.000001,100,0.5]` 与 `[NaN,+Inf,-Inf,1,2,0.5]`，以及 `(10000,)`、`(100,100)`、`(10,10,10)`。`out` 覆盖连续、切片、转置和多轴步进布局。 | 有效值与 `torch.acosh` 近似等于；`acosh(1)=0`；NaN、`-Inf` 和有限 `x<1` 输出 NaN，`+Inf` 输出 `+Inf`；shape/dtype 保持。提供 `out` 时返回对象必须就是 `out`。float16 通常用 `1e-3/1e-3`，其余边界测试用 `1e-5/1e-6`。 |
| `atan` | `F` shape 上取均匀 `[-10,10)`；另测 float16/32/64 的 `[0,1,-1,+Inf,-Inf]`、float16/32 的 `[NaN,1]`、`(4,5,6)` 非连续 `out`，以及经原地仿射变换的 `(10,10)`。 | 与 `torch.atan` 近似等于并保持 shape/dtype；`atan(0)=0`，两端无穷趋于 `+/-pi/2`，NaN 传播；`out` 返回对象恒等。 |
| `celu` | 一个 `F` 张量；`alpha=1.0`，`inplace` 为 true/false。 | 与 `torch.nn.functional.celu` 近似等于。测试没有断言原地模式的 storage identity。 |
| `clamp` | 三个同 shape、同 dtype 的 `F`/`randn` 张量 `input,min,max`；没有约束 `min<=max`，没有广播 case。 | 与 `torch.clamp(input,min,max)` 近似等于。 |
| `cos` | `F` 中只有 float32 分支实际执行；float16 分支直接返回。 | 与 `torch.cos` 近似等于。 |
| `cosh` | 一个 `F` 张量。 | 与 `torch.cosh` 近似等于。 |
| `exp` | `F` 中只有 float32 分支实际执行；float16 分支直接返回。 | 与 `torch.exp` 近似等于。 |
| `gelu` | 一个 `F` 张量；实际仅 `approximate="none"`，`"tanh"` 被 pytest 标记跳过。 | 与 `torch.nn.functional.gelu` 近似等于。 |
| `isinf` | 一个 `F` shape 张量；先取 `randn`，再按同 shape 均匀随机掩码把约 40% 元素改为 `+Inf`、约 40% 改为 `-Inf`，其余保持有限采样值。 | 布尔输出与 `torch.isinf` 精确等于。没有 NaN case。 |
| `isnan` | 一个 `F` shape 张量；先取 `randn`，再按同 shape 均匀随机掩码把约 40% 元素改为 NaN。 | 布尔输出与 `torch.isnan` 精确等于。没有 Inf case。 |
| `neg` | 一个 `F` 张量。 | 与 `torch.neg` 近似等于。 |
| `pow` | 两个同 shape 的 `F`/`randn` 张量；只有 float32 实际执行，float16 直接返回。底数可为负且指数为任意采样浮点。 | 与 `torch.pow` 近似等于，`equal_nan=True`。 |
| `relu` | 一个 `F` 张量；`inplace` 为 true/false。 | 与 `torch.nn.functional.relu` 近似等于；未断言原地 storage identity。 |
| `round` | 一个 `F` 张量。 | 与 `torch.round` 近似等于。 |
| `rsqrt` | 一个 `F` 张量，允许采样到负值。 | 与 `torch.rsqrt` 近似等于，`equal_nan=True`。 |
| `sgn` | `F` 的实数输入；当基础 dtype 为 float32 且 `is_complex=True` 时，还构造同 shape 的 complex64 输入。float16 的 `is_complex=True` 分支仍使用实数。 | 与 `torch.sgn` 近似等于。 |
| `sigmoid` | `F` 中只有 float32 实际执行；float16 直接返回。 | 与 `torch.sigmoid` 近似等于。 |
| `sign` | 一个 `F` 张量。 | 与 `torch.sign` 近似等于。 |
| `signbit` | 一个 `F` 张量。 | 布尔输出与 `torch.signbit` 通过 `allclose` 比较。 |
| `silu` | 一个 `F` 张量；没有 inplace case。 | 与 `torch.nn.functional.silu` 近似等于。 |
| `sin` | `F` 中只有 float32 实际执行；float16 直接返回。 | 与 `torch.sin` 近似等于。 |
| `tanh` | `F` 中只有 float32 实际执行；float16 直接返回。 | 与 `torch.tanh` 近似等于。 |
| `threshold` | 一个 `F` 张量；`threshold` 均匀取 `[-1,1]`，替换值均匀取 `[0,1]`。 | 与 `torch.nn.functional.threshold` 近似等于。 |

## 二元、比较与位运算

| 算子 | 测试输入域和参数 | 测试后置条件 |
| --- | --- | --- |
| `add` | 两个同 shape 的 `F` 张量；`alpha` 来自标准高斯标量。 | 与 `torch.add(input,other,alpha)` 近似等于。 |
| `sub` | 两个同 shape 的 `F` 张量；`alpha` 来自标准高斯标量。 | 与 `torch.sub(input,other,alpha)` 近似等于。 |
| `mul` | 两个同 shape 的 `F` 张量。 | 与 `torch.mul` 近似等于。 |
| `div` | 两个同 shape 的 `F` 张量；`rounding_mode` 实际为 `None` 或 `"floor"`，`"trunc"` 被跳过。除数未显式排除 0。 | 与 `torch.div` 近似等于。没有 `equal_nan`。 |
| `eq` | 两个同 shape 的 `F` 张量。 | 与 `torch.eq` 精确等于。 |
| `ne` | 两个同 shape 的 `F` 张量。 | 与 `torch.ne` 精确等于。 |
| `lt` | 两个同 shape 的 `F` 张量。 | 与 `torch.lt` 精确等于。 |
| `le` | 两个同 shape 的 `F` 张量。 | 与 `torch.le` 精确等于。 |
| `gt` | 两个同 shape 的 `F` 张量。 | 与 `torch.gt` 精确等于。 |
| `ge` | 两个同 shape 的 `F` 张量。 | 与 `torch.ge` 精确等于。 |
| `bitwise_and` | 两个同 shape 的 `I` 张量。 | 与 `torch.bitwise_and` 精确等于。 |
| `bitwise_or` | 两个同 shape 的 `I` 张量。 | 与 `torch.bitwise_or` 精确等于。 |
| `bitwise_not` | 一个 `I` 张量。 | 与 `torch.bitwise_not` 精确等于。 |
| `fmax` | 基本 case 是两个同 shape 的 `F` 张量；广播 case 为 `(4,1,32)` 与 `(1,64,32)`，dtype float16/32。输入均来自 `randn`，没有 NaN/Inf。 | 实际 reference 是 `torch.maximum`，不是 `torch.fmax`；输出分别近似等于 reference，广播 shape 必须为 `(4,64,32)`。因此测试不能确定 NaN 下的 fmax 语义。 |
| `maximum` | 除同 shape `F` 外，覆盖 `(4,1,32)` 与 `(1,64,32)` 广播，以及 float32 `(100,)` 的 `out=`。 | 与 `torch.maximum` 近似等于；广播输出为 `(4,64,32)`；`out` 内容被写成 reference，但未断言返回对象。 |

## 线性代数与归一化

| 算子 | 测试输入域和参数 | 测试后置条件 |
| --- | --- | --- |
| `mm` | `M`：`input:(m,k)`、`other:(k,n)`。 | 输出 `(m,n)` 与 `torch.mm` 近似等于。 |
| `bmm` | `M` 外加 batch `b in [4,16]`：`input:(b,m,k)`、`other:(b,k,n)`。 | 输出 `(b,m,n)` 与 `torch.bmm` 近似等于。 |
| `addmm` | `M`：`input:(m,n)`、`x:(m,k)`、`y:(k,n)`；`alpha,beta` 为标准高斯标量。 | 与 `torch.addmm(input,x,y,beta,alpha)` 近似等于。 |
| `addmv` | 复用 `M` 生成器但忽略 `n`：`input:(m,)`、`mat:(m,k)`、`vec:(k,)`；`alpha,beta` 为标准高斯标量。 | 输出 `(m,)` 与 `torch.addmv` 近似等于。 |
| `batch_norm` | `F` 仅 rank 2--4；channel 为 `shape[1]`。`affine` 决定是否提供同 dtype 的 `(C,)` weight/bias；`eps` 为 `1e-5` 或 `1e-3`；`training=True`，running mean/var 均为 None。 | 与 `torch.nn.functional.batch_norm` 近似等于。没有 eval/running-stat 更新后置条件。 |
| `instance_norm` | `F` shape 被前置 1 扩到至少 rank 3；weight/bias 独立可为 None 或 `(C,)`。`eps in {1e-8,1e-5,1e-3}`，`use_input_stats` 与 `track_running_stats` 各为 bool。需要 running stats 时，两者为 `(C,)`，variance 由 `abs(randn)` 得到。 | 输出与 Torch reference 近似等于；仅当 `use_input_stats and track_running_stats` 时，更新后的 running mean/var 也分别近似等于 reference。 |
| `layer_norm` | 一个 `F` 张量；`normalized_shape` 是输入 shape 的任意非空后缀。weight/bias 独立可为 None，否则 shape 等于该后缀；`eps in {1e-8,1e-5,1e-3}`。 | 与 `torch.nn.functional.layer_norm` 近似等于。 |
| `rms_norm` | 一个 `F` 张量；`normalized_shape` 是任意非空 shape 后缀；weight 为 None 或同后缀 shape；`eps in {None,0,1e-5,1e-3}`。 | 与 `torch.nn.functional.rms_norm` 近似等于。测试没有为 `eps=0` 补充非零 RMS 的逻辑前提。 |
| `rotary_position_embedding` | `input:(B,L,H,D)`，`B in {1,4}`、`L in {1,128}`、`H in {1,8}`、`D in {32,64}`，float16/32；`D` 为偶数。sin/cos table 均为 `(L,D/2)`，按 `base=10000` 生成；`interleaved`、`inplace` 各为 bool。 | 对每个二元分组或前后半分组应用 `(x0*cos-x1*sin, x0*sin+x1*cos)`，结果与测试内的 Torch 参考函数近似等于。float32 为 `rtol=0,atol=1e-3`，float16 为 `1e-3/1e-3`；未断言 inplace storage identity。 |
| `scaled_dot_product_attention` | `q:(B,Hq,L,D)`、`k/v:(B,Hkv,S,D)`；`B in [1,4]`，`Hq in {2,4,8,16,32}`，`Hkv` 是不大于 `Hq` 的 2 的幂，`L,S in [1,512]`，`D in {32,64}`，float16/32，`enable_gqa=True`。mask 为 None、`(L,S)` bool，或仅含 `0/-Inf` 的浮点；mask 与 `is_causal=True` 不同时出现。scale 为 None 或 `[0.05,0.5]`；causal variant 为 None/lower-right/upper-left，lower-right 要求 `L<=S`。KV-cache case 把 k/v 最后一个 slot 清零并另传原值与 slot view。 | 输出与 `torch.nn.functional.scaled_dot_product_attention` 近似等于；lower-right causal 用对应 bias 转成显式 mask。float32 容差 `1e-2`，float16 `2.5e-2`。测试没有直接断言 present slot 的最终内容，只通过输出间接观察 cache 行为。 |

## 归约、排序与形状变换

| 算子 | 测试输入域和参数 | 测试后置条件 |
| --- | --- | --- |
| `argsort` | rank 1--3、float16/32、共享随机 shape；被排序轴随机取合法非负 dim，且该轴长度若原本大于 128 会改成 `[10,128]`；`descending` 为 bool。 | 用返回索引 gather 后的值与 Torch 索引 gather 后的值近似等于，并检查单调性。没有要求索引张量与 Torch 精确一致，所以重复值的 tie order 未指定。 |
| `diag` | 1-D：`n in {1,5,10}` 及列出的正负 diagonal，另有长度 0 与 `k in {0,3,-3}`；2-D：`(5,5)`、`(3,5)`、`(5,3)`、`(10,10)` 与合法/越界正负 diagonal；float16/32。 | 输出内容与 `torch.diag` 通过默认 `allclose`，空输入和越界 diagonal 还要求 shape 相同。 |
| `logsumexp` | 基本输入为 `F`，dim 随机取任一正轴或对应负轴，`keepdim` 为 bool。另用固定 3-D/4-D shape 覆盖 dim 0/1/2、连续 `out`、切片、转置、permute 和多轴步进 `out`。 | 与 `torch.logsumexp` 近似等于；提供 `out` 时返回对象必须就是该 view，连续 case 还检查 contiguous；另检查有/无 `out` 的结果一致。 |
| `max` | 一个 `F` 张量。覆盖全局归约，以及任一正轴或对应负轴的归约；轴归约的 `keepdim` 为 bool。 | 全局值与 `torch.max` 近似等于；轴归约的 values 近似相等且 indices 精确相等。 |
| `mean` | 一个 `F` 张量，覆盖全局与任一正/对应负轴，轴归约的 `keepdim` 为 bool；另测 int32 `(1024,1024)`、值域 `[0,10)` 的全局 mean。 | 浮点 case 与 `torch.mean` 近似等于；整数 case 要求浮点输出并与 `torch.mean(input.float())` 近似等于。 |
| `median` | rank 1--3、float16/32；dim 为合法非负轴，被归约轴长度最多 128；`keepdim=False`。 | values 与 Torch 近似等于；返回 indices gather 出的输入值必须等于 reference values。没有要求 indices 与 Torch indices 相同，因此 tie index 未指定。 |
| `msort` | 一个 `F` 张量。 | 与 `torch.msort` 近似等于，即测试观察 dim 0 的升序 values。 |
| `quantile` | `F` 中只有 float32 实际执行；q 为 `[0,1)` 的 Python scalar，或长度 `[1,5]`、逐元素 `[0,1)` 的 1-D tensor；dim 为 None 或合法非负轴；`keepdim` 为 bool；interpolation 遍历 linear/lower/higher/nearest/midpoint。 | 与 `torch.quantile` 近似等于。float16 被显式跳过。 |
| `select_copy` | 一个 `F` 张量；dim 为合法非负轴，index 为该轴 `[0,size-1]`。 | rank 减 1 的输出与 `torch.select_copy` 近似等于。没有负 dim/负 index case。 |
| `softmax` | 一个 `F` 张量；dim 为任一合法非负轴；输出 dtype 独立随机选 float16/32/64，可能与输入 dtype 不同。 | 与 `torch.nn.functional.softmax(input,dim,dtype)` 近似等于。容差仍取自输入 dtype，而不是输出 dtype。 |
| `sort` | 基本域为 `F`，dim 取任一合法正/负轴，`descending`、`stable` 为 bool；重复值 case 为 `(16,33)`、整数 `[-4,5)` 转 float32、dim -1、stable=true；`out` case 为 float16 `(19,23)` 与 values/indices 缓冲。 | values 近似等于且 indices 精确等于 Torch；重复值 stable case 两者均精确；`out` 返回的 values/indices 必须与给定缓冲共享相同 data pointer 并被精确写入。 |
| `stack` | 2--5 个同 shape float32 张量；每个输入固定 rank 3，各维独立取 `[16,64]`；dim 仅为 0、1、2。 | 输出在 dim 插入长度为张量数的新轴，与 `torch.stack` 在 `1e-6/1e-6` 下近似等于，并且 contiguous。未测 dim 3 和负 dim。 |
| `rot90` | `F` shape；rank 1 时追加一个长度 2 的轴，使 rank 至少 2。dims 是两个不同的合法非负轴；`k = r + 4q`，`r in {0,1,2,3}`、`q in [-100,100]`。 | 与 `torch.rot90` 近似等于，因而观察 `k mod 4` 的全部四种旋转。 |

## 池化、卷积与随机算子

| 算子 | 测试输入域和参数 | 测试后置条件 |
| --- | --- | --- |
| `adaptive_avg_pool2d` | `input:(B,C,H,W)`，float16/32；`B in [1,3]`、`C in [1,4]`；output size 为 `(1,1)/(2,2)/(4,4)`，且 `H=out_h*q_h`、`W=out_w*q_w`，`q_h,q_w in [2,6]`。 | 与 Torch adaptive average pool 近似等于；float32 容差 `1e-3`，float16 `5e-3`。只覆盖可整除尺寸。 |
| `adaptive_max_pool2d` | 同为 4-D float32；`B in [1,3]`、`C in [1,4]`；output size 为 `(1,1)/(2,2)/(3,3)/(5,7)`，输入高宽分别是输出高宽的 `[1,5]` 倍。 | 与 Torch adaptive max pool 在 `1e-3/1e-3` 下近似等于；只覆盖可整除尺寸。 |
| `avg_pool2d` | `input:(2,3,112,112)`，float16/32；kernel `(1,1)` 或 `(3,3)`；stride None、1 或 `(2,3)`；padding 0、1 或 `(2,3)`，但任一 padding 大于对应 kernel/2 的组合跳过；`ceil_mode=False`。 | 与 `torch.nn.functional.avg_pool2d` 近似等于；float32 容差 `1e-5`，float16 `1e-3`。 |
| `conv2d` | `input:(2,3,112,112)`、weight `(4,3,r,s)`、bias `(4,)`，float16/32；`(r,s)` 为 `(1,1)` 或 `(3,3)`；stride 为 1、2、`(2,3)`；padding 为 0、1、`(2,3)`；dilation 为 1、2、`(2,3)`；默认 groups=1。 | 与 `torch.nn.functional.conv2d` 近似等于；float32 容差 `1e-5`，float16 `1e-3`。没有无 bias、grouped 或 depthwise case。 |
| `max_pool1d` | float32 `input:(B,C,L)`，`B,C in [1,4]`、`L in [10,50]`；(kernel,stride) 为 `(2,2)/(3,2)/(3,1)`；padding 0/1/2；ceil_mode bool。 | reference 成功的组合与 Torch 在 `1e-3/1e-3` 下近似等于。若 Torch 因非法组合抛 `RuntimeError`，测试直接返回而没有数值后置条件；候选若先抛异常则失败。 |
| `max_pool2d` | `input:(2,3,112,112)`，float16/32；kernel `(1,1)/(3,3)`；stride None、1、`(2,3)`；padding 0、1、`(2,3)`，过大 padding 组合跳过；dilation 1、2、`(2,3)`；`ceil_mode=False`。 | 与 Torch max pool 通过默认 `allclose`。 |
| `max_pool3d` | float32 `input:(B,C,D,H,W)`，`B in [1,2]`、`C in [1,3]`、空间维各 `[8,16]`；kernel 为 2/3 或三元组各 2/3；stride 为 1/2 或三元组各 1/2；padding 0/1；ceil_mode bool。 | 与 Torch max pool 在 `1e-3/1e-3` 下近似等于。 |
| `lp_pool1d` | float32 `input:(B,C,L)`，`B,C in [1,4]`、`L in [4,32]`；norm 为 1/2/3；kernel `[1,min(5,L)]`；stride 为 None 或 `[1,kernel]`；ceil_mode bool。 | 与 Torch Lp pool 在 `1e-3/1e-3` 下近似等于，`equal_nan=True`。 |
| `lp_pool2d` | float32 `input:(B,C,H,W)`，`B in [1,3]`、`C in [1,4]`、`H,W in [4,24]`；norm 1/2/3；kernel 为标量或二元组、每轴最多 5；stride 为 None 或不大于对应 kernel；ceil_mode bool。 | 与 Torch Lp pool 在 `1e-3/1e-3` 下近似等于，`equal_nan=True`。 |
| `lp_pool3d` | float32 `input:(B,C,D,H,W)`，`B in [1,2]`、`C in [1,3]`、空间维各 `[4,16]`；norm 1/2/3；kernel 为标量或三元组、每轴最多 4；stride 为 None 或不大于对应 kernel；ceil_mode bool。 | 与 Torch Lp pool 在 `1e-3/1e-3` 下近似等于，`equal_nan=True`。 |
| `dropout` | 一个 `F` 张量；`p` 由 `random.uniform(0,1)` 生成；调用默认 training 行为，没有 eval/inplace case。 | 只要求 shape 等于 Torch reference、两次独立随机执行的非零比例差小于 0.1，并要求候选的非零位置满足 `output=input/(1-p)`。不要求逐元素 mask 与 Torch 相同。 |
| `alpha_dropout` | 一个 `F` 张量；`p in [0.1,0.5]`；覆盖 training true/false。 | training=true 时 shape 与 reference 相同，以接近 saturation 值的元素定义 drop mask，要求其比例与独立 Torch 随机执行相差小于 0.1，并要求其余元素满足 SELU dropout 的仿射式；training=false 时输出与输入精确相等。 |
| `bincount` | 1-D int32 input；case 为 `(size,weights,minlength)`：`(100,F,0)`、`(100,T,0)`、`(100,T,50)`、`(1000,F,2000)`、`(50,T,10)`、`(0,F,5)`。非空 input 值域 `[0,size)`；weights 若存在则为同长度 float32 `randn`。 | 无 weights 的整数结果与 `torch.bincount` 精确等于；有 weights 的浮点结果在 `atol=1e-4` 下近似等于。 |

## 关键缺口

这 75 份是“测试派生 Spec”，不是完整 PyTorch API Spec，尤其存在以下边界：

- 随机生成器没有固定 seed，同一 commit 的具体 shape 和标量会随收集过程改变；本目录记录
  的是生成规则而不是一次运行的样本值。
- 大多数算子只覆盖连续输入、同 dtype、CUDA 和正维度，没有系统测试空张量、alias、
  overlapping view、异常、广播、NaN/Inf、signed zero、整数溢出或越界。
- `cos/exp/pow/sigmoid/sin/tanh` 的 float16 case 没有执行断言；`gelu(tanh)`、
  `div(trunc)` 和 `quantile(float16)` 被跳过。
- `fmax` 的测试 oracle 是 `torch.maximum`，二者在 NaN 语义上可能不同；现有样本不能决定
  该名称应采用哪种完整语义。
- 排序与中位数对重复值的索引语义覆盖不一致；随机算子的断言是统计/关系性质，不是与
  reference 使用同一随机 mask 的逐元素等价。
- `out=`、inplace、running stats 和 KV cache 只有少数接口带直接 frame/alias 断言；未断言
  的副作用不能从数值输出相等推导。

因此，将这些 Spec 用于程序验证前，仍需由项目选择并声明：是只验证这里的测试域，还是
将其升级为全形状/全值域合约。后一种情况必须补充可满足的前置条件、IEEE-754/整数语义、
frame 条件和量化范围，并独立验证新增条款。

## `matmul` 包装入口

未计入 75 的 `test_matmul.py` 复用 `M`，batch 参数为 None、1、2 或 3：None 时输入为
`(m,k)` 与 `(k,n)`，否则为 `(b,m,k)` 与 `(b,k,n)`，输出与 `torch.matmul` 近似等于。
对应实现仅接受 2-D/3-D，并分别转发到 `mm`/`bmm`，所以它是组合 API Spec，不是第 76 个
独立 kernel Spec。
