# 真实 Add 验证

本文档记录 `examples/add` 中程序对的来源、生成、PairSpec、验证证据和限制。
它的目的不是展示一个人为整理过的等式，而是让用户能够沿着上游源码、编译器
生成物、原始 TTIR 和最终报告逐层核查。

## 验证对象

ntops 上游的 Add 语义定义位于 `src/ntops/kernels/add.py`：

```python
def application(input, other, alpha, output):
    output = input + alpha * other
```

同一仓库的 `tests/test_add.py` 将 `ntops.torch.add` 与
`torch.add(input, other, alpha=alpha)` 比较。本次固定以下 specialization：

| 项目 | 值 |
| --- | --- |
| shape | `[8, 16]` |
| input/other/output dtype | `float32` |
| 逻辑元素数 | `128` |
| 布局 | 连续二维，size `(8,16)`、stride `(16,1)` |
| 编译目标 | CUDA sm80，warp size 32 |
| ntops block | 256 |
| TorchInductor XBLOCK | 128 |
| launch | 两侧各 1 个 program |

固定的上游版本为：

| 工程 | 提交/版本 |
| --- | --- |
| `InfiniTensor/ntops` | `9ae4166ad342e4745f0eed13a5a20d069e994fc0` |
| `InfiniTensor/ninetoothed` | `efe519d1b12a820e7aa605d775af3d52c8b0d605`，0.26.0 |
| PyTorch | 2.8.0 |
| Triton/libtriton | 3.7.1，源码标签对应提交 `f797708c0626e5f9840ca5b0a98790e2c7cb09ad` |

完整文件 SHA-256 在 `examples/add/provenance.json` 中，提取工具还会检查 ntops
的 `add.py` 和 `element_wise.py` 哈希，避免相同提交参数被错误目录或脏源码替代。

## 两侧如何生成

左侧执行以下真实编译链：

```text
ntops.kernels.add.premake(2, float32, block_size=128)
  -> ninetoothed 0.26.0 SSA lowering
  -> 生成 ntops_add.triton.py 和 SSA/布局 JSON
  -> Triton ASTSource，BLOCK=256，CUDA sm80
  -> Triton make_ttir
  -> ntops_add.ttir
```

生成的 TTIR 有 16 个运行时参数。`arg0/arg1/arg3` 是 input/other/output 指针，
`arg2` 是 `f64` alpha 标量，`arg4..arg15` 是三组二维 size/stride。复杂的二维
地址计算、边界条件、cast 和 256-lane mask 都来自生成器，没有人工简化。

右侧为了保留“alpha 由 0-D tensor 传入”的 ABI 差异，捕获的 PyTorch 表达式是
`input + alpha_tensor * other`，不是把 Python 常量 alpha 内联。流程为：

```text
三个 FakeTensor CUDA 输入
  -> make_fx(tracing_mode="fake")
  -> aten.mul.Tensor + aten.add.Tensor FX 图
  -> TorchInductor GraphLowering 和 Triton codegen
  -> triton_poi_fused_0 原始生成源码
  -> Triton ASTSource，XBLOCK=128，CUDA sm80
  -> Triton make_ttir
  -> torch_inductor_add.ttir
```

当前机器没有 CUDA，且 macOS 无法执行 NVIDIA ptxas，因此这里使用 PyTorch 的
FakeTensor 和离线 CUDA sm80 属性完成真实 Inductor 调度与代码生成，没有执行
GPU kernel。Inductor 生成源码在仅去除行尾空白后完整保存在
`sources/torch_inductor_add.generated.py`。它的 pointwise decorator 负责运行时
launch/autotune，会初始化 CUDA driver；离线 TTIR 编译文件只删除这个 decorator，
保留原样的 `@triton.jit` 函数文本。这个适配不改变 kernel AST，但它仍是生成链的
一个受信步骤，已在 provenance 中明确记录。

PyTorch 2.8 的 Triton 可用性检测还依赖已从 Triton 3.7.1 移除的 `triton_key`
接口。提取工具提供一个只返回固定版本的兼容函数，将 backend hash 固定为离线
sm80 标识，并向 Inductor 提供 A100/sm80 设备属性。这三项只影响“是否允许进入
Triton codegen”、缓存元数据和调度目标，不改写 FX 图或生成的 kernel 函数；它们
仍属于提取链可信边界，并在 provenance 的 `compatibility_shims` 中逐项列出。

TTIR 的 debug location 被机械替换为仓库相对路径。这只改变诊断位置，不改变
operation、类型、SSA 边或属性。

## PairSpec

`examples/add/pair.json` 将物理 ABI 对齐到四个逻辑角色：

| 角色 | ntops 左侧 | TorchInductor 右侧 |
| --- | --- | --- |
| `Input` | block `arg0` | block `arg0` |
| `Other` | block `arg1` | block `arg2` |
| `Alpha` | scalar `arg2` | `arg1[0]` 的 scalar-block load |
| `Output` | block `arg3` | block `arg3` |

spec 固定左侧三组 `(size0,size1,stride0,stride1)` 为 `(8,16,16,1)`，声明
`Input`、`Other`、`Output` 两两不相交，并要求 128 个 Output 元素完整覆盖且无
竞争。右侧 `arg4` 是 Inductor ABI 中的 xnumel，但 kernel 首句已经把 xnumel
specialize 为常量 128，优化后的 TTIR 不读取该参数。PairSpec 使用
`side_bindings` 分别声明 `lhs.arg4=8` 和 `rhs.arg4=128`，不会把同一参数位置的
两个 ABI 含义混为一个事实；报告也会分别列出这些可信绑定。

## 实际验证记录

执行：

```bash
.venv/bin/python -m etv check examples/add/pair.json \
  --out build/add_upstream
```

得到：

```text
PROVED ntops_add_vs_torch_inductor_add: OBSERVABLE_MEMORY_EQUIVALENT
```

证明报告中的主要义务为：

| 义务 | 结果 | 证据摘要 |
| --- | --- | --- |
| FRONTEND | `PROVED` | 两侧均由 libtriton 3.7.1 parse、verify、提升 |
| ABI | `PROVED` | 四个逻辑角色按上表显式对齐 |
| INDEX | `PROVED` | 左侧穷举 256 lane，右侧穷举 128 lane |
| MASK | `PROVED` | 两侧有效逻辑域均为 128 个元素 |
| COVERAGE | `PROVED` | Output `[0,128)` 完整覆盖 |
| RACE_FREEDOM | `PROVED` | 有效 output 地址单射 |
| ADDRESS | `PROVED` | 同一逻辑元素映射到相同 output offset |
| LOAD | `PROVED` | input、other 和 alpha 读取前沿一致 |
| DEFINEDNESS | `PROVED` | 本例没有未闭合的偏函数定义域 |
| COMPUTE | `PROVED` | 128 对计算根属于相同 e-class |
| STORE | `PROVED` | Output 最终内存相等，其他逻辑 block 不变 |

compute 阶段的 egglog 统计是 641 e-node、641 e-class、0 次迭代，停止原因是
`ROOTS_ALREADY_CONGRUENT`。也就是说，ABI 和读取角色归一化之后，两侧已经具有
相同的 `input + alpha * other` 计算结构，本例不需要任何代数规则匹配，更没有
使用 `trusted_fact` 重写。所有规则仍统一完成准入；只有尚未同余的根才触发饱和。

## 可复现步骤

在已经构建 Triton 3.7.1 的 Python 3.12 环境中：

```bash
git clone https://github.com/InfiniTensor/ntops.git /tmp/etv-ntops
git -C /tmp/etv-ntops checkout 9ae4166ad342e4745f0eed13a5a20d069e994fc0

git clone https://github.com/InfiniTensor/ninetoothed.git /tmp/etv-ninetoothed
git -C /tmp/etv-ninetoothed checkout efe519d1b12a820e7aa605d775af3d52c8b0d605

python -m pip install -r requirements-extraction.txt
python -m pip install -e /tmp/etv-ninetoothed
python tools/extract_add_pair.py \
  --ntops-repo /tmp/etv-ntops \
  --ninetoothed-repo /tmp/etv-ninetoothed \
  --output examples/add
python -m etv check examples/add/pair.json --out build/add_upstream
```

第二次运行后可以用 `git diff --exit-code -- examples/add` 检查生成物是否一致。

## 结论边界

本次 `PROVED` 是固定 shape、布局、launch 和 `ABSTRACT_FLOAT` 下的可观察内存
等价。它不证明：

- 两个 kernel 的 IEEE-754 或逐位结果相等；左侧 alpha 为 f64、右侧 load 为
  f32，cast 在当前抽象中被抹除，这尤其不能推广为位级结论；
- 任意 shape、stride、dtype、target 或 launch 下均等价；
- 实际 GPU 运行、性能、ptx/cubin 后端正确性或 buffer 边界安全；
- ninetoothed、TorchInductor、Triton 和 ETV 编译/提升实现本身没有 bug。

因此该样例解决的是“对真实生成的这一个 TTIR 程序对完成有界翻译验证”，而不是
证明整个 ntops、PyTorch 或 Triton 编译器在所有输入上正确。
