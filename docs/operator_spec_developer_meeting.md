# 讨论稿

## 1.希望达成的结论

希望确认三：

1. 每个算子的**预期支持域**是什么，与**测试覆盖域**之间的gap；
2. 哪些输入属于合法调用，哪些输入应被拒绝，哪些输入暂不承诺；

具体来说，区分：

- **测试域**：现有 pytest 实际生成并执行的输入；
- **实现域**：当前 kernel 实际能够正确处理的输入；
- **目标域**：希望对外承诺兼容的输入，理论上是实现域的子集；


## 2. 从形式化验证角度理解 Spec

设 ntops kernel 为 `K`，对应的 Torch reference 为 `R`，输入前置条件为 `P(x)`。等价性验证
最终希望证明：

```text
forall x. P(x) => observable(K(x)) == observable(R(x))
```

这里至少有三个独立问题：

1. `P(x)` 是否准确描述了要验证的输入域；
2. `K` 和 `R` 的语义模型是否正确；
3. 等价性证明过程是否可靠。

当前解决第一个问题的一部分：记录候选前置条件。它不包含输出值
语义、浮点误差、异常、副作用、alias 或等价性证明结果。因此，YAML 中出现某个条件只表示
“验证时允许假设这个条件”，不表示算子已经通过形式化验证。

前置条件过强会产生“只证明了很小特例”的结果；前置条件过弱则可能使真实正确的 kernel
无法通过验证，或者掩盖未建模的未定义行为。Spec 的边界必须由算子开发人员和验证人员共同
确认，不能只从测试样例猜测。

## 3. 当前 Spec 的提取过程

当前流程固定到 ntops commit
`9ae4166ad342e4745f0eed13a5a20d069e994fc0`，分为证据抽取和规格物化两步：

```text
ntops tests + wrapper signatures
              |
              v
Python AST 静态抽取
              |
              v
ntops_test_evidence.json       原始、可审计证据
              |
              +------> 人工阅读测试并整理输入域摘要
              |                docs/ntops_test_specs.md
              v
谓词归一化 + 人工规则
              |
              v
75 份 standalone YAML          specs/ntops/operators/*.yaml
```

### 3.1 AST 自动抽取

[`extract_ntops_test_specs.py`](../tools/extract_ntops_test_specs.py) 不导入 ntops，也不运行测试，
而是解析 Python AST，保存：

- `pytest.mark.parametrize` 及其他 decorator；
- 测试函数参数；
- `torch.randn`、`torch.randint` 等张量构造；
- `ntops.torch.*` 候选调用；
- Torch reference 调用；
- `assert` 表达式；
- 测试 helper、常量和函数体；
- 上游仓库、commit 和许可证。

这样做的优点是可重复、不会消耗随机数、不会因为当前机器没有 CUDA 而丢失测试结构。输出
是 [`ntops_test_evidence.json`](../specs/ntops_test_evidence.json)，其定位是“证据”，不是全称
逻辑契约。

### 3.2 人工归纳

测试中的条件经常写在普通 Python 控制流中，例如：

```python
if padding_h > kernel_h / 2:
    pytest.skip("Invalid padding")
```

也可能由随机数、helper、条件分支或多个测试函数共同定义。当前由人工阅读这些测试，将输入
域整理到 [`ntops_test_specs.md`](ntops_test_specs.md)，并对容易表达的条件选用标准谓词。

### 3.3 YAML 物化

[`materialize_ntops_operator_specs.py`](../tools/materialize_ntops_operator_specs.py) 完成以下工作：

- 从 ntops Python wrapper 提取参数名、必选/可选状态和默认值；
- 展开公共的 float、integer 和 matmul 测试域，使每份 YAML 可以独立阅读；
- 解析字面量 `parametrize`，排除带 `pytest.skip` 标记的 case；
- 写入已人工整理的算子专用谓词；
- 生成索引和 JSON Schema；
- 检查 75 个 kernel 与 75 份 YAML 一一对应。

### 3.4 自动事实与人工判断

| 内容 | 当前来源 | 可靠程度 |
| --- | --- | --- |
| 参数名、默认值 | ntops wrapper AST | 直接证据 |
| 字面量参数集合 | pytest decorator AST | 直接证据 |
| 张量构造和断言文本 | test AST | 直接证据 |
| 被 skip 的字面量 case | pytest decorator/控制流 | decorator 可自动处理；普通控制流可能需人工处理 |
| rank、shape、dtype、参数关系 | 测试与 helper 的人工归纳 | 已复核，但不是自动证明 |
| 未结构化的复杂组合 | `observed_domain_zh` | 人类可读，机器暂不能直接推理 |
| 完整 PyTorch 合法域 | 当前未提取 | 不应从现有文件推断 |

## 4. 当前 YAML 格式

每份文件只有四个顶层字段：

```yaml
schema: operator-requires/v1
operator: ...
requires:
  observed_domain_zh: ...
  arguments: ...
  relations: ...
evidence: ...
```

### 4.1 `operator`

记录唯一 ID、名称、修订号和类别。类别只用于组织文件，不承担逻辑语义。

### 4.2 `requires.observed_domain_zh`

对测试输入域的完整中文摘要。它用于保留尚未结构化的随机生成、分支和组合条件。该字段对
人有用，但当前验证器不能把自然语言自动当成可信前提。

### 4.3 `requires.arguments`

逐个列出 wrapper 的所有输入参数：

```yaml
input:
  kind: tensor
  constraints:
  - {op: rank_equals, tensor: input, value: 4}

stride:
  kind: scalar_or_shape_parameter
  default: null
  constraints:
  - {op: value_in, value: stride, values: [null, 1, [2, 3]]}
```

`required: true` 表示调用时必须提供；`default` 只是接口默认值，不表示唯一合法值；
`optional: true` 表示可以为 `null`，相应张量约束仅在其非空时适用。

### 4.4 `requires.relations`

只依赖一个参数的条件放在该参数的 `constraints`。同时依赖多个参数的条件放在
`relations`：

```yaml
relations:
- {op: same_shape, values: [input, other]}
- {op: dim_equal, left: [input, 1], right: [mat2, 0]}
```

### 4.5 `evidence`

记录上游仓库、固定 commit、许可证、测试文件、测试函数和行号。`inferred_from_tests` 的
含义是“由有限测试归纳”，不是 `formally_verified`。`reviewed: true` 表示人工检查过提取
结果，不表示逻辑命题已经证明。

JSON Schema 目前检查字段结构和允许的谓词名称，但还没有对每个谓词做完整的参数类型检查
或可满足性检查。

## 5. 当前支持的前置条件类型与例子

下面按算子开发人员通常关心的问题分类。谓词名称是机器格式，右侧给出直观含义和现有例子。

### 5.1 Rank

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `rank_equals` | rank 等于固定值 | `adaptive_avg_pool2d`: `rank(input)=4` |
| `rank_between` | rank 位于闭区间 | `abs`: `1 <= rank(input) <= 4` |
| `rank_in` | rank 属于有限集合 | `diag`: `rank(input) in {1,2}` |

### 5.2 Shape 和维度

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `shape_equals` | 整体 shape 等于常量或符号 shape | `avg_pool2d`: `(2,3,112,112)`；`addmm.mat1`: `(m,k)` |
| `all_dims_between` | 每一维大小在闭区间内 | 普通浮点测试张量各维在 `[1,1024]` |
| `dim_equal` | 两个指定维度相等 | `mm`: `input.dim(1)=mat2.dim(0)` |
| `dim_even` | 指定维度为偶数 | rotary embedding 的最后一维为偶数 |
| `channel_axis` | 指定轴解释为 channel 轴 | batch norm 的 axis 1 是 `C` |
| `nonempty_shape_suffix` | 参数是输入 shape 的非空后缀 | layer norm 的 `normalized_shape` |
| `optional_shape_equals` | 可选张量非空时具有指定 shape | batch norm 的 weight/bias 为 `None` 或 `(C,)` |
| `same_shape` | 多个张量 shape 相同 | 当前 `add` 测试要求 input/other 同 shape |
| `all_same_shape` | 张量序列中的所有张量 shape 相同 | `stack(tensors)` |

### 5.3 Dtype 和 device

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `dtype_in` | dtype 属于集合 | `abs.input` 为 float16/float32 |
| `same_dtype` | 多个输入 dtype 相同 | `add.input` 与 `add.other` 同 dtype |
| `all_same_dtype` | 张量序列中所有 dtype 相同 | `stack(tensors)` |
| `device_is` | 固定设备 | 当前大多数测试固定为 CUDA |

当前格式尚未覆盖 device 间关系、dtype promotion 规则、memory format 和 layout 集合。

### 5.4 广播、矩阵与注意力关系

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `broadcastable` | 多个 shape 满足广播条件 | `maximum(input, other)` |
| `integer_between` | 一组符号整数在闭区间内 | matmul 的 `m,n,k in [1,1024]` |
| `head_count_divisible` | query head 数可被 key/value head 数整除 | grouped-query attention |

`broadcastable` 目前是一个高层谓词；进入 SMT 前仍需展开成逐维规则。

### 5.5 Axis 和 index

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `valid_axis` | `-rank <= dim < rank` | `logsumexp.dim`、`sort.dim` |
| `nonnegative_valid_axis` | `0 <= dim < rank` | `argsort.dim` |
| `optional_valid_axis` | `dim=None` 或合法正/负轴 | `max.dim`、`mean.dim` |
| `optional_nonnegative_valid_axis` | `dim=None` 或合法非负轴 | `quantile.dim` |
| `two_distinct_valid_axes` | 两个 axis 均合法且不同 | `rot90.dims` |
| `index_in_axis` | index 位于指定轴范围内 | `select_copy.index` |

### 5.6 标量、枚举和参数组合

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `equals` | 参数固定等于一个值 | `ceil_mode=false`、`out=null` |
| `value_in` | 参数属于有限集合 | `kernel_size in {(1,1),(3,3)}` |
| `value_between` | 标量位于区间 | dropout 的 `0 <= p < 1` |
| `value_pair_in` | 两个参数只能取列出的配对 | max_pool1d 的 `(kernel,stride)` |
| `integer_greater_equal` | 整数具有下界 | `bincount.minlength >= 0` |

区间端点是否包含由 `upper_inclusive` 等字段表示；缺省约定需要在后续谓词语义文档中固定。

### 5.7 张量元素和值生成方式

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `all_elements_nonnegative` | 每个元素非负 | `bincount.input` |
| `all_values_in_half_open_interval` | 每个值位于半开区间 | `quantile.q` 的每个值在 `[0,1)` |
| `generated_by` | 记录测试如何生成输入 | `input` 由 `torch.randn` 生成 |

`generated_by` 更接近 provenance，而不是理想的数学谓词。例如 `torch.randn` 并不能简洁
定义一个有限实数区间。后续应尽量把验证真正需要的值域写成显式谓词，把生成器只留作测试
复现信息。

### 5.8 张量序列和逻辑组合

| 谓词 | 含义 | 例子 |
| --- | --- | --- |
| `sequence_length_between` | 张量列表长度范围 | stack 接收 2--5 个张量 |
| `not_both` | 两个条件不能同时成立 | attention 中 mask 非空与 causal=true 不同时出现 |
| `is_not_null` | 参数非空 | 上述 `attn_mask` 子条件 |
| `is_true` | 布尔参数为真 | 上述 `is_causal` 子条件 |

## 6. 以 `avg_pool2d` 展示当前 Spec 与 Torch 的差距

当前测试派生 Spec 大致为：

```text
input.shape = (2,3,112,112)
input.dtype in {float16,float32}
input.device = cuda
kernel_size in {(1,1),(3,3)}
stride in {None,1,(2,3)}
padding in {0,1,(2,3)}
ceil_mode = false
count_include_pad = true
divisor_override = None
```

但测试还包含一个组合条件：每个方向的 padding 不能超过对应 kernel 的一半；不满足时直接
skip。这个条件目前只出现在 `observed_domain_zh`，`relations` 仍为空。也就是说，人能够正确
理解，机器暂时可能把被跳过的笛卡尔积组合也看作合法。这是需要补成结构化关系的已知缺口。

PyTorch 的 [`torch.nn.functional.avg_pool2d` 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.avg_pool2d.html)
允许更一般的输入空间：kernel、stride 和 padding 接受标量或 tuple，`stride` 缺省时取
kernel size，`ceil_mode` 和 `count_include_pad` 均有两种布尔取值，`divisor_override` 也可
指定。文档还明确给出了 padding 与 effective kernel 的约束。

因此，当前 Spec 不是 Torch Spec 的另一种写法，而是 Torch 合法域中的一个很小子集，并且
这个子集仍有一个尚未机器化的组合条件。

## 7. 能否直接从 Torch 提取集中式 Spec

### 7.1 结论

PyTorch 目前没有一份集中、机器可读、足以直接作为形式化验证前置条件的完整算子规格。
信息分散在多个来源中，每个来源只覆盖契约的一部分。

| Torch 来源 | 能提取什么 | 不能单独解决什么 |
| --- | --- | --- |
| [逐算子 API 文档](https://docs.pytorch.org/docs/stable/torch.html) | Python 签名、默认值、自然语言 shape/参数约束、部分公式 | 格式不完全统一；dtype/backend/边界行为经常不完整 |
| [`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) | 集中的 ATen schema、overload、参数类型、默认值、返回类型、alias/mutation 标注和 dispatch 元数据 | 通常不描述完整 rank、shape、值域或所有合法参数组合 |
| [ATen native schema 说明](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md) | `Tensor?`、`Tensor[]`、`int[2]`、`Tensor(a!)` 等 schema 与 alias 语义 | 是格式说明，不是每个算子的完整行为规范 |
| [OpInfo](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/core.py) 与 PyTorch 测试 | device/dtype 支持、sample inputs、skips、参考函数、梯度测试元数据 | sample inputs 仍是有限集合；内部测试 API 可能变化 |
| [Meta/FakeTensor](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fake_tensor.html) | 不读真实数据地传播输出 shape、stride、dtype、device；触发部分合法性检查 | 不是声明式输入域；数据相关条件和未覆盖 dispatch 仍需其他来源 |
| [Symbolic Shapes/ShapeEnv](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html) | 执行时产生 shape guards、范围与相等关系 | guards 依赖具体 trace 路径，不能保证枚举所有合法路径 |
| 实现、decomposition、error tests | 数学计算、实际断言、异常和特殊路径 | 后端相关，提取成本高；实现不是独立于实现的规范 |

PyTorch 官方对 custom op 的 [`torch.library.opcheck`](https://docs.pytorch.org/docs/stable/library.html)
能够在给定样例上检查 schema、FakeTensor、autograd 注册和编译路径的一致性，但官方也明确将
它定位为 sample-based registration checking；它不会自动生成完整合法输入域，也不是数学
正确性的形式化证明。

### 7.2 推荐的 Torch 多源提取方法

如果下一阶段从 Torch 而不是 ntops test 出发，建议采用以下流程：

1. **固定 Torch 版本或 commit**：所有来源必须来自同一个版本，Spec ID 包含 overload。
2. **以 dispatcher schema 为骨架**：解析 `native_functions.yaml`，或从安装版本读取
   `torch.ops.aten.<op>.<overload>._schema`，得到参数类型、默认值、返回值和 alias/mutation。
3. **以官方文档补充声明式条件**：解析 Sphinx 生成源或 docstring，抽取 rank、shape、参数
   范围和公式。自然语言到谓词的转换可以由规则或 LLM 提议，但必须保留原文定位并人工确认。
4. **以 OpInfo 扩充 dtype/device 和边界样例**：收集 sample inputs、supported dtypes、skips、
   error inputs 和 reference function。它们作为证据，不直接提升为全称事实。
5. **运行 Meta/Fake kernel**：在符号 shape 上执行，收集输出 metadata、ShapeEnv guards 和
   抛出的合法性检查，转换成候选 shape 谓词。
6. **分析 implementation/decomposition**：对文档和 Meta kernel 没表达的值域、特殊值、
   backend 分支和异常条件做定向抽取。
7. **差分与边界测试**：围绕候选谓词边界自动生成样例，检查 Torch 接受/拒绝行为，发现
   过强或过弱条件。
8. **人工批准并分级 provenance**：每条谓词标记来源，例如 `schema`、`docs`、`opinfo`、
   `meta_guard`、`implementation`、`test_inference` 或 `developer_asserted`。

不建议直接把 Torch 文档自然语言交给 LLM 后当作可信 Spec。LLM 适合生成候选谓词和定位
冲突，不适合在没有源码位置、版本和人工批准的情况下提升为验证公理。

## 8. 当前 75 个 Spec 与对应 Torch 算子的共性 Gap

| 维度 | 当前 ntops Spec | 对应 Torch 能力/契约 | 对验证的影响 |
| --- | --- | --- | --- |
| shape | 常见为固定 shape 或小范围生成器 | 通常支持更广的动态 shape | 证明范围显著小于 API 范围 |
| rank | 主要来自测试生成器 | 有些算子允许标量、空张量或更高 rank | 边界 rank 未覆盖 |
| dtype | 多数仅 float16/float32，少量整数 | Torch 常按 backend 支持更多 dtype，并有 promotion | 不能证明其他 dtype；promotion 未建模 |
| device | 基本固定 CUDA | Torch 通常有 CPU/CUDA 及其他 backend | 当前结论是 CUDA 子域 |
| layout/stride | 多数为 contiguous `randn` | Torch 可能支持 non-contiguous、channels-last、sparse 等 | 地址计算和 alias 风险未覆盖 |
| 参数组合 | 主要是有限枚举 | Torch 文档往往描述参数关系和更广范围 | 有效笛卡尔积可能被高估或低估 |
| tensor value | 多数是 `randn` | Torch 语义涉及 NaN/Inf/signed zero/溢出/除零 | IEEE-754 和整数边界不完整 |
| broadcasting | 多数二元测试要求同 shape | Torch 二元算子通常允许广播 | 现有 Spec 可能过强 |
| optional/out/inplace | 只记录测试中是否使用 | Torch schema 含 alias 和 mutation 契约 | 当前 requires-only YAML 不足以验证副作用 |
| exception/definedness | 测试常 skip 或直接 return | Torch 对非法调用有具体异常行为 | 无法区分应拒绝与暂未测试 |
| 随机算子 | 只记录有限统计测试输入 | 完整契约需要 RNG 状态和分布语义 | 不能用普通逐元素等价表达 |
| reference 映射 | 按测试实际调用 | 名称可能与 reference 不完全一致 | 例如现有 `fmax` 测试实际调用 `torch.maximum` |
| 数值接受标准 | 已从 YAML 删除 | Torch 与 kernel 可能允许 backend 相关误差 | 后续证明必须另行定义 exact/allclose/ULP |

现有测试还存在直接影响 Spec 的情况：部分 float16 分支在断言前返回，`gelu(tanh)`、
`div(trunc)` 和 `quantile(float16)` 被 skip；排序/中位数对重复值的 index 语义覆盖不完整。
这些都说明“测试中出现过某参数”不等于“该参数路径已验证”。

## 9. 建议由算子开发人员提供的最小信息

为了把测试派生 Spec 升级成开发者确认的支持契约，每个算子至少回答以下问题：

1. 支持哪些 rank；是否支持 0-D、空维和空张量；
2. 每个输入的 shape 规则，以及输入之间的维度相等、广播、整除关系；
3. 每个 backend 支持哪些 dtype，是否存在 dtype promotion；
4. 支持哪些 device、layout、stride、alignment 和 memory format；
5. 标量参数的合法范围与参数组合约束；
6. tensor 元素是否有范围限制，如何处理 NaN、Inf、signed zero、溢出和除零；
7. 可选参数何时允许为空；`out=`、inplace、view、alias 和状态更新行为；
8. 非法输入应抛异常、返回特殊值，还是属于未定义行为；
9. 与哪个 Torch operator overload 对齐，以及明确不对齐的行为；
10. 这是“当前实现支持”“计划支持”还是“已经验证”的范围。

建议开发人员直接审阅 YAML diff，而不是维护第二份自然语言表格。自然语言文档用于解释，
YAML 才是后续生成测试、SMT 前提和验证报告的单一输入。

## 10. 建议的后续工作顺序

1. 先修复现有 YAML 中仍停留在 `observed_domain_zh` 的组合条件，优先处理 pooling、attention、
   normalization 和带 cache/out 的算子；
2. 为每个谓词定义精确逻辑语义、字段类型和区间端点约定，并增加 satisfiability 检查；
3. 增加 `support_level` 与逐谓词 provenance，区分 test-inferred 和 developer-asserted；
4. 建立 ntops 算子到 Torch overload 的显式映射；
5. 实现 Torch 多源提取器，先用 `avg_pool2d`、`add`、`mm`、`layer_norm` 和 attention 做代表性
   试点；
6. 在输入前提稳定后，再单独定义输出语义、数值关系、异常和副作用规格；
7. 最终验证报告必须同时显示使用的前置条件、未建模行为和证明边界。

## 11. Meeting 中需要拍板的问题

- 我们要验证的是“现有测试域”“当前实现域”还是“目标 Torch 子集”？
- 算子开发人员是否愿意把 YAML 作为支持域的评审对象和代码评审必需项？
- Torch 兼容以哪个版本、哪个 overload 为准？
- 超出已声明域的调用应明确报错，还是允许未定义行为？
- 哪些算子优先补齐复杂 relation、layout、特殊值和 alias 契约？
- 每条新增条件由谁批准，什么证据等级才允许进入形式化验证假设？

如果这些问题没有明确答案，验证器仍能对有限样例或狭窄前提给出结果，但该结果不能代表
完整算子兼容性。
