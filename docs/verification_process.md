# 完整验证过程

本文档集中说明 ETV 的输入契约、TTIR 前端、语义模型、证明义务、等式饱和、结果解释与信任边界。它既是用户准备输入的格式说明，也是理解 `PROVED` 含义的完整参考。

## 验证目标

对于固定的 `PairSpec` 假设 `P`，ETV 的 `PROVED` 结果表示：

```text
对于逻辑输入/标量值以及初始内存 M0 的每一种赋值，
如果 P 成立，那么
  两个有限的已提升 Semantic TTIR 启动配置在当前语义中都有定义，
  契约要求的每个 Output 元素都恰好写入一次，
  并且最终可观察的 Output 内存相等。
```

这是针对固定程序对和固定 specialization 的有界翻译验证。启动维度、形状、步长和向量宽度由 PairSpec 固定；地址与掩码在整个有限启动域中穷举，但逻辑内存读取仍保留为符号值，因此计算等价结论覆盖任意输入值，而不是一次数值测试。

完整流水线为：

```text
PairSpec + raw TTIR 或 Semantic TTIR A/B
          |
          v
输入 schema、角色、事实与契约校验
          |
          v
libtriton 解析/验证 + 白名单语义提升（raw TTIR）
          |
          v
固定启动域符号求值
          |
          +--> 掩码、覆盖性、地址对应与无写入竞争
          +--> load/标量逻辑角色归一化
          +--> 浮点操作定义域检查
          |
          v
egglog 联合 e-graph 等式饱和
          +--> Z3 准入的代数规则
          +--> PairSpec facts 启用的可信规则
          |
          v
最终可观察内存比较
          |
          v
PROVED / DISPROVED / UNKNOWN + JSON/Markdown 报告
```

## 输入一：原始 TTIR

ETV 使用锁定的 Triton/libtriton 3.7.1 接受原始 `.ttir` 和 `.mlir`，不维护第二套 TTIR 文法。前端分为两个承诺不同的层次：

```text
原始 .ttir/.mlir
       |
       v
libtriton 3.7.1 解析 + MLIR/Triton verifier
       |
       v
etv-ttir-snapshot-v1（操作、SSA 边、类型、位置、规范文本）
       |
       v
受控语义提升（当前为无环逐点单 store 子集）
       |
       v
Semantic TTIR -> ETV 证明流水线
```

第一层接受 Triton 3.7.1 注册方言能够解析和验证的完整语法。快照的 `canonical_assembly` 保留 libtriton 的规范打印结果，通用操作表记录稳定编号的 operand/result、类型、block、region 数和源码位置。

第二层只提升具有已实现语义规则的白名单子集。`scf.for`、`tt.reduce`、原子操作以及其他合法但未建模的操作可以被 `etv parse` 解析，但 `etv check` 会返回具体原因的 `UNKNOWN`，不会把“解析成功”误报为“语义已支持”。

TTIR 示例：

```mlir
module {
  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %input = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %value = tt.load %input_ptr : tensor<128x!tt.ptr<f32>>
    %output = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %output_ptr, %value : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
```

常用命令：

```bash
python -m etv parse examples/ttir/add_mul.ttir \
  --out build/add_mul.snapshot.json
python -m etv parse input.ttir --no-assembly
python -m etv inspect input.ttir
python -m etv check examples/specs/add_raw_ttir_proved.json
```

当前提升范围：

| 类别 | 已提升操作 |
| --- | --- |
| 网格/形状 | `tt.get_program_id x`、`tt.make_range`、`tt.splat`、`tt.broadcast`、`tt.expand_dims`、保元素数 `tt.reshape` |
| 整数 | `arith.addi/subi/muli/divsi/remsi`、signed `arith.cmpi`、布尔 `andi/ori`、常用整数 cast |
| 抽象浮点 | `arith.addf/subf/mulf/divf/negf`、`math.sqrt/rsqrt/fma`、浮点宽度 cast |
| 内存 | 标量/张量 `tt.addptr`、`tt.load`、单个 `tt.store` |
| 选择 | `arith.select` |

静态 tensor shape 会被标量化为 `lane` 的多维索引，并支持 singleton 维广播。非 splat dense 常量、无符号比较、动态 shape、多结果操作、region/循环、归约、原子操作、共享内存和 block pointer 语义当前返回 `UNKNOWN`。

对没有 `other` 的 masked load，只有在它使用与最终 store 完全相同的 SSA mask 时才提升；否则被屏蔽 lane 的未定义值可能被观察，ETV 返回 `TTIR_UNDEFINED_LOAD_LANE`。

## 输入二：Semantic TTIR

Semantic TTIR 使用严格 JSON 格式 `etv-semantic-program-v1`。未知键会被拒绝。下面是逐点程序的主体示例：

```json
{
  "format": "etv-semantic-program-v1",
  "name": "linear_add",
  "launch": {
    "programs": {"op": "ceildiv", "args": [{"var": "X"}, 128]},
    "lanes": 128
  },
  "stores": [{
    "block": "out_ptr0",
    "logical_index": {"op": "iadd", "args": [
      {"op": "imul", "args": [{"var": "pid"}, 128]}, {"var": "lane"}
    ]},
    "offset": {"op": "iadd", "args": [
      {"op": "imul", "args": [{"var": "pid"}, 128]}, {"var": "lane"}
    ]},
    "mask": {"op": "lt", "args": [
      {"op": "iadd", "args": [
        {"op": "imul", "args": [{"var": "pid"}, 128]}, {"var": "lane"}
      ]}, {"var": "X"}
    ]},
    "value": {"op": "load", "block": "in_ptr0",
              "offset": {"op": "iadd", "args": [
                {"op": "imul", "args": [{"var": "pid"}, 128]}, {"var": "lane"}
              ]},
              "mask": true, "default": {"float": "0"}}
  }]
}
```

`pid` 和 `lane` 由求值器提供，其他整数变量必须在 PairSpec 中绑定。

表达式语言：

| 类型 | 操作 |
| --- | --- |
| i32 | `iadd`、`isub`、`imul`、`idiv`、`irem`、`ceildiv` |
| bool | `lt`、`le`、`gt`、`ge`、`eq`、`ne`、`and`、`or`、`not` |
| 抽象浮点 | `fadd`、`fsub`、`fmul`、`fdiv`、`fneg`、`fsqrt`、`frsqrt`、`fma` |
| 多态 | `select` |
| 内存/输入 | `load`、`scalar` |

整数和布尔字面量使用 JSON 字面量；抽象浮点常量使用精确有理数字符串，如 `{"float": "0"}` 和 `{"float": "1/6"}`。`load` 包含 `block`、`offset`、`mask` 和 `default`。

## PairSpec 契约

PairSpec 使用严格 JSON 格式 `etv-pair-v1`，主要字段如下：

- `lhs`、`rhs`：相对于 spec 文件的 Semantic JSON 或 raw TTIR 路径；
- `frontends`：可选的每侧前端配置，raw TTIR 必须提供 launch program 数；
- `semantic_mode`：当前仅支持 `abstract_float`；
- `roles`：两个物理 ABI 的逻辑角色对应关系；
- `facts.bindings`：固定的整数形状、步长和启动配置；
- `facts.assumptions`：报告中的可信前提，也可供 `trusted_fact` gate 精确匹配；
- `facts.disjoint`：成对不相交的逻辑 block 分组；
- `contract`：可观察输出、大小、覆盖性和所需 no-alias 角色；
- `limits`：正数形式的 e-graph 迭代、节点和超时预算；
- `rewrite_rules`：可选的局部代数规则或 fact-gated 可信规则。

角色端点包括 `block`、`scalar` 和 `scalar_block`。把 lhs 的 `scalar` 和 rhs 的 `scalar_block` 映射到同一逻辑角色，可以归一化 kernel 标量参数与 0-D 内存参数之间的 ABI 差异。

每侧 `frontends` 配置支持：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `kind` | `auto`、`semantic_json`、`ttir` | 默认 `auto`，按扩展名判断 |
| `function` | 非空字符串 | 指定 TTIR 模块中的入口函数 |
| `programs` | 整数表达式 | TTIR 未编码的宿主 launch program 数 |

```json
{
  "lhs": "../ttir/add_mul.ttir",
  "rhs": "../ttir/add_fma.ttir",
  "frontends": {
    "lhs": {"kind": "ttir", "function": "add_mul", "programs": 1},
    "rhs": {"kind": "ttir", "function": "add_fma", "programs": 1}
  }
}
```

raw TTIR 参数按稳定位置命名为 `arg0`、`arg1` 等。PairSpec 的角色映射与绑定使用这些名称，而不依赖原始 SSA 拼写。当前逐点提升以 store tensor 的静态元素数作为 `lanes`，逻辑输出索引为 `pid * lanes + lane`；非线性域尚需显式 frontier 支持。

## 语义模型

### 整数与布尔

- 整数变量和中间值使用有符号 i32；
- `idiv` 和 `irem` 向零截断，与 MLIR 有符号除法一致；
- 具体中间值超出 i32 范围时返回 `UNKNOWN(INTEGER_OVERFLOW)`；
- 每个枚举 lane 的掩码和地址都会完全求值。

因此当前证明是固定维度上的有限证明，不是关于任意维度的参数化定理。

### `ABSTRACT_FLOAT`

浮点叶节点取精确数学值，`fadd`、`fsub`、`fmul`、`fdiv` 与 `fma` 按实数算术解释。该模式有意忽略舍入、NaN、无穷大、有符号零、非规格化数、上溢/下溢、融合与非融合舍入差异、硬件近似及归约顺序。

`fdiv`、`fsqrt` 和 `frsqrt` 是偏函数。当前只有分母或被开方数可约简为满足条件的精确常量时，才能完成定义域证明；否则返回 `UNKNOWN(FLOAT_DEFINEDNESS_NOT_PROVED)`。人类可读假设不能绕过该检查。

### 内存

PairSpec 将物理参数映射到 `Input`、`Other`、`Alpha`、`Output` 等逻辑角色。Load 被归一化为：

```text
read(logical_block, concrete_element_offset)
```

Store 被建模为内存 token 转换：

```text
store(M0, Output, offset, symbolic_value, mask) -> M1
```

逻辑 block 是从 i32 元素偏移到值的全映射。PairSpec v1 没有 buffer 分配大小，因此 `PROVED` 不包含 GPU 越界安全结论。当前还要求每侧只有一个输出 store 模板、必要 block 明确 no-alias，并证明有效输出地址单射。

## 证明义务

验证器按以下顺序处理：

1. 校验输入 schema、语义模式和资源限制；
2. 对齐两侧 ABI 参数的逻辑角色，并检查绑定与 no-alias 前提；
3. 枚举固定启动域中的每一个 `(program_id, lane)`；
4. 证明两侧逻辑掩码域相等且完整覆盖契约输出；
5. 证明每侧有效输出地址单射，排除写入竞争；
6. 证明相同逻辑输出对应相同物理 Output 地址；
7. 将 load 与标量归一化为共享逻辑叶节点，并比较读取前沿；
8. 证明浮点偏函数在所有被观察路径上有定义；
9. 将两侧计算根放入同一个 egglog e-graph，证明值等价；
10. 由值、地址、掩码、覆盖性和框架条件推出最终可观察内存相等。

地址或掩码差异会产生有限域见证。计算根未合并时，ETV 继续搜索确定性的精确有理数反例；找到反例返回 `DISPROVED`，否则保持 `UNKNOWN`。

## egglog 等式饱和

ETV 0.3.0 使用 `egglog==13.2.0` 作为唯一等式饱和后端。`etv/egraph.py` 将类型化表达式编码为包含操作、子节点、附加数据和 sort 的 egglog term，并读取逐轮 `RunReport`。hash-consing、e-matching、union-find 和同余重建均由 egglog 提供。

所有逻辑输出位置的 lhs/rhs 计算根进入同一个 e-graph。当前不按操作族筛选规则、不划分子图，也不提取最低成本表达式。统一饱和过程为：

1. 用 Z3 准入所有内建及 PairSpec 声明的 `algebraic` 规则；
2. 用 PairSpec facts 检查所有 `trusted_fact` 规则的 gate；
3. 将全部已准入规则放入同一个 `etv_unified` ruleset；
4. 对所有尚未等价的输出根统一运行饱和；
5. 用 egglog e-class 等同性判定对应计算根是否等价。

结构相等、同余闭包、代数重写与启用的事实规则由此在同一个 e-graph 中共同生效。

### 规则类别与准入

| 类别 | 准入方式 | 证据 | 信任影响 |
| --- | --- | --- | --- |
| `algebraic` | Z3 检查 `lhs != rhs` 为 UNSAT | `ALGEBRAIC` | Z3 与编码属于可信计算基 |
| `trusted_fact` | 检查 PairSpec fact gate | `TRUSTED_AXIOM` | 等式本身未经验证，可能破坏健全性 |

内建代数规则包括交换律、结合律、零元、单位元、减法定义、双重取负、加法逆元、乘法分配律，以及 `fma(a,b,c) = a*b+c`。它们只对 `ABSTRACT_FLOAT` 的实数语义成立，不是 IEEE-754 位级规则。

PairSpec 也可以声明规则：

```json
"rewrite_rules": [{
  "id": "specialized_sub_is_add",
  "kind": "trusted_fact",
  "statement": "该 specialization 中 fsub(a,b) 等价于 fadd(a,b)",
  "lhs": {"op": "fsub", "args": [{"match": "a"}, {"match": "b"}]},
  "rhs": {"op": "fadd", "args": [{"match": "a"}, {"match": "b"}]},
  "requires": [{
    "kind": "assumption",
    "text": "subtraction is equivalent to addition for this specialization"
  }],
  "provenance": {"generated_by": "human"}
}]
```

`algebraic` 规则不能包含 fact gate，且必须能够被 Z3 Real 编码并证明。`trusted_fact` 至少包含一个 gate：

```json
{"kind": "binding_equals", "name": "N", "value": 16}
{"kind": "assumption", "text": "lhs and rhs layouts denote the same logical tensor"}
{"kind": "disjoint", "roles": ["Input", "Other", "Output"]}
```

gate 只检查事实是否在 PairSpec 中声明，**不证明该事实逻辑上蕴含规则等式**。这是当前明确保留的健全性缺口。错误的 `trusted_fact` 规则可能把不等价程序合并并产生不健全的 `PROVED`；报告会将其标为 `admitted_unverified` 和 `TRUSTED_AXIOM`，但警告本身不能消除风险。

规则模式使用 `{"match": "x"}` 表示 metavariable；rhs 的 metavariable 必须在 lhs 出现。精确值、逻辑输入与内存读取可写为：

```json
{"float": "1/2"}
{"input": "Alpha"}
{"read": {"role": "Input", "offset": {"match": "i"}}}
```

lhs 不允许是裸 metavariable，规则 id 必须唯一且不能覆盖内建规则。

### LLM 辅助边界

LLM 可以提出重写规则或未来的子图划分方案，但其输出不构成证明。规则可记录：

```json
"provenance": {
  "generated_by": "llm",
  "generator": "model-or-pipeline-id",
  "prompt_sha256": "64 位小写十六进制 SHA-256"
}
```

LLM 生成的 `algebraic` 候选仍必须通过 Z3；非代数候选必须显式声明为 `trusted_fact` 并暴露未验证警告。当前 CLI 不联网调用模型，也不会自动写入规则或进行 LLM 子图划分。

egglog 按 `max_iterations`、`max_enodes` 和 `timeout_ms` 逐轮运行。报告记录准入状态、SMT 或 fact gate 结果、provenance、egglog 匹配次数和是否实际使用。匹配次数是聚合统计，不是可独立检查的最小证明。

## 结果与证据

| 状态 | 含义 |
| --- | --- |
| `PROVED` | 所有必要证明义务都在列出的假设与语义下完成。 |
| `DISPROVED` | 有限地址/掩码见证或具体抽象值模型证明两者不相等。 |
| `UNKNOWN` | 因缺少事实、语义、规则或预算，命题尚未判定。 |

`UNKNOWN` 不表示不等价，随机测试或性质测试也绝不会被提升为 `PROVED`。

| 证据等级 | 用途 |
| --- | --- |
| `BOUNDED_EXHAUSTIVE` | 对固定启动域完整枚举 |
| `STRUCTURAL` | 通过语法导向方法完成定义性义务 |
| `CONGRUENCE` | 相同操作与相等子 e-class 推导父节点相等 |
| `ALGEBRAIC` | Z3 UNSAT 准入的抽象实数重写 |
| `TRUSTED_AXIOM` | Pair 角色、绑定、no-alias、数值抽象及匹配的可信规则 |
| `EMPIRICAL` | 仅用于测试，不能触发生产合并 |

主要 TTIR 前端错误码：

| 错误码 | 含义 |
| --- | --- |
| `TTIR_FRONTEND_UNAVAILABLE` | 没有可加载的 3.7.1 libtriton |
| `TTIR_VERSION_MISMATCH` | 运行时版本不是锁定版本 |
| `TTIR_PARSE_ERROR` | 官方 MLIR parser 拒绝输入 |
| `TTIR_VERIFY_ERROR` | parser 成功但 IR verifier 失败 |
| `TTIR_FUNCTION_REQUIRED` | 模块入口存在歧义 |
| `TTIR_LAUNCH_REQUIRED` | PairSpec 未声明 launch program 数 |
| `TTIR_OP_UNSUPPORTED` | 操作合法，但当前无语义提升规则 |
| `TTIR_REGION_SEMANTICS_UNSUPPORTED` | 控制流或归约 region 尚未建模 |

## 信任边界与不保证事项

当前结果信任：

1. libtriton 3.7.1 parser、verifier 与规范打印；
2. ETV 的 TTIR 到 Semantic TTIR 提升器，或用户直接提供的 Semantic TTIR；
3. PairSpec 的角色对应、固定绑定、assumption 与 no-alias 声明；
4. 求值器与内存 token 实现；
5. Z3 实数算术结果、egglog 13.2.0 及 ETV term 编码；
6. 所有实际匹配的 `trusted_fact` 规则本身确实正确；
7. 用户接受 `ABSTRACT_FLOAT` 解释。

当前没有导出可由独立内核逐步检查的 egglog proof certificate。解析完整性与语义提升范围也是独立概念：`PROVED` 不表示 libtriton 能接受的任意程序均已被 ETV 建模。

ETV 当前不保证：IEEE-754 或 GPU 位级等价、容差等价、buffer 边界安全、动态 shape、循环/归约、原子或共享内存语义、多 kernel 行为，以及超出固定 specialization 的参数化正确性。每份报告都会列出适用假设和相应限制。

完整 Semantic JSON 示例见 `examples/specs/add_proved.json`，raw TTIR 示例见 `examples/specs/add_raw_ttir_proved.json`。安装与锁定依赖见 [依赖与环境](dependencies.md)，组件实现见 [架构](architecture.md)，当前覆盖范围见 [实现状态](implementation_status.md)。
