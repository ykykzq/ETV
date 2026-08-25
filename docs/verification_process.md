# 完整验证过程

本文描述 ETV 0.5.0 的输入契约、符号化、条件规则、e-graph 判定和结果边界。

## 验证目标

对 PairSpec 前提 `P`、左侧九齿 TTIR 程序 `L` 和右侧 Torch Prims 程序 `R`：

```text
P => Defined(L) and Defined(R)
     and ObservableOutput(L) = ObservableOutput(R)
```

参数化结论量化所有声明的 shape/launch 参数和所有抽象输入值。rank 本身固定，不在
量化范围内。当前只支持单输出、单 store、无循环的逐点子集。

## 输入

一次异构验证包含三个文件：

1. 九齿侧 raw TTIR；
2. Torch 侧 `etv-prims-program-v1`；
3. `etv-pair-v1` PairSpec。

PairSpec 片段：

```json
{
  "lhs": "ttir/parametric_2d_add.ttir",
  "rhs": "prims/torch_add.prims.json",
  "frontends": {
    "lhs": {
      "kind": "ttir",
      "function": "parametric_2d_add",
      "programs": {"op": "ceildiv", "args": [{"var": "c"}, 256]}
    },
    "rhs": {"kind": "prims"}
  },
  "facts": {
    "side_bindings": {
      "lhs": {"arg4": {"var": "a"}, "arg5": {"var": "b"}},
      "rhs": {"torch_dim0": {"var": "a"}, "torch_dim1": {"var": "b"}}
    },
    "parameters": {"a": {"min": 1}, "b": {"min": 1}, "c": {"min": 1}},
    "constraints": [
      {"op": "eq", "args": [{"op": "imul", "args": [{"var": "a"}, {"var": "b"}]}, {"var": "c"}]}
    ]
  },
  "llm": {"enabled": true, "generate_rules": false},
  "partition": {"enabled": true, "min_partitions": 2, "max_partitions": 16}
}
```

末尾两个字段是可选的子图划分配置。划分开启时必须显式开启 LLM；若只需划分而不需
失败后的规则生成，可将 `generate_rules` 设为 `false`。

未知字段、错误 frontend 元数据、重复物理角色、错误 sort、未绑定符号、空参数域、
无效资源限制或非法规则都会在验证前被拒绝。

## 第一步：两侧前端提升

### TTIR

libtriton 3.7.1 注册所有内置方言、解析模块并运行 IR verifier。ETV 随后提升已支持
的逐点操作：program id、range/splat/broadcast、整数索引、比较、pointer add、
load/store 和抽象浮点运算。未知合法 TTIR op 返回 `UNKNOWN`，不会文本猜测。

TTIR 中没有 host grid，PairSpec 的 `programs` 是必要输入。该表达式参与后续符号
证明，而不是先求成一个固定常量。

### Prims

Prims 前端严格读取固定 rank 输入和显式节点。广播必须由
`prims.broadcast_in_dim` 表示。每个张量被提升为 `element(indices)` 纯函数，输入
读取保留为物理 load。输出 shape 的乘积形成逻辑域。

Torch 侧不经过 TorchInductor。Prims 的内部 `programs/lanes` 只用于共同逻辑表示，
不是 GPU launch 推断。

## 第二步：共同参数与角色

ETV 合并共享 binding 和左右单侧 binding。以下量保持符号：

- 固定 rank 的每个 shape dimension；
- TTIR launch program 数和相关规模；
- `pid`、`lane`；
- 任意逻辑输出位置 `k`；
- load/store offset 和 mask 中的整数表达式。

以下内容不符号化：rank、TTIR tensor lane 数、操作 arity 和数据类型。它们必须在
前端结构上确定。

角色映射不会立即改写表达式。物理端点先保留 side 标签，例如：

```text
lhs:arg0   -> Input
rhs:input  -> Input
lhs:arg2   -> Alpha
rhs:alpha[0] -> Alpha
```

映射是之后生成关系规则的条件之一。

## 第三步：符号安全与覆盖义务

令：

```text
D(a,b,c) := parameter ranges and a*b=c and all required i32 definedness
K(k)     := 0 <= k < c
```

除参数域非空检查外，ETV 都查询：

```text
D and counterexample
```

`UNSAT` 表示整个参数域中不存在该类反例。证明义务包括：

1. 参数域非空，输出元素数始终为正；
2. 每侧 launch 数为正且整数运算有定义；
3. active lane 的 logical index 落在契约范围；
4. canonical writer `pid=k div lanes, lane=k rem lanes` 有效且 mask 为真；
5. 每个逻辑输出只有一个 writer；
6. active store 地址单射，无写竞争；
7. 对应输出地址满足声明关系。

这些查询关闭执行定义性和规则条件，但不直接宣布两侧程序等价。

## 第四步：建立未归一化重写目标

对任意 `k`，ETV 将 canonical writer 代入内部表达式，但不消除物理 load。例如：

```text
lhs_value = fadd(
  load(lhs:arg0, lhs_offset(k,b), lhs_mask(k,a,b), 0),
  fmul(input(lhs:arg2),
       load(lhs:arg1, lhs_offset(k,b), lhs_mask(k,a,b), 0)))

rhs_value = fadd(
  load(rhs:input, rhs_offset(k,a,b), true, 0),
  fmul(load(rhs:alpha, 0, true, 0),
       load(rhs:other, rhs_offset(k,a,b), true, 0)))
```

最终待验证根为：

```text
lhs_root = observe_store(lhs:arg3, lhs_output_offset, lhs_store_mask, lhs_value)
rhs_root = observe_store(rhs:output, rhs_output_offset, true, rhs_value)
```

两根初始必须不同。若前端错误地提前删除 side/地址/mask，测试中的
`initial_state.unmatched_root_pairs == 1` 会失败。

## 第五步：生成并验证关系规则

### load 规则

每个被观察 load 生成一条 ground 条件规则：

```text
load(side:physical_block, physical_offset(k), mask(k), default)
  -> read(LogicalRole, k)
```

准入条件为：

1. PairSpec 将该物理端点映射到 `LogicalRole`；
2. 在 `D and K(k)` 下 mask 恒真；
3. 在同一域上 physical offset 对应逻辑位置 `k`；
4. 所有整数中间值有定义。

rank-0 scalar block 规则把 offset 0 的 load 改写为 `input(Alpha)`。

### scalar 规则

直接 scalar ABI 由 PairSpec 角色事实产生：

```text
input(lhs:arg2) -> input(Alpha)
```

该映射属于 PairSpec 的可信前提。

### store 规则

每侧生成：

```text
observe_store(side:physical_output, offset(k), mask(k), value)
  -> observe_store(Output, k, true, value)
```

`value` 是 pattern 变量，因此 load/计算子树仍须通过自己的规则和同余闭包连接。

每条自动规则在报告中同时有：

- `rule_admission`：条件、求解器结果和 query hash；
- `rule_application`：egglog match count 和是否实际使用。

准入成功但 match 为 0 的规则不构成证明证据。

## 第六步：可选的成对子图划分

当 `partition.enabled=true` 时，ETV 在计算重写前对两侧完整程序做一次 LLM 扫描。
请求包含完整 launch/store 语义、PairSpec 上下文和全部待验证计算根；相同左右拓扑
的根实例组成 family，模型只需为每个 family 选择对应的左右节点路径。

ETV 对返回值执行以下机器检查：每个 family 必须有唯一整根分区；同侧路径不得重复；
非根分区必须是浮点运算节点；左右最近父分区必须相同；依赖 DAG 必须无环；分区数
必须落在配置范围。模型提供的 `semantic` 只用于日志，不参与证明。

例如模型可以给出：

```json
{
  "partitions": [
    {
      "id": "scale_other",
      "family": "family_0",
      "semantic": "Alpha 乘 Other",
      "lhs_path": "root.args[1]",
      "rhs_path": "root.args[1]"
    },
    {
      "id": "output_add",
      "family": "family_0",
      "semantic": "与 Input 相加",
      "lhs_path": "root",
      "rhs_path": "root"
    }
  ]
}
```

ETV 自动得到 `scale_other -> output_add`。先用独立 e-graph 证明乘法子图，再把左右
乘法根替换为同一个 `input(partition:...)`，最后证明加法父图。根分区覆盖没有单独
选出的节点，因此划分不会丢掉程序语义。任何提案/局部证明失败都回退到整图路径；
局部抽象反例不直接成为全程序反例。

## 第七步：e-graph 重写

若启用并成功完成第六步，以下阶段在每个 `SubgraphBatch` 的独立 e-graph 中运行，
顺序为依赖子图到父图；否则在原联合图中运行。

执行阶段为：

```text
FACT_DERIVED_RELATIONAL_REWRITES
ALGEBRAIC_REWRITES                 # 需要时
LLM_ASSISTED_REWRITES              # 显式启用且前两阶段失败时
```

第一阶段运行 load/scalar/store 规则。egglog 将 canonical 表示 union 到原表达式的
e-class，随后通过同余闭包逐层合并 `fmul`、`fadd` 和 `observe_store` 父节点。

如果计算结构仍不同，第二阶段运行 17 条内建抽象代数规则及 PairSpec 规则。代数规则
默认须通过 Z3 Real 的 `lhs != rhs` 不可满足检查。规则本身的证明与应用日志分离。

每轮记录：

```json
{
  "iteration": 1,
  "before": {"enodes": 31, "eclasses": 31},
  "after": {"enodes": 33, "eclasses": 25},
  "rule_matches": {"parametric_load_lhs_0": 1},
  "updated": true
}
```

实际数值取决于输入；这是报告结构示例。

## 第八步：规则缺失时的 LLM 辅助

当根仍未连接且 `llm.enabled=true`：

1. 模型从现有表达式树选择左右节点路径；
2. 模型生成必须复制现有 fact requirement 的候选规则；
3. schema 拒绝未知 op、未绑定 metavariable、无条件规则和非法 provenance；
4. 候选按相同策略准入；
5. 只有重新饱和后的 e-class 合并影响结论。

LLM 请求/响应 hash、模型、token usage、选择节点和规则 ID 进入报告。API key 只从
环境变量读取，不写入 PairSpec 或报告。

子图验证期间禁用额外规则生成，避免每个小图再次调用模型。若局部证明失败，回退的
整图验证仍可按本节流程使用原有规则生成能力。

## 第九步：判定

参数化异构路径的成功条件是：

```text
all required symbolic obligations proved
and (
  eclass(lhs_observe_store_root) == eclass(rhs_observe_store_root)
  or all paired subgraph obligations are proved in dependency order and composed
)
```

不是以下任一条件：

- 两个未符号化样例恰好输出相同；
- SMT 直接比较最终浮点表达式；
- 字符串形式相同；
- LLM 声称相同；
- load 在进入 e-graph 前被强制重命名为同一叶子。

安全/地址反例可产生 `DISPROVED`。规则不足且没有可信反例时为 `UNKNOWN`。任何必要
义务为 `UNKNOWN` 都不能被 e-graph 的局部等式覆盖。

## 报告关键字段

```text
inputs.frontends
proof.parametric_domain.checks
proof.parametric_domain.rewrite_targets
proof.parametric_domain.fact_derived_rewrites
proof.partitioning
proof.rule_admission
proof.egraph.initial_state
proof.egraph.after_fact_rewrites
proof.egraph.stats.iteration_trace
proof.egraph.rule_application
proof.egraph.unverified_rule_uses
```

`report.json` 是确定性的，不记录墙上时钟时间。`report.md` 汇总 proof blocks、规则
和信任边界。

## 当前限制

- 只量化固定 rank 的维度值，不支持动态 rank；
- 单 store、单可观察 Output；
- 子图划分仅支持提升后的无副作用纯计算表达式树，不切分控制流、内存副作用或多 store；
- 不支持循环、归约、多 kernel、原子和 shared memory；
- Prims 仅支持文档列出的逐点子集和显式广播；
- `ABSTRACT_FLOAT` 不是 IEEE-754；
- 暂无独立 proof certificate；
- 未验证规则可在显式策略下使用，但会降低证明可信度并被报告。

Add 的逐条规则及每轮图变化见
[参数化 Add 验证全过程](add_parametric_verification_details.md)。
