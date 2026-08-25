# 架构

## 总体数据流

```text
九齿侧                                      Torch 侧
ntops -> ninetoothed -> Triton -> raw TTIR   PyTorch -> TorchRefsMode -> Prims JSON
                         |                                  |
                         v                                  v
                libtriton parse/verify              strict Prims schema
                         |                                  |
                         +--------> 共同 ETV IR <------------+
                                      |
                           固定 rank 参数与事实环境
                                      |
                  launch/index/mask/address 条件证明（SMT）
                                      |
                    生成带证明条件的 load/store 关系规则
                                      |
                    可选：LLM 提议成对子图根（一次整程序扫描）
                                      |
                  ETV 检查覆盖/归属/依赖 DAG/类型并逐子图验证
                                      |
                  fact rewrites -> algebraic rewrites -> LLM rule fallback
                                      |
                         根同 e-class 才能判定等价
                                      |
                       PROVED / DISPROVED / UNKNOWN
```

两个前端不要求源 IR 相同，也不把 Torch 参考程序伪装成 GPU kernel。Prims 图描述纯
张量函数；提升器将其嵌入逻辑逐元素域。TTIR 则保留真实 `program_id/lane`、mask、
pointer offset 和 store。二者通过 PairSpec 声明的角色、shape、指针和 launch 事实
建立关系。

## 输入边界

### 九齿侧

`tools/extract_add_pair.py` 仍使用原有链路：固定 ntops/ninetoothed 提交生成 Triton
源码，再由 Triton 3.7.1 AST 前端和 TTIR pass 生成 raw TTIR。验证时
`etv/ttir/libtriton.py` 注册完整方言、解析、运行 IR verifier，并生成稳定快照；
`etv/ttir/lift.py` 只提升已建模的无环逐点子集。

TTIR 不编码 host launch grid，因此 `frontends.lhs.programs` 必须由 PairSpec 给出，
并且可以是 shape 参数的表达式，例如 `ceildiv(c,256)`。

### Torch 侧

Torch 侧不再经过 TorchInductor。提取工具在 `TorchRefsMode` 中用 `make_fx` 得到 Prims
图，并写为 `etv-prims-program-v1`。`etv/prims.py` 严格检查固定 rank shape、显式
`broadcast_in_dim`、SSA 引用、arity 和支持的逐点 op，再直接提升到共同 IR。

Prims 图没有 GPU launch。内部 `programs=product(output_shape), lanes=1` 仅表示逻辑
张量观察域，不能解释为 Torch 的实际 kernel 配置。格式见[Prims 输入](prims_format.md)。

### PairSpec

`etv/schema.py` 读取 `etv-pair-v1`，拒绝未知字段、重复物理角色、未绑定符号、非法
参数域和不完整规则。关键事实包括：

- `parameters`：固定 rank 各维值和 launch 规模的量化域；
- `constraints`：如 `a*b=c`；
- `side_bindings`：TTIR 参数、Prims shape 槽位到共同参数的映射；
- `roles`：物理 block/scalar/scalar-block 到逻辑 Input/Other/Alpha/Output；
- `disjoint`：no-alias 前提；
- `contract`：可观察输出和逻辑元素数。

raw TTIR 与 Prims 的异构验证必须声明 `facts.parameters`。固定 `[8,16]` 示例也使用
`a=[8,8]`、`b=[16,16]`、`c=[128,128]` 的 singleton 参数域，避免退回非符号化
的有限枚举来判定这类程序对。

## 共同内部表示

`etv/model.py` 定义不可变的 `Expr/Program/StoreTemplate`。前端差异保留到重写之前：

```text
TTIR load:  load(lhs:arg0, complex_offset(k,b), k<a*b, 0)
Prims load: load(rhs:input, row_major_offset(k,a,b), true, 0)
```

物理端点带 side 标签，不能因名称偶合而 hash-cons 到同一节点。输出根也不是单独的
浮点值，而是：

```text
observe_store(side:physical_output, offset(k), mask(k), value(k))
```

因此地址、mask、load、标量 ABI 和计算任一层未建立关系，最终 store 根都不会相等。

## 参数符号化

`etv/parametric.py` 保留固定 rank 的维度值、TTIR program 数、`pid/lane` 和任意逻辑
输出 `k`。它用 canonical writer：

```text
pid  = k div lanes
lane = k rem lanes
0 <= k < output_numel
```

Z3 证明参数域非空、整数定义性、launch 覆盖、mask、唯一写入、地址单射和对应地址
关系。这些结果是条件规则的准入证据，不会直接把两侧表达式替换成相同叶子。

## 条件规则

参数化路径自动生成三类关系规则：

1. scalar role：`input(lhs:arg2) -> input(Alpha)`；
2. load：`load(side:physical, offset, mask, default) -> read(LogicalRole,k)`；
3. store：`observe_store(side:physical_output,offset,mask,v) -> observe_store(Output,k,true,v)`。

每条 load/store 规则的报告包含：

- PairSpec 物理端点到逻辑角色的映射；
- `0<=k<output_numel` 与全部 shape/launch 约束；
- mask 恒真的反例查询及其 `UNSAT` 结果；
- 地址等于 canonical logical address 的反例查询及其 `UNSAT` 结果；
- query hash、Z3 版本和规则 statement。

规则的“条件成立”和“实际应用”严格分开。前者进入 `rule_admission`，后者由 egglog
匹配次数进入 `rule_application`。只有后者导致的 e-class 合并可关闭等价目标。

## e-graph 阶段

`etv/egraph.py` 使用 `egglog==13.2.0`。执行顺序为：

1. 插入当前义务的两侧根（整图或一个子图）并记录 `initial_state`；
2. 运行 fact-derived relational rewrites；
3. 若仍未合并，运行经 Z3 准入的代数规则和 PairSpec 自定义规则；
4. 若仍未合并且显式启用 LLM，选择未匹配子节点并生成带 fact gate 的候选规则；
5. 重新准入并饱和；根仍不相等则返回反例或 `UNKNOWN`。

每轮记录前后 e-node/e-class 数、逐规则 match count 和 `updated`。e-graph 不删除旧
表示；规则把右侧表示加入相同 e-class，并由同余闭包向父节点传播。

## 成对子图划分

`etv/partition.py` 实现可选的“大图提议、小图证明”路径。启用条件为 PairSpec 同时
声明 `llm.enabled=true` 与 `partition.enabled=true`。划分器只调用一次
`program_partitioning` API，请求包含：

- 左右完整 `Program` 的 launch、全部 store 字段和表达式；
- PairSpec 角色、binding、约束、assumption 与观察契约；
- 验证阶段全部计算根，按忽略叶子具体值的左右结构分成 root family；
- 每个 family 的代表表达式及所有可选节点路径。

LLM 返回 `id/family/semantic/lhs_path/rhs_path`，只负责提出语义边界。ETV 不接受
模型给出的覆盖结论、依赖边或等价结论，而是确定性地检查：

1. 每个 family 恰有一个 `root/root` 分区，因此整棵计算树不会漏掉；
2. 同侧路径不重复，非根分区不能选择叶子；
3. 左右对应分区的 sort 均为 `abstract_float`；
4. 根据路径包含关系计算最近父分区，左右父 ID 必须一致；
5. 依赖图必须无环，分区总数满足 PairSpec 限制。

路径树使同一程序中的分区只能嵌套或互不相交。每个父分区只保留自身独占节点，直接
子分区替换为左右同名的 `input(partition:...)`。验证按子到父的拓扑顺序进行；只有
子分区已证明等价，父分区才能使用该共同边界。每个 `SubgraphBatch` 建立独立
e-graph，从而限制单次饱和看到的图规模。所有根分区都证明后，`DECOMPOSITION` 与
同余性完成组合证明。

无效提案、API 错误或任一局部义务未证明时，ETV 不使用划分结论，而是记录
`SUBGRAPH_PARTITION` 非必要块并回退到原来的单图验证。局部抽象反例不会直接升级为
全程序 `DISPROVED`，因为边界输入值未必在原程序中可达。

## LLM 边界

LLM 有两条相互独立的可选路径：`etv/partition.py` 在计算证明前做一次整程序划分
提议；`etv/llm.py` 在确定性规则失败后选择节点并提出候选规则。模型不能修改输入
事实、绕过结构检查、直接 union 根或决定最终状态。代数候选仍按策略交给 Z3；
非代数候选若被信任，实际使用会作为 `TRUSTED_AXIOM` 和健全性警告写入报告。

## 结果与信任边界

`PROVED` 要求所有必要安全/覆盖义务关闭，并且全部 store 根经规则应用后属于同一
e-class；启用划分时则要求每个依赖有序子图根进入对应 e-class 并完成组合。报告
schema v4 包含输入哈希、前端版本、参数查询、划分审计、规则准入、规则应用、逐轮
e-graph 状态和未验证规则警告。

当前可信计算基包括 PairSpec 前提、libtriton、两个提升器、符号整数编码、Z3、
egglog term 编码/同余维护和内存观察模型。尚无独立 proof certificate。
