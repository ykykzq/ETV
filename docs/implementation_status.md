# 实现状态

## MVP 验收结果

| 用例 | 预期 | 已实现结果 |
| --- | --- | --- |
| 二维连续 add 与线性 add | 证明等价 | `PROVED` |
| `ABSTRACT_FLOAT` 下 mul+add 与 FMA | 使用已准入的桥接规则证明 | `PROVED`，`fma_def` 已由 Z3 验证 |
| `output_stride0 = N + 1` | 地址反例 | `DISPROVED(OUTPUT_ADDRESS_MISMATCH)` |
| rhs 掩码为 `k < X - 1` | 掩码反例 | `DISPROVED(MASK_MISMATCH)` |
| rhs 的 add 改为 sub | 值反例 | `DISPROVED(COMPUTE_MISMATCH)` |
| 缺少 no-alias 声明 | 无法得出结论 | `UNKNOWN(MISSING_ALIAS_FACT)` |
| raw TTIR mul+add 与 raw TTIR FMA | libtriton 解析后证明 | `PROVED`，两侧分别解析为 25/24 个操作 |
| 含 `scf.for` 的 raw TTIR | 完整解析但不提升 | parser/verifier 通过；验证阶段为 `UNKNOWN(TTIR_REGION_SEMANTICS_UNSUPPORTED)` |

当前测试套件包含 36 项测试。完整的 Python 3.12 + libtriton + egglog 环境中 36 项全部通过；不安装可选 TTIR 依赖时，31 项通过、5 项真实 libtriton 测试按条件跳过。覆盖范围包括 schema 拒绝、有符号除法/取余、i32 溢出、基于性质的扁平化验证、egglog 统一饱和/同余/资源预算、Z3 规则准入/拒绝、fact gate 与可信规则审计、确定性报告、CLI 退出码、libtriton 快照稳定性、region 解析以及 raw TTIR 端到端证明。

## 组件矩阵

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| 严格的 PairSpec/结果契约 | 已完成 | JSON v1，拒绝未知字段 |
| 逻辑 ABI/角色对齐 | MVP 范围内已完成 | block、scalar、scalar-block |
| 事实/no-alias 上下文 | MVP 范围内已完成 | 固定绑定与成对不相交分组 |
| Semantic TTIR 类型化表达式 | 逐点子集已完成 | 整数、掩码、load、抽象计算、store |
| raw TTIR/MLIR 解析器 | 已完成 | Triton/libtriton 3.7.1，全方言注册、parse、verify、稳定快照 |
| raw TTIR 语义提升 | 逐点子集已完成 | 静态 tensor 标量化、广播、地址、load、计算、单 store；未建模操作返回 `UNKNOWN` |
| 有限启动域求值器 | 已完成 | 覆盖全部固定 program/lane，含 signed-i32 保护 |
| 掩码/地址/覆盖性证明 | 固定 specialization 范围内已完成 | 有界穷举 |
| 无竞争证明 | 单 store 范围内已完成 | 地址单射性 |
| 内存观察器 | 单 Output 范围内已完成 | 全映射 block；不支持原位更新、原子操作和边界安全证明 |
| 等式饱和 | 计算 MVP 范围内已完成 | egglog 13.2.0、统一 ruleset、同余闭包、迭代/节点/时间预算 |
| 代数规则准入 | 当前规则已完成 | Z3 实数 UNSAT 检查、查询哈希；SAT/UNKNOWN 拒绝 |
| Fact 规则准入 | 已完成但属于信任缺口 | PairSpec gate 决定启用；等式未经验证，报告为 `admitted_unverified` |
| 统一规则运行 | 已完成 | 不再按操作族筛选；全部已准入规则参与同一次饱和 |
| 具体反例搜索 | 有理数算术子集已完成 | 确定性、精确值 |
| 浮点定义域 | 精确常量定义域已完成 | 符号定义域返回 `UNKNOWN` |
| JSON/Markdown 报告 | 已完成 | 确定性、输入哈希与信任边界 |
| `scf.for` / `tt.reduce` 摘要 | 未实现 | M3 |
| IEEE/容差浮点证明 | 未实现 | 明确位于当前语义范围之外 |
| 多 kernel/外部摘要 | 未实现 | 后续工作 |
| LLM 规则 provenance | 输入与审计已完成 | 可记录 generator/prompt hash；自动生成和模型调用未实现 |

## 与压缩包里程碑的关系

### M0：前端与契约

MVP 范围内已完成。PairSpec、结果 schema、Semantic TTIR JSON、锁定的 libtriton 解析器/verifier、类型化快照、逐点提升器、`parse`/`inspect`/`check` 命令以及 raw TTIR golden/端到端测试均已实现。

### M1：add 端到端验证

已从 raw TTIR 边界端到端完成。索引、掩码、地址、标量 ABI、load、计算、store、Z3 规则准入、负向变异、报告和测试均已可用。

### M2：逐点算子扩展与候选生成

部分完成。表达式语言和 egglog 支持核心逐点操作；所有已准入规则在统一图中运行，PairSpec 可声明 SMT 代数规则或 fact-gated 可信规则。运行时值采样器、共享前沿规则综合器以及更大的真实 TTIR 语料尚未实现。

### M3：循环与归一化

未实现。层归一化/RMS 归一化报告仍作为设计与验收输入。仅添加一个 `theta` 节点而不完成覆盖性和归纳证明义务，不视为完成该里程碑。

### M4：自动规则生成

类型化规则 DSL、LLM provenance、SMT 准入和 fact gate 已存在，但没有自动模型调用或候选生成器。LLM 产生的 `algebraic` 规则不能绕过 Z3；非代数候选只能作为带明确警告的 `trusted_fact` 规则进入 PairSpec。尚缺隔离候选存储、边界条件综合和回归晋升策略。

## 后续工程步骤

1. 把压缩包所描述但未附带的 ntops/Inductor 完整 add TTIR 纳入语料，并生成稳定快照与 Semantic IR golden。
2. 为非线性 launch/store 域增加显式逻辑 frontier 标注，移除当前 `pid * lanes + lane` 的逐点约定。
3. 使用并行的 Z3 位向量证明义务替代具体整数枚举，以支持参数化索引/掩码规则，同时保留有界回归测试。
4. 增加带机器可读谓词的条件规则，从 `sqrt/rsqrt/div` 的正值/非零定义域开始。
5. 只有同时实现覆盖性、唯一性、单位元、地址和归约顺序证明义务后，才为 `scf.for`、`tt.reduce` 引入 `Theta/Reduce/Tile` 摘要。
6. 为 egglog 等价结果增加可独立检查的 proof certificate 或解释导出，缩小当前可信计算基。
