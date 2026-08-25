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

当前测试套件包含 23 项测试，覆盖 schema 拒绝、有符号除法/取余、i32 溢出、基于性质的扁平化验证、e-graph 重建、Z3 规则准入、确定性报告、CLI 退出码以及上述全部用例。

## 组件矩阵

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| 严格的 PairSpec/结果契约 | 已完成 | JSON v1，拒绝未知字段 |
| 逻辑 ABI/角色对齐 | MVP 范围内已完成 | block、scalar、scalar-block |
| 事实/no-alias 上下文 | MVP 范围内已完成 | 固定绑定与成对不相交分组 |
| Semantic TTIR 类型化表达式 | 逐点子集已完成 | 整数、掩码、load、抽象计算、store |
| raw TTIR/MLIR 解析器 | 未实现 | 必须在 Linux/CUDA 上使用已注册 Triton 方言的解析器 |
| 有限启动域求值器 | 已完成 | 覆盖全部固定 program/lane，含 signed-i32 保护 |
| 掩码/地址/覆盖性证明 | 固定 specialization 范围内已完成 | 有界穷举 |
| 无竞争证明 | 单 store 范围内已完成 | 地址单射性 |
| 内存观察器 | 单 Output 范围内已完成 | 全映射 block；不支持原位更新、原子操作和边界安全证明 |
| 等式饱和 | 计算 MVP 范围内已完成 | 哈希驻留、并查集、重建、预算、账本 |
| 规则准入 | 当前规则已完成 | Z3 实数 UNSAT 检查和查询哈希 |
| 候选驱动规则 | 操作族层面已完成 | 只有未匹配的计算切片才会激活规则 |
| 具体反例搜索 | 有理数算术子集已完成 | 确定性、精确值 |
| 浮点定义域 | 精确常量定义域已完成 | 符号定义域返回 `UNKNOWN` |
| JSON/Markdown 报告 | 已完成 | 确定性、输入哈希与信任边界 |
| `scf.for` / `tt.reduce` 摘要 | 未实现 | M3 |
| IEEE/容差浮点证明 | 未实现 | 明确位于当前语义范围之外 |
| 多 kernel/外部摘要 | 未实现 | 后续工作 |
| LLM 规则提议 | 未实现 | 只有建立隔离工作流后才允许提议 |

## 与压缩包里程碑的关系

### M0：前端与契约

部分完成。PairSpec、结果 schema、Semantic TTIR JSON、校验和检查命令已经存在；raw `libtriton` 解析器/提升器和 golden TTIR 测试尚未实现。

### M1：add 端到端验证

在 Semantic TTIR 边界内已完成。索引、掩码、地址、标量 ABI、load、计算、store、Z3 规则准入、负向变异、报告和测试均已可用。

### M2：逐点算子扩展与候选生成

部分完成。表达式语言和 e-graph 支持核心逐点操作，规则激活由候选驱动。运行时值采样器、共享前沿规则综合器以及更大的真实 TTIR 语料尚未实现。

### M3：循环与归一化

未实现。层归一化/RMS 归一化报告仍作为设计与验收输入。仅添加一个 `theta` 节点而不完成覆盖性和归纳证明义务，不视为完成该里程碑。

### M4：自动规则生成

目前只有准入门禁。任何生成规则都不能直接进入注册表。未来在加入 LLM 规则提议器前，需要类型化规则 DSL、隔离存储、边界条件综合、求解器证书和回归晋升策略。

## 后续工程步骤

1. 构建并锁定带匹配 Triton 版本的 Linux/CUDA 开发容器；实现已注册方言的 MLIR 适配器和源码位置映射。
2. 提升压缩包中的真实 add TTIR 对，并要求生成字节稳定的 golden Semantic IR，然后才能把提升过程移出信任边界。
3. 使用并行的 Z3 位向量证明义务替代具体整数枚举，以支持参数化索引/掩码规则，同时保留有界回归测试。
4. 增加带机器可读谓词的条件规则，从 `sqrt/rsqrt/div` 的正值/非零定义域开始。
5. 只有同时实现覆盖性、唯一性、单位元、地址和归约顺序证明义务后，才引入 `Theta/Reduce/Tile`。
6. 当关系型循环/定义域事实超出本地 e-graph 的有限职责时，重新评估 egglog。
