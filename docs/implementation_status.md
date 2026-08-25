# 实现状态

## 已完成路径

| 用例 | 结果 |
| --- | --- |
| 九齿 Add raw TTIR vs Torch Add Prims，固定 `[8,16]` singleton 参数域 | `PROVED`；使用符号关系规则，不走 TorchInductor |
| 九齿形态二维 `[a,b]` TTIR vs Torch 二维 Prims，`a*b=c` | `PROVED`；任意允许的正 i32 维度 |
| Semantic IR 二维 Add vs Prims Add | `PROVED`；无需 libtriton 的相同重写路径回归 |
| mul+add vs FMA | `PROVED`；应用经 Z3 验证的 `fma_def` |
| 错误 stride/mask/compute | `DISPROVED` 并给出反例 |
| 缺少 no-alias 或符号关系 | `UNKNOWN`/非 `PROVED` |
| 含 `scf.for` 的 TTIR | parse/verify 成功，提升返回 `UNKNOWN` |

当前测试共 58 项。无 libtriton 环境为 52 项通过、6 项 raw TTIR 测试跳过；安装
Triton/libtriton 3.7.1 后运行全部 58 项。

## 组件矩阵

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| PairSpec/结果 schema | 已完成 | 严格字段、角色、事实、参数域、契约和规则策略 |
| raw TTIR 前端 | 已完成基础路径 | libtriton 3.7.1 parse/verify/快照与无环逐点提升 |
| Torch Prims 前端 | 已完成基础路径 | 严格 `etv-prims-program-v1`，固定 rank、显式广播、核心逐点 op |
| TorchInductor 路径 | 已移除 | Torch 侧不再生成 Triton/TTIR |
| 固定 rank 符号 shape | 已完成基础路径 | 维度值与 launch 规模符号化，不支持动态 rank |
| 参数/shape 关系 | 已完成 | Z3 Int 参数域和 `a*b=c` 一类机器可读约束 |
| 覆盖、唯一写入、地址、竞争 | 已完成单 store 路径 | 参数化 SMT 反例查询 |
| 未归一化 load/store 根 | 已完成 | 保留 side、物理端点、offset、mask、value |
| fact-derived 关系规则 | 已完成 | scalar/load/store 条件规则，规则准入与应用分离 |
| e-graph | 已完成 | egglog 13.2.0、整图或逐子图独立饱和、同余及逐轮日志 |
| 代数规则 | 已完成当前集合 | 17 条内建规则默认经 Z3 Real 证明 |
| 自定义条件规则 | 已完成 | required/best_effort/trusted 策略与信任警告 |
| LLM 辅助 | 可选路径已完成 | DeepSeek 节点选择、带 fact gate 规则生成、严格 schema/provenance |
| 成对子图划分 | 已完成纯计算基础路径 | 一次整程序 LLM 扫描、根覆盖/路径/依赖 DAG 检查、逐子图 egglog 与组合证明 |
| 具体反例 | 已完成当前算术子集 | 地址、mask 与精确有理数值反例 |
| 报告 | 已完成 | schema v4，输入哈希、条件证明、规则应用、迭代轨迹、信任边界 |
| 循环/归约 | 未实现 | `scf.for`、`tt.reduce` 不提升 |
| 控制流/副作用子程序划分 | 未实现 | 当前只切分已提升的纯浮点表达式树，不切分 store、循环或 shared memory |
| 多 program/kernel orchestration | 未实现 | 单 kernel、单 store |
| IEEE-754/容差 | 未实现 | 当前为 `ABSTRACT_FLOAT` |
| 独立 proof certificate | 未实现 | egglog 与 ETV 编码仍在可信计算基 |

## 当前判定边界

对 TTIR/Prims 异构输入，ETV 要求 `facts.parameters`。即使固定 shape 也使用 singleton
参数域。验证成功必须同时满足：

```text
参数/launch/覆盖/地址条件已证明
and fact-derived load/store 规则实际匹配
and （完整 observe_store 根进入同一 e-class
     or 所有依赖有序子图根分别进入同一 e-class 并完成组合）
```

SMT 不直接比较最终抽象浮点程序；物理 load 也不会在进入 e-graph 前被重命名成同一
read。`initial_state`、`rule_application` 和 `after_fact_rewrites` 共同防止这种退化。

## LLM 与规则生成

现有 LLM 路径在确定性规则失败后运行：先选择节点，再生成必须依赖已有 fact 的规则。
模型不能决定结论。候选规则可能由 Z3 证明，也可在 PairSpec 明确选择的弱化策略下
作为可信公理；后者若实际使用，会进入最终报告警告。

尚缺：候选规则隔离仓库、自动边界条件综合、跨程序对规则晋升、独立验证证书。

## 子图划分

PairSpec 可通过 `partition.enabled` 启用划分，同时必须设置 `llm.enabled`。模型一次扫描
左右完整程序与全部计算根，返回成对子图路径；ETV 重新计算覆盖、唯一归属、左右依赖
拓扑与拓扑序。每个子图独立饱和，已证明的子图用共同边界 token 连接父图。模型的
语义标签和路径提议不是可信等价公理，所有局部对仍须由现有规则体系证明。

当前缺口是跨 store、控制流、归约、共享内存和多 kernel 的 region 划分；也尚未根据
e-node 预算自动决定是否启用、调整粒度或二次划分。

## 下一步

1. 扩展 Prims dtype/layout/view/reshape 语义，并让 stride 对应成为显式关系规则；
2. 将整数证明扩展为与 TTIR 溢出完全一致的位向量/混合整数模型；
3. 为 `div/sqrt/rsqrt` 增加机器可读定义域谓词；
4. 增加 e-graph explanation/proof certificate 导出；
5. 只有覆盖、归纳、单位元、归约顺序和地址义务齐备后才支持循环/归约；
6. 将划分从纯表达式树扩展到显式 region/内存 effect，并为跨分区等式建立接口摘要。
