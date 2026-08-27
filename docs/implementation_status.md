# 实现状态

## 已完成

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| 正式输入边界 | 已完成 | 两侧只接受 raw `.ttir`/`.mlir`，必须显式入口与 launch |
| libtriton 前端 | 已完成基础路径 | 固定 3.7.1，parse、verify、完整遍历和稳定快照 |
| 共同内部 IR | 已完成 | 独立 `etv/ir.py`，与 PairSpec model、e-graph 分离 |
| TT IR 语义提升 | 已完成逐点子集 | program/lane、指针、mask、可选单 store 观察、显式整数 cast、整数和浮点表达式 |
| PairSpec v2 | 已完成 | metadata、LLM assumptions、统一 predicates、observation 和外部 rewrites |
| 固定规模验证 | 已完成 | 有限 launch/lane 枚举、地址/mask/覆盖与抽象值反例 |
| 参数化验证 | 已完成基础路径 | 固定 rank 符号 shape、launch、`a*b=c` 等 SMT 关系 |
| 等式饱和 | 已完成 | egglog 13.2.0、谓词派生规则、代数规则、同余与应用日志 |
| rewrite registry | 已完成 | 独立用户规则文件、来源标记、SHA-256、单次运行隔离 |
| 规则准入 | 已完成当前集合 | 代数规则用 Z3；布局规则依赖 PairSpec predicate/SMT 条件 |
| cast 义务 | 已完成显式位宽路径 | cast operator/类型保留；差异必须命中 rewrite；`index` 位宽未知时返回 UNKNOWN |
| LLM 规则辅助 | 可选路径已完成 | 自然语言上下文、节点选择、predicate gate、严格 schema 和审计 |
| LLM 子图划分 | 可选路径已完成 | 整程序扫描、成对子图、依赖 DAG 检查和组合证明 |
| 报告 | 已完成 | schema v5、谓词/规则来源、soundness 分级、义务、e-graph、反例 |
| uv 环境 | 已完成 | `uv.lock`、互斥 extra、macOS libtriton 源码构建脚本 |
| ntops Torch 侧矩阵 | 已完成前端探针 | 76 模块各一个 specialization；67 个生成优化 TTIR |

## 当前样例

`examples/add/pair.json` 是固定 `[8,16]` 的真实程序对：

- 左侧：ntops Add -> ninetoothed -> Triton -> TT IR；
- 右侧：PyTorch 表达式 -> FakeTensor FX -> TorchInductor -> Triton -> TT IR；
- 左侧 256 lane，右侧 128 lane；
- 两侧经同一 libtriton 前端和同一内部 IR 后验证。

`examples/add/pair_parametric.json` 是参数化回归：二维 256-lane TT IR 对一维
128-lane TT IR，在正 i32 参数与 `a*b=c` 下证明等价。它不是声称固定规模
TorchInductor kernel 对任意 shape 通用，而是专门构造的符号 TT IR 前端/后端回归。

在 macOS arm64、Python 3.12.13、源码构建的 libtriton 3.7.1 环境中，正式固定例和
参数化例均返回 `PROVED(OBSERVABLE_MEMORY_EQUIVALENT)`，参数化报告包含 45 个 SMT
检查。当前完整环境中的 88 个 ETV 测试全部通过。

内部 JSON 夹具覆盖正确、错误 stride、错误 mask、错误计算、FMA、规则策略、LLM 和
子图划分。它们只测试证明核心，不属于生产前端。

## 未实现或有限支持

- 动态 rank；
- `scf.for`、`tt.reduce` 等循环和归约语义；
- 多 store 间的顺序 effect、原子操作、shared memory 和 barrier；单个 store/输出 leaf
  可由 PairSpec 的 `store_index` 独立观察；
- 多 kernel orchestration；
- 多维 program grid 的完整语义；
- buffer 分配大小与 GPU 内存安全；
- IEEE-754、快速数学、容差和位级等价；
- 独立 proof certificate；
- 自动证明 TT IR 到内部 IR 提升器本身正确。
- target data layout 驱动的 `index` 位宽与相应 cast 语义。

libtriton 能解析但提升器不支持的合法 TT IR 返回 `UNKNOWN`。这一区分保证“完整语法
前端”不被误解为“完整 Triton 语义验证器”。

## 信任缺口

代数重写默认要求 Z3 证明。依赖布局、ABI 或输入谓词的关系规则由 ETV 根据 PairSpec
生成并附带 SMT 条件。用户或 LLM 新增的非代数规则暂时允许在 predicate gate 满足后
作为可信规则；若它实际参与证明，报告会记录 `admitted_unverified`、规则来源和警告，
并把 `soundness.level` 降为 `conditional_on_unverified_rewrites`。错误谓词或错误可信
规则会破坏结论健全性，当前系统无法自动弥补。

`assumptions.for_llm` 明确不进入上述信任链；它只能影响启发式候选和划分建议。自定义
`z3_expr` 谓词可被 SMT 使用，但仍是输入前提，不允许通过 PairSpec 自行标为已证明。

## 后续优先级

1. 扩展 TT IR 提升语义到 view/layout、归约和结构化控制流；
2. 将整数模型扩展为与 TT IR 溢出属性及 target `index` 位宽一致的位向量/混合整数证明；
3. 为浮点定义域、IEEE-754 与快速数学建立可选语义；
4. 将子图划分从纯表达式树扩展到 region 和显式内存 effect；
5. 导出 egglog explanation 或可独立重放的证明证书；
6. 把当前 Torch 侧可提取性矩阵扩展为 ntops/Torch 双侧 TTIR、PairSpec 和端到端验证矩阵。

## 当前测试证据

在 macOS arm64、Python 3.12.13 的隔离环境中执行：

```text
88 passed
```

固定 ntops 提交包含 76 个算子测试模块和 2102 个参数化 CUDA 用例。本机无 CUDA，
这些上游数值测试全部跳过，不能计为通过。离线 TorchInductor 探针对每个模块测试一个
代表 specialization：67 个生成并编译为优化 TTIR，6 个默认走外部/ATen kernel，3 个
受 CPU-only PyTorch 的 FakeTensor trace 限制。该矩阵只检查 Torch 侧前端可提取性，
不等价于双侧形式验证。完整分类和复现命令见
[uv 环境与全算子测试](operator_testing.md)。
