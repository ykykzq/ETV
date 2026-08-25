# 架构

## 验证流水线

```text
PairSpec + raw TTIR 或 Semantic TTIR A/B
          |
          v
libtriton 3.7.1 解析/验证（raw TTIR）
          |
          v
类型化 TTIR 快照 + 受控语义提升
          |
          v
严格的 schema 与角色/事实校验
          |
          v
有限启动域枚举或固定秩参数化 SMT 证明
          |
          +--> 索引 / 掩码 / 覆盖性 / 竞争 / 地址证明义务
          |
          v
逻辑读入/标量叶节点 + 抽象计算表达式
          |
          v
egglog 统一 e-graph 等式饱和
          |
          +--> 经 Z3 证明或策略显式信任的条件规则
          +--> 可选 DeepSeek 节点选择与规则生成
          +--> egglog 同余闭包与具名规则日志
          |
          v
可观察内存比较
          |
          v
PROVED / DISPROVED / UNKNOWN + JSON/Markdown 报告
```

实现遵循压缩包设计中的 G0-G8 分解，但不会强制把每个诊断前沿都当作必要的证明前提。例如，两个程序可能读取不同的值，但这些值会在代数运算中抵消。因此，LOAD 前沿不匹配只作为诊断信息；最终等价性仍由值根节点和内存根节点决定。

## 组件

### `tools/extract_add_pair.py`

该工具位于验证器输入边界之前，负责从固定提交的 ntops Add 与 ninetoothed 生成
左侧 Triton 源码，并通过 PyTorch FakeTensor/FX/TorchInductor 生成右侧 Triton
源码，再调用 Triton 3.7.1 的 AST 前端和 TTIR pass 得到两侧 raw TTIR。工具检查
上游 Git HEAD、关键源码哈希和包版本，并生成 `provenance.json`。它不执行 GPU
kernel；TorchInductor launch decorator 的离线适配和 debug location 规范化都被
显式记录。详见[真实 Add 验证](add_validation.md)。

### `etv/schema.py`

读取 `etv-semantic-program-v1` 和 `etv-pair-v1`。未知字段、错误元数、缺失的角色端点、重复物理映射和无效资源限制都会被拒绝。变量绑定在成对求值阶段解析。这样可避免因拼写错误而在无提示的情况下证明一个更弱的契约。

### `etv/model.py`

定义不可变的类型化表达式、程序、角色端点、事实上下文、契约、证据等级以及三种结果状态。表达式不依赖 SSA 名称和源码位置。

### `etv/ttir/libtriton.py` 与 `etv/ttir/model.py`

`libtriton.py` 是锁定到 Triton 3.7.1 的唯一 raw TTIR parser 适配器。它创建 MLIR context、注册 Triton 的全部内置方言、解析文件并运行 IR verifier。随后把模块转换成 `etv-ttir-snapshot-v1`：稳定编号的 SSA 值、操作、类型、block/region、已知结构属性、源码位置以及完整规范汇编。规范汇编保留所有方言属性，因此快照不会把 Python 绑定未公开枚举接口的属性丢掉。

### `etv/ttir/lift.py` 与 `etv/ttir/frontend.py`

`frontend.py` 根据扩展名或 PairSpec 的显式 `kind` 在 Semantic JSON 和 raw TTIR 之间分派。`lift.py` 将已验证 TTIR 的无环逐点子集标量化为 Semantic TTIR：静态 tensor shape 被解释为多维 lane 索引，广播按 shape 映射，pointer 表示为逻辑 block 加元素偏移。

解析器接受完整的 Triton 3.7.1 TTIR；提升器只接受具有明确语义规则的子集。合法但未建模的操作会产生 `UNKNOWN`，不会退化为文本猜测。详见 [完整验证过程](verification_process.md)。

### `etv/evaluator.py` 与 `etv/parametric.py`

固定实例模式枚举每一个 `(program_id, lane)`，并具体求值整数和布尔表达式。
参数化模式保留固定秩 shape 参数、`pid`、`lane` 和任意逻辑输出 `k`，使用 Z3
证明参数域非空、覆盖、唯一写入、地址单射和两侧地址相等。只有被 SMT 证明相等的
读取地址才归一化成同一个逻辑叶节点。两种模式中的浮点计算都保持符号形式。

有符号除法和取余遵循 MLIR `divsi/remsi` 的向零截断语义。MVP 要求所有具体整数中间值都位于有符号 i32 范围内；溢出、除零或过大的有限域会产生 `UNKNOWN`。

### `etv/egraph.py`

这是 `egglog==13.2.0` 的薄适配层。它把 ETV 表达式编码为包含操作、子节点、附加数据和 sort 的 egglog term，把具名规则转换成统一 ruleset，并逐轮收集 `RunReport`、e-node/e-class 规模、规则匹配次数和停止原因。hash-consing、e-matching、union-find 与同余重建均由 egglog 实现。

所有输出位置的计算根进入同一个 e-graph。只要存在尚未等价的根，所有已准入代数规则和所有 fact gate 已满足的可信规则都会进入同一次饱和；不再根据表达式中出现的操作筛选规则。当前也不划分子图，避免遗漏跨分区等式。

### `etv/rules.py` 与 `etv/z3_validator.py`

`rules.py` 包含有限且经过审计的抽象代数规则模式，以及 PairSpec fact gate 的数据模型。内建代数规则参与等式饱和前，`z3_validator.py` 会将其转换为 Z3 实数算术，并证明 `lhs != rhs` 不可满足。自定义规则默认采用同样的严格策略；SAT、UNKNOWN 或不受支持的操作都会导致规则被拒绝。

PairSpec 规则的“验证”和“应用”分别记录。默认策略仍要求 `algebraic` 规则通过
Z3；`best_effort` 会尝试证明但允许失败后以 `TRUSTED_AXIOM` 使用，`trusted` 则
直接跳过证明。`trusted_fact` 以及所有未经证明而实际匹配的规则都会在报告中列出
健全性警告。fact gate 还可精确引用机器可读的 shape constraint。

### `etv/llm.py`

可选 DeepSeek 辅助只在普通规则未能连接输出根后运行。第一步从未匹配表达式树中
选择候选节点，第二步生成必须引用现有 fact gate 的条件规则。所有响应经过严格
schema 解析，再按相同规则策略准入；模型不能删除最终输出根，也不能直接产生证明。
当前仍使用联合 e-graph，不划分子图。LLM 新规则会触发第二次统一饱和，但不会把
根节点拆到互不通信的分区。

### `etv/verify.py`

按照以下顺序协调证明义务：

1. 语义模式与 no-alias 前置条件；
2. 角色对齐；
3. 有限启动域枚举，或固定秩参数域上的 SMT 证明；
4. 逻辑掩码域相等与完整覆盖；
5. 输出地址单射性，即无写入竞争；
6. 对应 Output 地址相等；
7. 逻辑 load 前沿比较；
8. 浮点操作定义域的结构化验证；
9. e-graph 计算根节点等价；
10. 最终内存 token 相等。

地址或掩码差异会产生直接的有限域见证或参数模型。如果计算根节点没有合并，确定性的精确有理数模型搜索会尝试构造可重放的值反例。无法证明且找不到反例时，结果保持为 `UNKNOWN`，绝不会错误地返回 `DISPROVED`。

### `etv/reporting.py` 与 `etv/cli.py`

生成确定性的 JSON 和易读的 Markdown。JSON 包含输入哈希、假设、证明块结果、全部规则的准入状态、fact gate 结果、egglog 版本、资源限制、规则匹配日志和可信规则警告。证明产物中不包含墙上时钟时间戳或耗时。

CLI 还提供 `parse` 命令生成完整 TTIR 快照，并让 `inspect` 同时支持 raw TTIR 与 Semantic JSON。

## 为什么使用 egglog

工程已经从本地 e-graph 迁移到 egglog。选择 Python egglog 而不是 Rust egg，是为了直接复用现有 Python 类型化前端，同时获得成熟的 e-matching、同余闭包、统一 ruleset 和逐轮运行统计。版本精确锁定为 13.2.0，避免 Python DSL 和底层绑定变化造成行为漂移。

当前没有声称获得独立 proof certificate：规则准入和匹配日志可审计，但 egglog 的等价维护仍属于可信计算基。LLM 生成结果经过相同准入流程；当前没有子图划分，未来如引入分区也不能阻断本应共享的等式。

## 研究基础与设计决策

### 程序验证与翻译验证

程序验证在显式前置条件覆盖的所有状态上建立程序与规格之间的数学关系。Hoare 的公理化方法使用前置条件、程序语句和后置条件表达这一关系；循环还需要不变式，完全正确性还需要终止性论证。参见 [Hoare, 1969](https://doi.org/10.1145/363235.363259)。

ETV 处理的是关系型规格：

```text
P(lhs_state, rhs_state)
  => Defined(lhs) and Defined(rhs)
  and ObservableMemory(Exec(lhs)) = ObservableMemory(Exec(rhs))
```

一般程序等价性不可判定，因此工程采用翻译验证：只验证一次编译或优化产生的具体程序对，而不证明编译器对所有程序正确。该思路源自 [Pnueli 等](https://link.springer.com/chapter/10.1007/BFb0054170)。[Alive2](https://web.ist.utl.pt/nuno.lopes/pubs.php?id=alive2-pldi21) 展示了在编译器 IR 上结合符号执行、显式内存语义、SMT、有界循环与具体反例的实用路径。由此，ETV 只在证明义务闭合时返回 `PROVED`，发现见证时返回 `DISPROVED`，缺少语义、事实或预算时保持 `UNKNOWN`。

### 等式饱和

经典等式饱和持续加入已知等式并维护共享表示，不破坏性地选择单一重写顺序，参见 [Tate 等，2009](https://cseweb.ucsd.edu/~lerner/papers/popl09.html)。[egg](https://arxiv.org/abs/2004.03082) 通过摊销重建与 e-class 分析实现实用的同余维护，[egglog](https://arxiv.org/abs/2304.04332) 进一步结合了 Datalog 风格的事实和关系。

e-graph 只负责维护等价关系，不能让错误规则变正确。可信合并必须来自语法 hash-consing、同余、已经验证的等式或显式可信公理。ETV 因而在规则进入 egglog 之前进行 Z3 或 fact gate 准入，并将证据等级和未验证风险写入报告。

### 项目参考资料的影响

| 资料 | 对当前实现的主要影响 |
| --- | --- |
| [Gimlet Labs：验证 AI 生成的 GPU Kernel](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels) | 从可观察 store 反向理解计算；有限化输出索引但保持值符号化；明确区分精确实数与 GPU 浮点语义 |
| [Zhan 等，2024](https://arxiv.org/abs/2412.20992) | 使用类型化证明 IR、显式 PairSpec 标注和 SMT 准入；掩码、地址、内存保持为一等证明对象 |
| [Emerge，Zhan 等，2026](https://arxiv.org/abs/2603.21851) | 联合 e-graph、共享逻辑前沿、规则生成与分级验证、失败定位；当前实现采用统一 ruleset，不进行可能遗漏跨分区等式的子图划分 |
| [VOLTA](https://arxiv.org/abs/2511.12638) | 明确限制支持的 GPU 程序类别，并将无写入竞争作为独立证明义务 |

用户提供的工程资料还覆盖 PairSpec、G0-G8 证明义务、Add 的逻辑角色与 ABI 对齐、TTIR 环境、循环和归约的未来义务，以及 e-graph 可行性。实现冲突按以下原则处理：

1. 浮点语义采用较晚系统设计中的 `ABSTRACT_FLOAT`，不沿用早期材料中的整数或位向量近似；
2. raw TTIR 由 Triton 3.7.1 的已注册方言 parser/verifier 接受，正则表达式不承担语法和合法性判断；
3. 工程从本地 e-graph 迁移到 `egglog==13.2.0`，换取成熟的 e-matching、同余闭包与统一 ruleset，但 egglog 和 term 编码进入可信计算基；
4. 默认要求代数候选由 Z3 证明；显式弱化策略允许将未证明规则作为可信公理应用，但必须公开其健全性风险；
5. `theta` 节点本身不能证明循环。覆盖性、唯一性、单位元、地址和归纳/归约顺序义务完成前，循环与归约保持 `UNKNOWN`。
