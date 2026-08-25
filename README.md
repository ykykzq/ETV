# ETV

ETV（Equivalence for TTIR Verification）验证九齿生成的 TTIR kernel 与 Torch Prims
参考程序是否具有相同的可观察输出。两侧使用不同前端，但会被提升到同一个类型化
内部表示，再通过带条件的 e-graph 重写完成等价判定。

```text
ntops Add -> ninetoothed -> Triton -> raw TTIR --+
                                                 +-> ETV IR -> 成对子图 -> 条件规则 -> e-graph
PyTorch Add -> TorchRefsMode/make_fx -> Prims ---+
```

Torch 侧不经过 TorchInductor，也不生成 Triton kernel。用户输入仍是两个程序和一份
PairSpec：左侧 raw TTIR、右侧 `etv-prims-program-v1`、角色/shape/指针对应事实以及
验证契约。

ETV 严格返回以下结果之一：

- `PROVED`：在 PairSpec 的全部前提下，Output 可观察内存等价；
- `DISPROVED`：找到了可重放的地址、掩码或抽象值反例；
- `UNKNOWN`：缺少事实、语义、规则或资源，无法闭合证明。

## 参数化范围

当前支持固定 rank、符号维度和符号 launch 规模，不支持动态 rank。例如在
`a>0`、`b>0`、`c>0`、`a*b=c` 的前提下，可验证二维 `[a,b]` 九齿 Add 与二维
Torch Prims Add 对任意允许的 `a,b,c` 等价。

参数化路径不会先把 load 直接归一化成同一个叶子。进入 e-graph 的初始根保留：

- 两侧物理 block/标量端点；
- `pid/lane` 经逻辑输出 `k` 替换后的符号地址；
- load/store mask；
- 完整 `observe_store(address, mask, value)`。

ETV 从 PairSpec 角色映射、shape 关系和 launch 关系产生 load/scalar/store 条件规则。
Z3 只证明这些规则在声明参数域上的条件；最终只有规则在 egglog 中实际匹配、并使
两侧 store 根进入同一个 e-class，才能返回 `PROVED`。规则准入和规则应用分别记录。

## 快速开始

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

安装 Triton/libtriton 3.7.1 后运行参数化真实输入边界：

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

预期输出：

```text
PROVED parametric_ninetoothed_add_vs_torch_prims_add: OBSERVABLE_MEMORY_EQUIVALENT
```

固定 `[8,16]` 上游样例：

```bash
.venv/bin/python -m etv check examples/add/pair.json \
  --out build/add_upstream
```

不安装 libtriton 也可运行等价的内部 IR/Prims 参数化回归：

```bash
.venv/bin/python -m etv check \
  tests/fixtures/semantic/specs/add_parametric_prims.json \
  --out build/add_parametric_prims
```

检查任一输入：

```bash
.venv/bin/python -m etv inspect examples/add/prims/torch_add.prims.json
.venv/bin/python -m etv parse examples/add/ttir/ntops_add.ttir \
  --out build/ntops_add.snapshot.json
```

Linux 可通过 `.[dev,ttir]` 安装 raw TTIR 前端；macOS 的源码构建步骤见
[依赖与环境](docs/dependencies.md)。

## 规则与 LLM

内建代数规则默认先由 Z3 Real 证明。PairSpec 自定义规则支持 `required`、
`best_effort`、`trusted` 三种准入策略。参数化 load/store 规则由 ETV 根据 PairSpec
自动生成，并附带 mask、地址和角色条件的证明记录。

常规规则无法连接待验证根时，可以显式启用 DeepSeek：模型先选择表达式节点，再
生成必须引用现有 fact gate 的候选规则。模型输出经过严格 schema 和既有准入流程；
模型本身不能给出 `PROVED`。凭据只从 `DEEPSEEK_API_KEY` 读取。

对于较大的纯计算图，也可以在 PairSpec 同时设置 `llm.enabled=true` 和
`partition.enabled=true`。ETV 向 DeepSeek 发送一次左右完整程序、PairSpec 上下文和
全部计算根族；模型只提出左右子图根。ETV 随后检查整根覆盖、路径唯一性、左右依赖
拓扑、无环性和类型，并按子图依赖顺序分别运行 egglog。已经证明的子图才会在父图
中替换为同名类型化边界输入。提案非法或局部证明失败时自动回退到原整图验证，且把
原因写入 `proof.partitioning`。

```json
{
  "llm": {"enabled": true, "generate_rules": false},
  "partition": {"enabled": true, "min_partitions": 2, "max_partitions": 16}
}
```

## 语义边界

当前 `ABSTRACT_FLOAT` 将浮点操作解释为数学实数。因此 `PROVED` 不表示 IEEE-754、
容差或 GPU 位级等价。当前还不支持动态 rank、循环、归约、多 kernel、原子操作、
shared memory 或 buffer 分配边界证明。

libtriton、Prims 提升器、ETV 内存模型、Z3、egglog 与 PairSpec 前提都位于当前可信
计算基。实际使用的未经验证规则会进入报告的信任警告。

## 命令

```text
etv check SPEC [--out DIR] [--json]
etv inspect PROGRAM
etv parse INPUT [--function NAME] [--out FILE] [--no-assembly]
etv rules [--json]
etv validate-rule [RULE_ID]
etv explain REPORT_JSON
```

`check` 退出码为 `PROVED=0`、`DISPROVED=1`、`UNKNOWN/输入错误=2`。

## 仓库结构

```text
etv/prims.py                   严格 Prims JSON 读取与内部 IR 提升
etv/ttir/                     libtriton 解析、快照与 TTIR 提升
etv/parametric.py             符号义务与 fact-derived 条件规则
etv/egraph.py                 egglog 适配、逐轮状态和规则应用日志
etv/llm.py                    可选节点选择与条件规则候选
etv/partition.py              整程序 LLM 扫描、成对子图检查与依赖 DAG
tools/extract_add_pair.py     九齿 TTIR/Torch Prims 输入提取
examples/add/                 Add 程序对、PairSpec 与 provenance
docs/prims_format.md          Prims 输入格式
docs/verification_process.md  完整验证契约
docs/add_parametric_verification_details.md 参数化 Add 逐步重写轨迹
```

详细设计见[架构](docs/architecture.md)，当前覆盖与缺口见
[实现状态](docs/implementation_status.md)。
