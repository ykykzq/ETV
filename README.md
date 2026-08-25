# ETV

ETV（Equivalence for TTIR Verification）是面向固定 Triton kernel specialization
的最小可用翻译验证器。它使用锁定的 Triton/libtriton 3.7.1 读取原始 TTIR，
验证掩码、覆盖、地址、竞争、计算和最终可观察内存，并严格返回以下结果之一：

- `PROVED`：在 PairSpec 的全部前提下，Output 可观察内存等价；
- `DISPROVED`：找到了可重放的地址、掩码或抽象值反例；
- `UNKNOWN`：缺少事实、语义、规则或资源，不能得出前两种结论。

仓库中的用户示例不是手写的“理想 TTIR”。它来自两个实际编译链：

```text
InfiniTensor/ntops Add -> ninetoothed -> Triton -> TTIR
PyTorch 表达式 -> FakeTensor FX -> TorchInductor -> Triton -> TTIR
```

两侧均固定为 `[8, 16]`、`float32`、CUDA sm80 specialization。来源提交、源码
哈希、生成工具和每个 artifact 的 SHA-256 见
[provenance.json](examples/add/provenance.json)，完整复现实验见
[真实 Add 验证](docs/add_validation.md)。

## 快速开始

仅运行不需要 raw TTIR 的单元测试：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

安装 Triton/libtriton 3.7.1 后运行真实 Add 验证：

```bash
.venv/bin/python -m etv check examples/add/pair.json \
  --out build/add_upstream
```

预期输出：

```text
PROVED ntops_add_vs_torch_inductor_add: OBSERVABLE_MEMORY_EQUIVALENT
```

解析其中一侧 TTIR：

```bash
.venv/bin/python -m etv parse \
  examples/add/ttir/torch_inductor_add.ttir \
  --out build/torch_inductor_add.snapshot.json
```

Linux 可通过 `.[dev,ttir]` 安装 raw TTIR 前端；PyPI 不提供 macOS Triton
wheel，源码构建步骤见[依赖与环境](docs/dependencies.md)。

## `PROVED` 的边界

对于固定 PairSpec，ETV 穷举每个已启动的 program/lane，证明有效输出域完整、
输出地址单射且两侧地址对应，再把逻辑读取和标量 ABI 归一化，把全部输出计算根
放入同一个 `egglog==13.2.0` e-graph。代数重写在进入 e-graph 前由 Z3 证明，
依赖布局等事实的规则必须由 PairSpec gate 启用并记录为未经验证的可信公理。

当前 `ABSTRACT_FLOAT` 把浮点运算解释为数学实数。本项目的 `PROVED` 不是
IEEE-754、容差或 GPU 位级等价，也不证明 buffer 分配边界安全。libtriton 负责
语法和 IR 合法性；ETV 的 TTIR 语义提升器仍位于可信计算基中。准确契约见
[完整验证过程](docs/verification_process.md)。

## 命令

```text
etv check SPEC [--out DIR] [--json]
etv inspect PROGRAM
etv parse INPUT [--function NAME] [--out FILE] [--no-assembly]
etv rules [--json]
etv validate-rule [RULE_ID]
etv explain REPORT_JSON
```

`check` 的退出码分别为：`PROVED=0`、`DISPROVED=1`、`UNKNOWN/输入错误=2`。
`--out` 生成确定性的 `report.json` 与 `report.md`。

## 仓库结构

```text
etv/                         验证器实现
tools/extract_add_pair.py    真实 Add 程序对提取与编译工具
examples/add/                实际生成的源文件、TTIR、PairSpec 和 provenance
tests/fixtures/semantic/     仅供单元测试的合成夹具，不作为用户验证样例
tests/                       单元、性质、CLI 与真实 TTIR 集成测试
docs/add_validation.md       真实 Add 的来源和逐步验证记录
docs/verification_process.md 输入、语义、证明、等式饱和与信任边界
docs/architecture.md         组件和数据流
docs/implementation_status.md 当前覆盖与缺口
docs/dependencies.md         固定依赖、提取依赖和环境重建
```

ETV 是验证器而不是优化器：它不从 e-graph 提取最低成本项；成功条件是对应的
可观察计算根属于同一个 egglog e-class。
