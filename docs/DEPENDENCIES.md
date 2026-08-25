# 依赖与环境

## 运行时依赖

| 依赖 | 声明版本 | 实现期间安装的版本 | 用途 |
| --- | --- | --- | --- |
| Python | `>=3.9` | `3.9.6` | 实现与 CLI |
| pip | 可编辑安装要求 `>=21.3` | `26.0.1` | PEP 660 可编辑安装 |
| setuptools | `>=61` | venv 中为 `58.0.4`；隔离构建会获取兼容版本 | PEP 517 构建后端 |
| `z3-solver` | `==4.16.0.0` | `4.16.0.0`（`z3 4.16.0`） | 规则准入的 UNSAT 检查 |

验证器运行时不依赖网络。Z3 通过 Python API 导入，不解析 CLI 输出。

固定使用 4.16.0.0 是因为其 macOS arm64 wheel 使用标准的 `macosx_15_0_arm64` 平台标签。实现期间可用的较新 5.0.0.0 包安装了非标准的 `macosx_13_3_arm64` wheel；该 wheel 虽能执行，但无法通过 `pip check`，因此被拒绝。

## 开发与验证依赖

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| `pytest` | `8.4.2` | 单元测试、集成测试和 CLI 测试 |
| `hypothesis` | `6.141.1` | 整数恒等式的性质测试 |

直接开发依赖记录在 `requirements-dev.txt` 中，传递依赖由 pip 解析。可通过以下命令重建环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

实现时使用的环境是 macOS Darwin arm64。本地虚拟环境已被 git 忽略。

## 有意未安装的依赖

### Triton/libtriton

研究设计建议使用 Triton 自身已注册方言的 MLIR 解析器。当前工作区运行在没有 CUDA 的 Apple Silicon 上，而目标 TTIR 产物及内部绑定与特定 Linux/CUDA Triton 版本耦合。在这里安装一个无关版本无法验证该前端。因此，raw TTIR 适配器被明确记录为 M0 缺口，而没有用正则表达式解析来掩盖这一问题。

### egglog

压缩包将 Python 证明 API 的暴露能力列为必须进行的集成验证。MVP 只需要计算项等价，本地 e-graph 用一个规模较小的模块即可提供完整合并账本。当循环/定义域关系或可扩展的共享前沿综合足以证明引入成本后，应在完成证明重建验证的前提下引入 egglog。

### SymPy

Z3 已直接证明当前所有代数规则，因此第二个简化器不会增强证明。对于规模更大的带条件代数规则注册表，SymPy 仍可用作可读性和回归测试辅助工具。

## 依赖信任

Z3 属于当前可信计算基。Pytest 和 Hypothesis 只影响开发证据，运行时包不会导入它们。e-graph 引擎由本仓库实现，并由直接的同余与重写测试覆盖。
