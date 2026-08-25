# 依赖与环境

## 运行时依赖

| 依赖 | 声明版本 | 实现期间安装的版本 | 用途 |
| --- | --- | --- | --- |
| Python | `>=3.11` | `3.12.13` | 实现、egglog 与 CLI |
| pip | 可编辑安装要求 `>=21.3` | `26.1.2` | PEP 660 可编辑安装 |
| setuptools | `>=61` | `84.0.0` | PEP 517 构建后端 |
| `z3-solver` | `==4.16.0.0` | `4.16.0.0`（`z3 4.16.0`） | 规则准入与参数化证明的 SMT 检查 |
| `egglog` | `==13.2.0` | `13.2.0` | 唯一的 e-graph 与等式饱和后端 |
| Triton/libtriton | `==3.7.1`（`ttir` extra） | `3.7.1` | 原始 TTIR/MLIR 方言注册、解析和 IR 验证 |

完成安装后，默认验证路径不依赖网络。只有 PairSpec 显式启用 `llm` 时才会通过
DeepSeek HTTPS API 发送候选表达式和可用 fact gate；客户端使用 Python 标准库，
不增加运行时包依赖。Z3、egglog 与 libtriton 均通过 Python API 导入，不解析 CLI 输出。

子图划分复用同一个 DeepSeek 客户端，不引入新的运行时包。启用
`partition.enabled` 后会发送一次左右完整 Semantic Program、PairSpec 事实和计算根
摘要；API key 仍只从 `DEEPSEEK_API_KEY` 读取，报告只保存模型/usage 与请求响应哈希。

egglog 13.2.0 要求 Python 3.11 以上，因此 ETV 不再支持原来的 Python 3.9 基础路径。其 wheel 同时包含 Rust egglog 绑定；上游包还声明了 `typing-extensions`、`black`、`graphviz`、`anywidget`、`cloudpickle>=3` 和 `opentelemetry-api` 等传递依赖。本工程不直接调用其中的 notebook/可视化功能，但保留上游完整依赖集合以避免维护非官方裁剪包。

固定使用 4.16.0.0 是因为其 macOS arm64 wheel 使用标准的 `macosx_15_0_arm64` 平台标签。实现期间可用的较新 5.0.0.0 包安装了非标准的 `macosx_13_3_arm64` wheel；该 wheel 虽能执行，但无法通过 `pip check`，因此被拒绝。

## 开发与验证依赖

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| `pytest` | `8.4.2` | 单元测试、集成测试和 CLI 测试 |
| `hypothesis` | `6.141.1` | 整数恒等式的性质测试 |

直接开发依赖记录在 `requirements-dev.txt` 中，传递依赖由 pip 解析。仅运行 Semantic JSON/Prims 路径可通过以下命令重建环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

实现时使用的环境是 macOS Darwin arm64。本地虚拟环境已被 git 忽略。

## 真实 Add 提取依赖

`examples/add` 的日常验证只需要 ETV 和 libtriton；只有重新生成该目录时才需要
以下额外依赖。它们记录在 `requirements-extraction.txt`，不进入 ETV 运行时依赖：

| 依赖 | 固定版本/提交 | 用途 |
| --- | --- | --- |
| `InfiniTensor/ntops` | `9ae4166ad342e4745f0eed13a5a20d069e994fc0` | 被验证 Add 的上游实现和布局 arrangement |
| `InfiniTensor/ninetoothed` | `efe519d1b12a820e7aa605d775af3d52c8b0d605`，0.26.0 | 从 ntops DSL/SSA 生成 Triton 源码 |
| PyTorch | `2.8.0` | `TorchRefsMode`、`make_fx` 和 Prims 图捕获；不使用 TorchInductor |
| Triton/libtriton | `3.7.1` | 仅九齿侧 AST 前端、TTIR pass 和最终 parser/verifier |

实现期间隔离环境中由 PyTorch/ninetoothed 解析出的主要传递依赖包括
`numpy==2.3.4`、`sympy==1.14.0`、`mpmath==1.3.0`、`filelock==3.32.2`、
`fsspec==2026.7.0`、`jinja2==3.1.6`、`MarkupSafe==3.0.3` 和
`networkx==3.6.1`。这些库不由 ETV 直接调用，未来重新安装时应以固定直接依赖
解析和 `pip check` 为准。

提取工具要求 ninetoothed 以 editable 模式指向固定 checkout，并检查 ntops/
ninetoothed 的 Git HEAD、两个 ntops 源文件哈希以及 Python 包版本。详细命令见
[真实 Add 验证](add_validation.md)。

Linux 上启用 raw TTIR：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,ttir]'
```

PyPI 的 `triton==3.7.1` 要求 Python 3.10 至 3.14，并只发布 manylinux x86_64/aarch64 wheel。ETV 的 Python 3.11 下限由 egglog 决定；raw TTIR 路径还要求可加载的 3.7.1 libtriton。

## Triton 版本与安装

ETV 将 parser 精确锁定为 Triton/libtriton 3.7.1。该版本提供 `ir.load_dialects(context)`、`ir.parse_mlir_module(path, context)` 和 MLIR module verifier，可注册 Triton、TritonGPU、算术、数学、SCF、GPU、CF、LLVM 等内置方言。精确版本检查可以避免 TTIR 自定义语法、属性和 Python 绑定演进造成静默行为漂移；版本不匹配时返回 `TTIR_VERSION_MISMATCH`。

Linux 直接使用前述 `.[dev,ttir]` 安装即可。解析与验证 TTIR 不要求 CUDA 或 GPU。

PyPI 不提供 macOS wheel。当前 macOS arm64 环境已用以下源码构建过程验证：

```bash
brew install python@3.12 cmake ninja
git clone --depth 1 --branch v3.7.1 https://github.com/triton-lang/triton.git
python3.12 -m venv /tmp/etv-triton-venv
/tmp/etv-triton-venv/bin/python -m pip install \
  'cmake>=3.20,<4.0' 'ninja>=1.11.1' 'pybind11>=2.13.1' \
  setuptools wheel lit
TRITON_BUILD_PROTON=OFF MAX_JOBS=4 \
  /tmp/etv-triton-venv/bin/python -m pip install \
  --no-build-isolation -e /path/to/triton
/tmp/etv-triton-venv/bin/python -m pip install -e '.[dev]'
```

构建过程会下载 Triton 锁定的 LLVM，并需要数 GB 临时空间。运行时会检查 `triton.__version__ == "3.7.1"`。
本次使用的 v3.7.1 提交为
`f797708c0626e5f9840ca5b0a98790e2c7cb09ad`。

## 实现过程中安装的构建依赖

为在当前 macOS arm64 机器上真实验证 libtriton 前端，额外安装/使用了：

| 依赖 | 版本或约束 | 用途 |
| --- | --- | --- |
| Homebrew Python | `3.12.13` | Triton 3.7.1 支持的解释器 |
| CMake | Homebrew `4.4.1`；隔离环境 `3.31.10` | Triton 构建；官方要求 `<4.0`，实际构建使用 3.31.10 |
| Ninja | `1.13.x` | 并行构建 libtriton |
| pybind11 | `3.0.4` | Python C++ 绑定构建 |
| lit | `18.1.8` | Triton 构建系统工具 |
| Triton 锁定 LLVM | v3.7.1 构建脚本下载的 macOS arm64 包 | MLIR/Triton 编译基础设施 |

这些构建产物位于隔离的 `/tmp` 环境，不提交到仓库。

## 已验证环境

- macOS 26 arm64；
- Python 3.12.13；
- 从官方 `v3.7.1` 标签构建的 libtriton；
- 真实生成的 ntops/ninetoothed Add TTIR 与 Torch Prims 图通过两条独立前端提升，
  程序对经 fact-derived load/store 重写得到 `PROVED(OBSERVABLE_MEMORY_EQUIVALENT)`；
- 包含 `scf.for`/`scf.yield` 的测试模块通过完整解析，并正确保持在语义提升边界之外。

SymPy 由 PyTorch 提取环境传递安装，但 ETV 不使用它进行规则准入。内建代数规则
仍由 Z3 直接证明；自定义规则默认也如此，但 PairSpec 可显式选择弱化策略并承担
报告中列出的信任风险。

## 依赖信任

libtriton 的 parser/verifier、Z3、egglog 13.2.0、ETV 的 egglog term 编码和语义提升器属于当前可信计算基。Pytest 和 Hypothesis 只影响开发证据。任何实际应用且标记为 `admitted_unverified` 的规则都属于输入信任边界；报告只会暴露该风险，无法弥补错误规则造成的不健全性。
