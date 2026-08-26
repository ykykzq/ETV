# 依赖与环境

## 运行时依赖

| 依赖 | 声明版本 | 实现期间安装的版本 | 用途 |
| --- | --- | --- | --- |
| Python | `>=3.11` | `3.12.13` | 实现、egglog 与 CLI |
| uv | `>=0.12.0,<0.13` | `0.12.6` | 环境、锁文件与包安装 |
| setuptools | `>=61` | `84.0.0` | PEP 517 构建后端 |
| `z3-solver` | `==4.16.0.0` | `4.16.0.0`（`z3 4.16.0`） | 规则准入与参数化证明的 SMT 检查 |
| `egglog` | `==13.2.0` | `13.2.0` | 唯一的 e-graph 与等式饱和后端 |
| Triton/libtriton | `==3.7.1`（`ttir` extra） | `3.7.1` | 原始 TTIR/MLIR 方言注册、解析和 IR 验证 |

完成安装后，默认验证路径不依赖网络。只有 PairSpec 显式启用 `llm` 时才会通过
DeepSeek HTTPS API 发送候选表达式、自然语言上下文和可用 predicate gate；客户端使用 Python 标准库，
不增加运行时包依赖。Z3、egglog 与 libtriton 均通过 Python API 导入，不解析 CLI 输出。

子图划分复用同一个 DeepSeek 客户端，不引入新的运行时包。启用
`partition.enabled` 后会发送一次左右完整 ETV IR Program、PairSpec 谓词、自然语言
上下文和计算根
摘要；API key 仍只从 `DEEPSEEK_API_KEY` 读取，报告只保存模型/usage 与请求响应哈希。

egglog 13.2.0 要求 Python 3.11 以上，因此 ETV 不再支持原来的 Python 3.9 基础路径。其 wheel 同时包含 Rust egglog 绑定；上游包还声明了 `typing-extensions`、`black`、`graphviz`、`anywidget`、`cloudpickle>=3` 和 `opentelemetry-api` 等传递依赖。本工程不直接调用其中的 notebook/可视化功能，但保留上游完整依赖集合以避免维护非官方裁剪包。

固定使用 4.16.0.0 是因为其 macOS arm64 wheel 使用标准的 `macosx_15_0_arm64` 平台标签。实现期间可用的较新 5.0.0.0 包安装了非标准的 `macosx_13_3_arm64` wheel；该 wheel 虽能执行，但无法通过 `pip check`，因此被拒绝。

## 开发与验证依赖

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| `pytest` | `8.4.2` | 单元测试、集成测试和 CLI 测试 |
| `hypothesis` | `6.141.1` | 整数恒等式的性质测试 |

开发依赖统一声明在 `pyproject.toml` 的 `dev` extra 中，直接和传递依赖由
`uv.lock` 固定。仅运行
证明内核单元测试（raw TT IR 集成测试会跳过）可通过以下命令重建环境：

```bash
uv sync --locked --python 3.12 --extra dev
uv run --locked --extra dev pytest -q
```

实现时使用的环境是 macOS Darwin arm64。本地虚拟环境已被 git 忽略。

## 真实 Add 提取依赖

`examples/add` 的日常验证只需要 ETV 和 libtriton；只有重新生成该目录时才需要
以下额外依赖。它们统一声明在 `pyproject.toml` 的 `extraction` extra 中，不进入 ETV
核心运行时依赖：

| 依赖 | 固定版本/提交 | 用途 |
| --- | --- | --- |
| `InfiniTensor/ntops` | `9ae4166ad342e4745f0eed13a5a20d069e994fc0` | 被验证 Add 的上游实现和布局 arrangement |
| `InfiniTensor/ninetoothed` | `efe519d1b12a820e7aa605d775af3d52c8b0d605`，0.26.0 | 从 ntops DSL/SSA 生成 Triton 源码 |
| PyTorch | `2.8.0` | FakeTensor FX、TorchInductor GraphLowering 与 Triton kernel codegen |
| Triton/libtriton | `3.7.1` | 两侧 AST 前端、TTIR pass 和最终 parser/verifier |

实现期间隔离环境中由 PyTorch/ninetoothed 解析出的主要传递依赖包括
`numpy==2.5.2`、`sympy==1.14.0`、`mpmath==1.3.0`、`filelock==3.32.4`、
`fsspec==2026.7.0`、`jinja2==3.1.6`、`MarkupSafe==3.0.3` 和
`networkx==3.6.1`。这些库不由 ETV 直接调用，重新安装时以 `uv.lock` 为准。

提取工具要求 ninetoothed 以 editable 模式指向固定 checkout，并检查 ntops/
ninetoothed 的 Git HEAD、两个 ntops 源文件哈希以及 Python 包版本。详细命令见
[真实 Add 验证](add_validation.md)。

`extraction` 与 `ttir` 不能装进同一个 uv resolution：Linux 的 PyTorch 2.8.0 依赖
Triton 3.4.0，而 ETV 固定 libtriton 3.7.1。该冲突已写入 `[tool.uv].conflicts`，
`uv sync --all-extras` 会主动拒绝。macOS 上 `extraction` 不直接安装 ninetoothed，
因为其无条件 Triton 依赖只提供 Linux wheel；源码环境在固定 3.7.1 构建完成后以
`--no-deps` 安装 ninetoothed。

完整提取/全算子环境由脚本建立：

```bash
tools/setup_uv_ttir.sh
```

Linux 上启用 raw TTIR：

```bash
uv sync --locked --python 3.12 --extra dev --extra ttir
uv run --locked --extra dev --extra ttir pytest -q
```

PyPI 的 `triton==3.7.1` 要求 Python 3.10 至 3.14，并只发布 manylinux x86_64/aarch64 wheel。ETV 的 Python 3.11 下限由 egglog 决定；raw TTIR 路径还要求可加载的 3.7.1 libtriton。

## Triton 版本与安装

ETV 将 parser 精确锁定为 Triton/libtriton 3.7.1。该版本提供 `ir.load_dialects(context)`、`ir.parse_mlir_module(path, context)` 和 MLIR module verifier，可注册 Triton、TritonGPU、算术、数学、SCF、GPU、CF、LLVM 等内置方言。精确版本检查可以避免 TTIR 自定义语法、属性和 Python 绑定演进造成静默行为漂移；版本不匹配时返回 `TTIR_VERSION_MISMATCH`。

Linux 直接使用前述 `uv sync --extra dev --extra ttir` 即可。解析与验证 TTIR 不要求
CUDA 或 GPU。

PyPI 不提供 macOS wheel。当前 macOS arm64 环境已用仓库脚本中的以下等价流程验证：

```bash
tools/setup_uv_ttir.sh
.venv-ttir/bin/python -m pytest -q
```

脚本用 uv 安装 `ttir-build` extra，再以固定提交源码构建；过程会下载 Triton 锁定的
LLVM，并需要数 GB 临时空间。运行时会检查 `triton.__version__ == "3.7.1"`。
本次使用的 v3.7.1 提交为
`f797708c0626e5f9840ca5b0a98790e2c7cb09ad`。

## 实现过程中安装的构建依赖

为在当前 macOS arm64 机器上真实验证 libtriton 前端，额外安装/使用了：

| 依赖 | 版本或约束 | 用途 |
| --- | --- | --- |
| Homebrew Python | `3.12.13` | Triton 3.7.1 支持的解释器 |
| uv | `0.12.6` | 创建、同步和锁定全部 Python 环境 |
| CMake | `3.31.6` | Triton 构建；固定低于 4.0 |
| Ninja | `1.13.0` | 并行构建 libtriton |
| pybind11 | `3.1.0` | Python C++ 绑定构建 |
| lit | `23.1.0` | Triton 构建系统工具 |
| wheel | `0.48.0` | editable/source build 支持 |
| Triton 锁定 LLVM | v3.7.1 构建脚本下载的 macOS arm64 包 | MLIR/Triton 编译基础设施 |

构建产物默认位于 git 忽略的 `.venv-ttir` 和 `build/upstream`，不提交到仓库。
`TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF` 只关闭 Triton 自己的 C++ 单元测试和
googletest 下载，不关闭 libtriton Python 模块，也不影响本工程随后执行的 TT IR
集成测试。

## 已验证环境

- macOS 26 arm64；
- Python 3.12.13；
- 从官方 `v3.7.1` 标签构建的 libtriton；
- 真实生成的 ntops/ninetoothed Add TTIR 与 TorchInductor Add TTIR 通过相同 libtriton
  前端、独立语义提升得到内部 IR，程序对经关系/代数重写得到
  `PROVED(OBSERVABLE_MEMORY_EQUIVALENT)`；
- 包含 `scf.for`/`scf.yield` 的测试模块通过完整解析，并正确保持在语义提升边界之外。
- ETV 完整环境测试为 `88 passed`；ntops 的 76 个算子模块、2102 个 CUDA 用例均完成
  收集，但在本机因无 CUDA 全部跳过；Torch reference 离线矩阵中 67 个生成 TTIR、
  6 个走外部 kernel、3 个受 CPU-only PyTorch trace 限制。详见
  [uv 环境与全算子测试](operator_testing.md)。

SymPy 由 PyTorch 提取环境传递安装，但 ETV 不使用它进行规则准入。内建代数规则
仍由 Z3 直接证明；外部用户代数规则默认也如此，但 PairSpec 可显式选择弱化策略并
承担报告中列出的信任风险。ETV 使用上游 PyPI `egglog==13.2.0` 的公开 Python API，
没有 fork、打补丁或 hack egglog、Z3、libtriton。

## 依赖信任

libtriton 的 parser/verifier、Z3、egglog 13.2.0、ETV 的 egglog term 编码和 TT IR
语义提升器属于当前可信计算基。Pytest 和 Hypothesis 只影响开发证据。任何实际应用且
标记为 `admitted_unverified` 的规则都属于输入信任边界；报告只会暴露该风险，无法
弥补错误规则造成的不健全性。
