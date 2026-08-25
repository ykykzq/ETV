# 依赖与环境

## 运行时依赖

| 依赖 | 声明版本 | 实现期间安装的版本 | 用途 |
| --- | --- | --- | --- |
| Python | `>=3.11` | `3.12.13` | 实现、egglog 与 CLI |
| pip | 可编辑安装要求 `>=21.3` | `26.0.1` | PEP 660 可编辑安装 |
| setuptools | `>=61` | venv 中为 `58.0.4`；隔离构建会获取兼容版本 | PEP 517 构建后端 |
| `z3-solver` | `==4.16.0.0` | `4.16.0.0`（`z3 4.16.0`） | 规则准入的 UNSAT 检查 |
| `egglog` | `==13.2.0` | `13.2.0` | 唯一的 e-graph 与等式饱和后端 |
| Triton/libtriton | `==3.7.1`（`ttir` extra） | `3.7.1` | 原始 TTIR/MLIR 方言注册、解析和 IR 验证 |

完成安装后，验证器运行时不依赖网络。Z3、egglog 与 libtriton 均通过 Python API 导入，不解析 CLI 输出。

egglog 13.2.0 要求 Python 3.11 以上，因此 ETV 不再支持原来的 Python 3.9 基础路径。其 wheel 同时包含 Rust egglog 绑定；上游包还声明了 `typing-extensions`、`black`、`graphviz`、`anywidget`、`cloudpickle>=3` 和 `opentelemetry-api` 等传递依赖。本工程不直接调用其中的 notebook/可视化功能，但保留上游完整依赖集合以避免维护非官方裁剪包。

固定使用 4.16.0.0 是因为其 macOS arm64 wheel 使用标准的 `macosx_15_0_arm64` 平台标签。实现期间可用的较新 5.0.0.0 包安装了非标准的 `macosx_13_3_arm64` wheel；该 wheel 虽能执行，但无法通过 `pip check`，因此被拒绝。

## 开发与验证依赖

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| `pytest` | `8.4.2` | 单元测试、集成测试和 CLI 测试 |
| `hypothesis` | `6.141.1` | 整数恒等式的性质测试 |

直接开发依赖记录在 `requirements-dev.txt` 中，传递依赖由 pip 解析。仅运行 Semantic JSON 路径可通过以下命令重建环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

实现时使用的环境是 macOS Darwin arm64。本地虚拟环境已被 git 忽略。

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
- `add_mul.ttir` 与 `add_fma.ttir` 均通过解析、IR 验证、提升和等价性证明；
- 包含 `scf.for`/`scf.yield` 的测试模块通过完整解析，并正确保持在语义提升边界之外。

## 有意未安装的依赖

### SymPy

Z3 已直接证明当前所有代数规则，因此第二个简化器不会增强证明。对于规模更大的带条件代数规则注册表，SymPy 仍可用作可读性和回归测试辅助工具。

## 依赖信任

libtriton 的 parser/verifier、Z3、egglog 13.2.0、ETV 的 egglog term 编码和语义提升器属于当前可信计算基。Pytest 和 Hypothesis 只影响开发证据。PairSpec `trusted_fact` 规则未经等式验证，也属于输入信任边界；报告只会暴露该风险，无法弥补错误规则造成的不健全性。
