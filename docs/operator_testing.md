# uv 环境与全算子测试

## 测试范围

本页记录 2026-08-26 在 macOS 26 arm64 上完成的环境建立和测试。测试对象固定为：

- ETV 当前提交的 88 个 pytest 用例；
- `InfiniTensor/ntops` 提交
  `9ae4166ad342e4745f0eed13a5a20d069e994fc0` 中的 76 个算子测试模块；
- ninetoothed 提交
  `efe519d1b12a820e7aa605d775af3d52c8b0d605`；
- Python 3.12.13、PyTorch 2.8.0、Triton/libtriton 3.7.1、uv 0.12.6。

检查时 ntops 上游 `master` 的 HEAD 仍是上述提交。76 个测试模块共展开为 2102 个
pytest 参数化用例。

## 环境建立

项目使用 `uv.lock` 固定直接和传递依赖。macOS 可先执行 `brew install uv`；其他平台
使用对应包管理器安装满足 `>=0.12.0,<0.13` 的 uv。核心测试环境通过以下命令建立：

```bash
uv sync --locked --python 3.12 --extra dev
uv run --locked --extra dev pytest -q
```

`extraction` 与 `ttir` 被声明为互斥 extra，因为 Linux 的 PyTorch 2.8.0 依赖
Triton 3.4.0，而 ETV 前端固定 Triton 3.7.1。macOS 又没有 PyPI Triton wheel，不能
依靠一个 `--all-extras` 环境正确解决这两个 ABI。仓库提供独立环境引导脚本：

```bash
tools/setup_uv_ttir.sh
.venv-ttir/bin/python -m pytest -q
```

脚本执行以下操作：

1. 用 uv 按锁文件建立 `.venv-ttir`，安装 `dev`、`extraction` 和 `ttir-build`；
2. 获取并校验固定提交的 Triton、ninetoothed 和 ntops；
3. 在 macOS 上源码构建 Triton 3.7.1/libtriton；
4. 用 `--no-deps` 安装固定源码 checkout，防止 ninetoothed 再引入 Triton 3.4.0；
5. 检查 PyTorch、Triton 和 `triton._C.libtriton` 可导入。

同步使用 `--inexact`，仅用于保留随后由脚本按 Git commit 管理的三个 editable 源码包；
pyproject 声明的依赖仍由 `--locked` 强制匹配 `uv.lock`。

构建目录默认位于 `build/upstream`，uv 缓存位于 `build/uv-cache`，均被 git 忽略。
源码构建需要网络、C/C++ 工具链、约 4 GB 磁盘空间和数分钟时间。可通过
`ETV_PYTHON`、`ETV_TTIR_VENV`、`ETV_UPSTREAM_ROOT` 和 `UV_CACHE_DIR` 覆盖默认位置。

## ETV 测试结果

核心 uv 环境未安装 libtriton 时：

```text
81 passed, 7 skipped
```

7 项全部是显式要求 Triton/libtriton 3.7.1 的 raw TT IR 前端测试。源码构建环境中：

```text
88 passed, 0 skipped, 0 failed
```

因此当前仓库测试在完整前端环境中全部通过；前一组的 skip 不能算作通过。

## ntops 数值测试结果

在固定 ntops checkout 中执行：

```bash
cd build/upstream/ntops
../../../.venv-ttir/bin/python -m pytest --collect-only -q \
  -o addopts='' --rootdir=.
../../../.venv-ttir/bin/python -m pytest -q -o addopts='' --rootdir=.
```

结果为：

```text
2102 tests collected
2102 skipped: CUDA not available
```

ntops 的所有测试函数都带 `skip_if_cuda_not_available`，输入也明确创建在 `cuda`。
当前机器没有 NVIDIA CUDA 设备，因此没有一个 ntops kernel 与 PyTorch reference 的
数值对拍实际执行。这个结果只证明测试集合可导入、可完整收集，不能证明 76 个算子
数值正确。

## TorchInductor TTIR 矩阵

为在无 CUDA 设备时继续检查 PyTorch reference 侧，运行：

```bash
.venv-ttir/bin/python tools/test_ntops_torch_ttir.py \
  --ntops-repo build/upstream/ntops \
  --output build/ntops_torch_ttir.json
```

工具首先要求它的 76 个 case 与 ntops 的 76 个 `test_*.py` 文件严格一一对应。每个
模块选一个代表性 specialization，在独立 worker 进程中执行：

```text
FakeTensor FX -> TorchInductor decomposition/lowering -> Triton source
-> Triton AST frontend -> optimized TTIR
```

目标固定为离线 `cuda:sm80`，不会初始化 CUDA driver，也不会执行 kernel。最终结果：

| 分类 | 数量 | 含义 |
| --- | ---: | --- |
| `ttir_generated` | 67 | 共 72 个 Triton kernel，全部成功编译为优化 TTIR |
| `no_triton_kernel` | 6 | 默认 Inductor wrapper 只使用外部/ATen kernel，没有 TTIR 输入 |
| `fx_trace_error` | 3 | macOS CPU-only PyTorch 无法完成该 reference 的离线 CUDA trace |

成功生成 TTIR 的 67 个模块为：

```text
abs, acosh, adaptive_avg_pool2d, adaptive_max_pool2d, add, addmv,
argsort, atan, avg_pool2d, batch_norm, bitwise_and, bitwise_not,
bitwise_or, celu, clamp, conv2d, cos, cosh, diag, div, dropout, eq,
exp, fmax, ge, gelu, gt, instance_norm, isinf, isnan, layer_norm, le,
logsumexp, lp_pool1d, lp_pool2d, lp_pool3d, lt, max, max_pool1d,
max_pool2d, max_pool3d, maximum, mean, msort, mul, ne, neg, pow, relu,
rms_norm, rot90, rotary_position_embedding, round, rsqrt,
scaled_dot_product_attention, sgn, sigmoid, sign, signbit, silu, sin,
softmax, sort, stack, sub, tanh, threshold
```

没有 Triton kernel 的 6 个模块为：

| 算子 | 默认 wrapper 调用 |
| --- | --- |
| `addmm` | `extern_kernels.addmm` |
| `bmm` | `extern_kernels.bmm` |
| `matmul` | `extern_kernels.mm` |
| `mm` | `extern_kernels.mm` |
| `median` | `torch.ops.aten.median.dim` |
| `select_copy` | `torch.ops.aten.select_copy.int` |

三个 trace 限制为：

- `alpha_dropout`：RNG decomposition 内部的 CUDA `bernoulli_/copy_` 需要 CUDA-enabled
  PyTorch；
- `bincount`：FakeTensor 抛出 `DynamicOutputShapeException`，输出长度依赖输入值；
- `quantile`：decomposition 的 CUDA out-copy 在 CPU-only PyTorch 中不可执行。

这三个结果不能外推为 CUDA-enabled Linux 上必然失败。它们应在 NVIDIA 环境中复测。

## 结论边界

67/76 仍只是早期“每个模块一个代表 specialization”的 Torch reference 探针，不能
解释为 67 个算子已证明等价。当前 `benchmark/` 已在 NVIDIA CUDA 环境执行 76 个算子
的全部 2102 个参数化用例，并直接截获原测试 reference callable/FX graph，经
TorchInductor 收集 RHS TTIR。生成器按 reference 与 tensor output leaf 拆分 PairSpec，
并记录 pointer storage/tensor provenance、view offset、alias、scratch 与未映射依赖。

完整数值、采集、PairSpec 和形式化统计以 `benchmark/summary.json`、
`benchmark/REPORT.md` 为准。多 kernel orchestration、LHS 多 launch、store 间顺序
effect，以及 `scf.for`/`tt.reduce` 等 TTIR 语义仍不在当前验证器支持范围；相关用例
必须如实记为未运行或 `UNKNOWN`，不能从原数值测试通过外推为形式化证明。
