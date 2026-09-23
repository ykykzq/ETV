# 单 LHS kernel / 多 RHS kernel 示例

本目录从完整 ntops benchmark 中抽取四个真实程序对。每个示例的 LHS 都只有一个
Ninetoothed/Triton kernel；RHS 是同一测试 specialization 经 TorchInductor 生成的有序
多 kernel 程序。TTIR、ABI、存储 provenance 和 launch 顺序均保持原样；PairSpec 中的
文件路径改为本目录相对路径，并依据生成源码签名、runtime 参数顺序和编译后 TTIR 形参
修正了旧 benchmark 中受标量特化错位影响的固定参数绑定。

## 文件结构

每个子目录包含：

| 文件 | 内容 |
| --- | --- |
| `lhs.ttir` | 单 kernel 的 LHS TTIR |
| `rhs-0.ttir`、`rhs-1.ttir` | RHS 按执行顺序捕获的 TTIR |
| `pair.json` | 可直接传给 ETV 的 PairSpec |
| `mapping.json` | 从运行时 storage provenance 构造 ABI 和中间存储边的诊断记录 |

`batch_norm` 和 `quantile` 的第一个 RHS kernel 各有两个被观察的 store，因此 PairSpec
把它们表示为同一执行 step 中的两个组件；这不是把一个物理 launch 误算成两个 launch。

## 示例概览

| 示例 | LHS | RHS 有序组件 | 中间数据流 | 当前结果 |
| --- | ---: | ---: | --- | --- |
| `argsort` | 1 kernel | 2 kernel / 2 组件 | `Input0 -> Internal0 -> Output` | `UNKNOWN: TTIR_REGION_SEMANTICS_UNSUPPORTED` |
| `rms_norm` | 1 kernel | 2 kernel / 2 组件 | 第一阶段生成归一化统计量，第二阶段读取输入和统计量生成输出 | `UNKNOWN: TTIR_REGION_SEMANTICS_UNSUPPORTED` |
| `batch_norm` | 1 kernel | 2 kernel / 3 组件 | 第一 kernel 同步产生 `Internal0`、`Internal1`，第二 kernel 合并两者 | `UNKNOWN: TTIR_REGION_SEMANTICS_UNSUPPORTED` |
| `quantile` | 1 kernel | 2 kernel / 3 组件 | 第一 kernel 产生两个中间结果，第二 kernel 同时读取它们和 `Input1` | `UNKNOWN: TTIR_REGION_SEMANTICS_UNSUPPORTED` |

这里的“组件”对应 PairSpec 的 `metadata.launches.rhs` 项；具有相同 `step` 和 TTIR 文件的
组件属于同一个物理 kernel 的不同 store。验证器先按 step 重建顺序内存，再将中间
storage 的读取替换为前序 store 的值。

## 运行

```bash
.venv-ttir/bin/python -m etv check examples/single_lhs_multi_rhs/argsort/pair.json \
  --out build/single_lhs_multi_rhs/argsort

.venv-ttir/bin/python -m etv check examples/single_lhs_multi_rhs/rms_norm/pair.json \
  --out build/single_lhs_multi_rhs/rms_norm

.venv-ttir/bin/python -m etv check examples/single_lhs_multi_rhs/batch_norm/pair.json \
  --out build/single_lhs_multi_rhs/batch_norm

.venv-ttir/bin/python -m etv check examples/single_lhs_multi_rhs/quantile/pair.json \
  --out build/single_lhs_multi_rhs/quantile
```

四个示例目前都在 TTIR 提升阶段返回 `UNKNOWN`，原因是 RHS 使用的 `tt.reduce` 或
`scf.for` region 语义尚未实现。这些示例证明多 launch PairSpec、顺序和 storage 边已经
被捕获，但不能作为多 launch 等价性已经证明的证据。

## 来源

- `argsort`：`test_argsort[True-shape8-0-dtype8-cuda-0.001-0.001]`
- `rms_norm`：`test_rms_norm[shape4-dtype4-cuda-0.001-0.001-True-1e-05]`
- `batch_norm`：`test_batch_norm[shape4-dtype4-cuda-0.001-0.001-False-0.001]`
- `quantile`：`test_quantile[shape0-dtype0-cuda-0.001-0.001-nearest-False-True]`

原始 manifest、源码、pytest 日志和完整验证报告仍保存在对应的 `benchmark/` 目录中。
