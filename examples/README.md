# 示例

`add/` 包含两套双 TT IR 程序对。

## 真实固定规模 Add

| 文件 | 内容 |
| --- | --- |
| `add/ttir/ntops_add.ttir` | ntops/ninetoothed/Triton 生成的左侧 TT IR |
| `add/ttir/torch_inductor_add.ttir` | PyTorch/TorchInductor/Triton 生成的右侧 TT IR |
| `add/pair.json` | 固定 `[8,16]`、128 元素的 PairSpec |
| `add/provenance.json` | 上游提交、工具版本、生成链和文件哈希 |
| `add/sources/` | 两侧 Triton 源、TorchInductor 原始源码与 FX 图 |

运行：

```bash
python -m etv check examples/add/pair.json --out build/add
```

## 参数化 Add

| 文件 | 内容 |
| --- | --- |
| `add/ttir/parametric_2d_add.ttir` | 二维 shape 参数、256 lane |
| `add/ttir/parametric_1d_add.ttir` | 线性 numel 参数、128 lane |
| `add/pair_parametric.json` | `a>0,b>0,c>0,a*b=c` 的 PairSpec |

运行：

```bash
python -m etv check examples/add/pair_parametric.json --out build/add_parametric
```

两个正式示例的左右程序都由 TT IR 前端读取。Semantic JSON 仅存在于
`tests/fixtures/semantic`，不属于示例或用户输入格式；其他小型 TT IR 集成夹具位于
`tests/fixtures/ttir`。

## 单 LHS kernel / 多 RHS kernel

`single_lhs_multi_rhs/` 从 benchmark 中抽取了 `argsort`、`rms_norm`、`batch_norm` 和
`quantile` 四个真实程序对，用于检查预划分多 launch 输入、顺序内存组合和中间 storage
依赖。详见 `single_lhs_multi_rhs/README.md`。
