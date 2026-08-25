# 示例

`examples/add` 展示同一 Add 语义的两条输入路线：

```text
左侧：ntops -> ninetoothed -> Triton -> raw TTIR
右侧：PyTorch -> TorchRefsMode/make_fx -> Torch Prims JSON
```

Torch 侧不经过 TorchInductor。

| 路径 | 内容 |
| --- | --- |
| `add/pair.json` | 固定 `[8,16]` PairSpec |
| `add/pair_parametric.json` | 固定 rank、符号维度 `a,b,c` 的 PairSpec |
| `add/provenance.json` | 上游提交、工具链和 artifact SHA-256 |
| `add/sources/ntops_add.triton.py` | ninetoothed 生成的 Triton 源码 |
| `add/sources/ntops_add.triton.json` | ninetoothed SSA/布局元数据 |
| `add/sources/torch_prims_add.fx.txt` | PyTorch 2.8 捕获的 Prims FX 图 |
| `add/prims/torch_add.prims.json` | 严格、可重放、固定 rank 的符号 Prims 输入 |
| `add/ttir/ntops_add.ttir` | Triton 3.7.1 为 CUDA sm80 生成的九齿侧 TTIR |
| `add/ttir/parametric_2d_add.ttir` | 参数化二维 TTIR 验证输入 |

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

结果应为：

```text
PROVED parametric_ninetoothed_add_vs_torch_prims_add: OBSERVABLE_MEMORY_EQUIVALENT
```

来源与固定实例见[Add 验证](../docs/add_validation.md)，逐条规则与 e-graph 变化见
[参数化 Add 验证全过程](../docs/add_parametric_verification_details.md)。
