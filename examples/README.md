# 示例

`examples/` 只包含一个面向用户的验证样例：从真实上游代码重新生成的 Add
程序对。旧的手写 Semantic JSON、故障变体和小型 FMA TTIR 已移到
`tests/fixtures/semantic/`，它们只用于验证 ETV 自身行为，不再作为工程能力展示。

## 内容

| 路径 | 内容 |
| --- | --- |
| `add/pair.json` | `[8,16]`、连续布局、单 program 的实际 PairSpec |
| `add/provenance.json` | 上游提交、源码与 artifact 哈希、工具链和适配声明 |
| `add/sources/ntops_add.triton.py` | ninetoothed 从 ntops Add 生成的 Triton 源码 |
| `add/sources/ntops_add.triton.json` | ninetoothed 生成的 SSA/布局元数据 |
| `add/sources/torch_inductor_add.fx.txt` | PyTorch FakeTensor 捕获的 FX 图 |
| `add/sources/torch_inductor_add.generated.py` | TorchInductor 原样生成的内核源码 |
| `add/sources/torch_inductor_add.triton.py` | 仅移除 launch decorator 的离线 TTIR 编译适配源码 |
| `add/ttir/*.ttir` | Triton 3.7.1 为 CUDA sm80 生成并优化的两侧 TTIR |

## 验证

```bash
.venv/bin/python -m etv check examples/add/pair.json \
  --out build/add_upstream
```

结果应为：

```text
PROVED ntops_add_vs_torch_inductor_add: OBSERVABLE_MEMORY_EQUIVALENT
```

复现提取过程和结果解释见[真实 Add 验证](../docs/add_validation.md)。
