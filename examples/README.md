# 示例

这些示例以机器方式检查压缩包中的 add 前沿证明及其必要负例。

| Spec | 结果 | 用途 |
| --- | --- | --- |
| `specs/add_proved.json` | `PROVED` | 二维步长感知 ntops 与线性 Inductor |
| `specs/add_fma_proved.json` | `PROVED` | `ABSTRACT_FLOAT` 下经 Z3 准入的 `fma == mul+add` |
| `specs/add_raw_ttir_proved.json` | `PROVED` | 两个由 libtriton 解析和验证的原始 TTIR 文件 |
| `specs/add_bad_stride.json` | `DISPROVED` | 后续行的首个 output 地址不同 |
| `specs/add_bad_mask.json` | `DISPROVED` | rhs 漏掉逻辑元素 127 |
| `specs/add_bad_compute.json` | `DISPROVED` | 精确值模型区分 add 与 sub |
| `specs/add_missing_alias.json` | `UNKNOWN` | 缺少所需 no-alias 前提 |

运行所有 Semantic JSON 用例：

```bash
.venv/bin/python -m etv check examples/specs/add_proved.json
.venv/bin/python -m etv check examples/specs/add_fma_proved.json
.venv/bin/python -m etv check examples/specs/add_bad_stride.json
.venv/bin/python -m etv check examples/specs/add_bad_mask.json
.venv/bin/python -m etv check examples/specs/add_bad_compute.json
.venv/bin/python -m etv check examples/specs/add_missing_alias.json
```

在安装 Triton 3.7.1 后运行 raw TTIR 用例：

```bash
.venv/bin/python -m etv parse examples/ttir/add_mul.ttir
.venv/bin/python -m etv check examples/specs/add_raw_ttir_proved.json
```

负例命令退出码为 1，UNKNOWN 为 2，适用于 CI。添加 `--out build/<case>` 可保留 `report.json` 与 `report.md`。
