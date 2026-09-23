# ntops 算子输入规格

本目录保存从固定版本 ntops 测试中抽取并人工复核的 75 份输入规格。每份 YAML 都可以独立
阅读，只描述测试实际覆盖的调用前提，不描述输出语义、数值关系、副作用或验证结论。

## 目录结构

- `operators/*.yaml`：每个独立 kernel 一份 `operator-requires/v1`；
- `index.yaml`：75 份规格的稳定索引、类别和测试文件映射；
- `operator_spec.schema.json`：独立输入规格的 JSON Schema；
- `../ntops_test_evidence.json`：生成规格时使用的逐测试原始证据；
- `../ntops_LICENSE`：上游测试片段的 Apache-2.0 许可证。

## 文件格式

每份规格只有 `schema`、`operator`、`requires` 和 `evidence` 四个顶层字段。
`requires.arguments` 按函数参数逐一记录类型、默认值和约束。单参数条件保存在对应参数的
`constraints` 中，涉及多个参数的 shape、dtype、广播或维度关系保存在 `relations` 中。
`observed_domain_zh` 是对原始测试生成域的直接摘要，用来保留暂未完全结构化的随机生成和
组合 case。带 `optional: true` 的参数，其张量约束仅在参数非空时适用。

公共浮点、整数及矩阵约束都已经复制到具体算子文件中；算子文件不引用 `common.yaml`、
其他规格或外部 JSON Pointer。`evidence` 只记录证据等级、上游仓库、固定 commit、许可证、
测试文件及测试函数位置。`inferred_from_tests` 表示这些约束来自有限测试生成器，不应解释成
完整 PyTorch API 的合法输入域。

## 重新生成与校验

```bash
.venv/bin/python tools/extract_ntops_test_specs.py \
  --ntops-repo build/upstream/ntops \
  --output specs/ntops_test_evidence.json
.venv/bin/python tools/materialize_ntops_operator_specs.py
.venv/bin/python tools/materialize_ntops_operator_specs.py --check
```

生成器需要开发依赖 PyYAML，并要求 ntops checkout 与证据文件均对应 commit
`9ae4166ad342e4745f0eed13a5a20d069e994fc0`。`--check` 不修改文件，只检查缺失、过期或多余
的算子 YAML。
