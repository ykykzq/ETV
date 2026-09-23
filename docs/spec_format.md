# 算子输入规格格式

`specs/ntops/operators/` 中的 YAML 只描述算子调用前必须满足的输入条件。输出值、数值误差、
副作用、别名关系和验证结论均不属于这批输入规格。

## 设计原则

1. 每个算子使用一份可独立阅读的文件，不引用公共规格或其他算子规格。
2. 每个函数参数在 `requires.arguments` 中单独出现。
3. 只依赖一个参数的条件放入该参数的 `constraints`。
4. 同时依赖多个参数的条件放入 `requires.relations`。
5. 无法简洁结构化的测试生成逻辑直接写入 `observed_domain_zh`，不隐藏在外部引用中。
6. YAML 的其余部分只保留算子身份和证据来源。

## 示例

```yaml
schema: operator-requires/v1
operator:
  id: ntops.torch.add
  name: add
  revision: 1
  category: binary_and_comparison
requires:
  observed_domain_zh: 两个同 shape 的浮点张量；alpha 来自标准高斯标量。
  arguments:
    input:
      kind: tensor
      constraints:
      - {op: rank_between, tensor: input, lower: 1, upper: 4}
      - {op: dtype_in, tensor: input, values: [float16, float32]}
      - {op: device_is, tensor: input, value: cuda}
      - {op: all_dims_between, tensor: input, lower: 1, upper: 1024}
      - {op: generated_by, tensor: input, generator: torch.randn}
    other:
      kind: tensor
      constraints:
      - {op: rank_between, tensor: other, lower: 1, upper: 4}
      - {op: dtype_in, tensor: other, values: [float16, float32]}
      - {op: device_is, tensor: other, value: cuda}
      - {op: all_dims_between, tensor: other, lower: 1, upper: 1024}
      - {op: generated_by, tensor: other, generator: torch.randn}
    alpha:
      kind: scalar_or_shape_parameter
      default: 1
      constraints:
      - {op: generated_by, value: alpha, generator: "random.gauss(0, 1)"}
    out:
      kind: optional_output_buffer
      default: null
      constraints:
      - {op: equals, left: out, right: null}
  relations:
  - {op: same_shape, values: [input, other]}
  - {op: same_dtype, values: [input, other]}
evidence:
  status: inferred_from_tests
  source:
    repository: https://github.com/InfiniTensor/ntops.git
    commit: 9ae4166ad342e4745f0eed13a5a20d069e994fc0
    license: Apache-2.0
  tests:
  - {path: tests/test_add.py, name: test_add, line: 11}
  reviewed: true
```

`default` 记录接口默认值，不表示该参数只能取默认值。带 `optional: true` 的参数，其张量
约束仅在参数非空时适用。空的 `constraints` 表示该参数由 `requires.relations` 中的跨参数
谓词约束；测试会拒绝既没有自身约束、也没有出现在跨参数谓词中的参数。

## 证据边界

`inferred_from_tests` 表示输入域来自固定版本上游测试。它只回答“这些测试生成了怎样的合法
调用”，不声称列出了算子的所有合法输入，也不构成程序等价性的形式化证明。

原始抽取结果保存在 `specs/ntops_test_evidence.json`，但算子 YAML 不依赖该文件才能解释其
输入条件；JSON 仅用于重新生成和审计证据。
