# 输入格式

ETV 接受严格 JSON。选择 JSON 是为了获得确定性解析与产物；它同时也是合法的 YAML 子集。未知键会被视为错误。

## 语义程序

格式标识符：`etv-semantic-program-v1`。

```json
{
  "format": "etv-semantic-program-v1",
  "name": "linear_add",
  "launch": {
    "programs": {"op": "ceildiv", "args": [{"var": "X"}, 128]},
    "lanes": 128
  },
  "stores": [{
    "block": "out_ptr0",
    "logical_index": {"op": "iadd", "args": [
      {"op": "imul", "args": [{"var": "pid"}, 128]},
      {"var": "lane"}
    ]},
    "offset": {"op": "iadd", "args": [
      {"op": "imul", "args": [{"var": "pid"}, 128]},
      {"var": "lane"}
    ]},
    "mask": {"op": "lt", "args": [
      {"op": "iadd", "args": [
        {"op": "imul", "args": [{"var": "pid"}, 128]},
        {"var": "lane"}
      ]},
      {"var": "X"}
    ]},
    "value": {"op": "load", "block": "in_ptr0",
              "offset": {"op": "iadd", "args": [
                {"op": "imul", "args": [{"var": "pid"}, 128]},
                {"var": "lane"}
              ]},
              "mask": true, "default": {"float": "0"}}
  }]
}
```

`pid` 和 `lane` 是求值器提供的变量。其他所有整数变量都必须在 PairSpec 中绑定。

## 配对规格

格式标识符：`etv-pair-v1`。

必需部分：

- `lhs`、`rhs`：相对于 spec 文件的 Semantic 程序路径。
- `semantic_mode`：当前仅支持 `abstract_float`。
- `roles`：两个物理 ABI 的逻辑角色对应关系。
- `facts.bindings`：固定的整数形状、步长和启动配置值。
- `facts.assumptions`：会写入报告的人类可读可信前提。
- `facts.disjoint`：其中逻辑 block 成对不相交的分组。
- `contract`：可观察输出、大小、覆盖性和所需的 no-alias 角色。
- `limits`：正数形式的 e-graph 迭代、节点和超时预算。

角色端点有三种类型：

```text
block         张量/内存 block
scalar        kernel 标量参数
scalar_block  从映射 block 的固定索引处读取的标量
```

把 lhs 的 `scalar` 和 rhs 的 `scalar_block` 映射到同一个逻辑角色，即可实现文档中 ntops 与 Inductor 之间的 Alpha ABI 归一化。

## 表达式语言

| 类型 | 操作 |
| --- | --- |
| i32 | `iadd`、`isub`、`imul`、`idiv`、`irem`、`ceildiv` |
| bool | `lt`、`le`、`gt`、`ge`、`eq`、`ne`、`and`、`or`、`not` |
| 抽象浮点 | `fadd`、`fsub`、`fmul`、`fdiv`、`fneg`、`fsqrt`、`frsqrt`、`fma` |
| 多态 | `select` |
| 内存/输入 | `load`、`scalar` |

整数和布尔字面量使用 JSON 字面量。抽象浮点常量使用精确的有理数字符串：

```json
{"float": "0"}
{"float": "1/6"}
```

`load` 包含 `block`、`offset`、`mask` 和 `default`。在该 MVP 中，默认值和加载值都是抽象浮点数。

`fdiv`、`fsqrt` 和 `frsqrt` 是偏函数。只有当分母/被开方数可以约简为满足条件的精确常量时，MVP 才会证明其定义域；否则结果为 `UNKNOWN(FLOAT_DEFINEDNESS_NOT_PROVED)`。人类可读假设不能绕过该检查。

完整契约请参阅 `examples/specs/add_proved.json` 及其引用的两个程序。
