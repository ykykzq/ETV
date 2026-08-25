# Torch Prims 输入格式

Torch 侧输入使用 `etv-prims-program-v1`。该文件是 Torch Prims FX 图的严格、稳定
JSON 边界，验证器运行时不导入 PyTorch，也不调用 TorchInductor。

## 示例

```json
{
  "format": "etv-prims-program-v1",
  "name": "torch_add_prims",
  "inputs": [
    {"name": "input", "dtype": "float32", "shape": [{"var": "d0"}, {"var": "d1"}]},
    {"name": "alpha", "dtype": "float32", "shape": []},
    {"name": "other", "dtype": "float32", "shape": [{"var": "d0"}, {"var": "d1"}]}
  ],
  "nodes": [
    {
      "name": "broadcast_alpha",
      "op": "prims.broadcast_in_dim",
      "args": ["alpha"],
      "shape": [{"var": "d0"}, {"var": "d1"}],
      "broadcast_dimensions": []
    },
    {"name": "scaled", "op": "prims.mul", "args": ["broadcast_alpha", "other"]},
    {"name": "result", "op": "prims.add", "args": ["input", "scaled"]}
  ],
  "output": {"value": "result", "block": "output"}
}
```

PairSpec 显式选择前端：

```json
"frontends": {
  "lhs": {"kind": "ttir", "function": "kernel", "programs": {"var": "grid"}},
  "rhs": {"kind": "prims"}
}
```

Prims 图本身描述逻辑张量域，所以 `function` 和 `programs` 对 `kind=prims` 非法。
九齿 TTIR 不携带 host launch grid，仍须由 PairSpec 提供 `programs`。

## 固定 rank 与符号维度

`shape` 必须是 JSON 数组；数组长度就是 rank，验证期间不变化。数组元素可以是正
整数，也可以是整数表达式或符号变量。符号名称通过 PairSpec 的共享/单侧 binding
与参数关联：

```json
"side_bindings": {
  "rhs": {
    "d0": {"var": "a"},
    "d1": {"var": "b"}
  }
}
```

这里符号化的是维度值 `a,b`，不是 rank 2。输出元素数由 shape 的乘积构造，形成
Torch 参考程序的符号逻辑执行域。

## 支持的节点

当前支持：

| Prims op | 内部操作 | 条件 |
| --- | --- | --- |
| `prims.broadcast_in_dim` | 显式索引映射 | 必须声明结果 shape 和递增、无重复的维度映射 |
| `prims.add` | `fadd` | 两个输入 shape 完全一致 |
| `prims.sub` | `fsub` | 同上 |
| `prims.mul` | `fmul` | 同上 |
| `prims.div` | `fdiv` | 同上；定义域另行证明 |
| `prims.neg` | `fneg` | 一个输入 |
| `prims.sqrt` | `fsqrt` | 一个输入；定义域另行证明 |
| `prims.rsqrt` | `frsqrt` | 一个输入；定义域另行证明 |

隐式广播会被拒绝。输入图必须显式包含 `prims.broadcast_in_dim`，防止验证器猜测
Torch 广播规则。未知字段、重复 SSA 名称、前向引用、错误 arity、shape 不一致、
非法维度映射和未知 op 均产生输入错误。

## 提升结果

Prims 张量被解释为纯函数 `element(indices)`。连续 row-major 输入变为：

```text
load(physical_input, row_major_offset(indices), true, 0)
```

标量 tensor 是 rank 0 输入，广播后仍读取 offset 0。输出被表示为逻辑逐元素程序：

```text
programs = product(output_shape)
lanes    = 1
logical_index = pid
store_offset  = pid
```

这不是声称 Torch Prims 有 GPU launch，而是把张量函数嵌入 ETV 共同内部表示的逻辑
观察域。与九齿侧真实 `pid/lane` 的对应关系随后由 PairSpec 条件和重写规则证明。

## 生成边界

`tools/extract_add_pair.py` 在 PyTorch 2.8 的 `TorchRefsMode` 中使用 `make_fx` 捕获
Add，检查实际调用序列为：

```text
prims.broadcast_in_dim.default
prims.mul.default
prims.add.default
```

之后写出该 JSON，并把固定 trace 的 shape 槽位替换为可由 PairSpec 绑定的符号名称。
FX 文本和 JSON 都纳入 `provenance.json` 哈希。
