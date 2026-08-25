# 语义与信任模型

## 被验证的命题

对于固定的 `PairSpec` 假设 `P`，ETV 的 `PROVED` 结果表示：

```text
对于逻辑输入/标量值以及初始内存 M0 的每一种赋值，
如果 P 成立，那么
  两个有限的 Semantic TTIR 启动配置在 MVP 语义中都有定义，
  契约要求的每个 Output 元素都恰好写入一次，
  并且最终可观察的 Output 内存相等。
```

这是有界翻译验证：启动维度、形状、步长和向量宽度均由 spec 固定。逻辑输入内存中的加载值保持符号形式，因此证明对所有值成立，而不是一次有限数值测试。

## 数值语义

### 整数与布尔层

- 整数变量和中间值使用有符号 i32。
- `idiv` 和 `irem` 向零截断，与 MLIR 有符号除法一致。
- 如果具体中间值超出有符号 i32 范围，不会在无提示的情况下按无界整数解释；结果为 `UNKNOWN(INTEGER_OVERFLOW)`。
- 对每一个枚举的 lane，掩码和地址都会被完全求值。

该有限求值证明当前 `(M,N,X)=(8,16,128)` 的扁平化用例；它不是关于任意维度的参数化定理。

### `ABSTRACT_FLOAT`

浮点叶节点的取值范围是精确数学值。操作按照普通算术解释：

```text
fadd(a,b) = a + b
fsub(a,b) = a - b
fmul(a,b) = a * b
fdiv(a,b) = a / b，在有定义时成立
fma(a,b,c) = a*b + c
```

规则准入求解器使用数学实数。在等式饱和前，ETV 以结构化方式检查定义性：`fdiv` 的分母必须是可证明非零的精确常量，`fsqrt` 的被开方数必须是可证明非负的精确常量，`frsqrt` 的被开方数必须是可证明为正的精确常量。符号定义域会产生 `UNKNOWN(FLOAT_DEFINEDNESS_NOT_PROVED)`。未来的 `a/sqrt(b) == a*rsqrt(b)` 规则必须携带并证明 `b > 0`，然后才能准入。

该模式有意忽略舍入、NaN、无穷大、有符号零、非规格化数行为、上溢/下溢、融合与非融合舍入的区别、硬件内建函数近似以及归约顺序。每份报告都会重复说明该限制。

## 内存语义

PairSpec 将物理参数映射到逻辑角色（`Input`、`Other`、`Alpha`、`Output`）。Load 被归一化为：

```text
read(logical_block, concrete_element_offset)
```

Store 表示一次内存 token 状态转换：

```text
store(M0, Output, offset, symbolic_value, mask) -> M1
```

在该模型中，逻辑 block 是从有符号 i32 元素偏移到值的全映射。PairSpec v1 不包含 buffer 分配大小，因此即使每个已求值地址都是合法 i32，`PROVED` 也不表示已经证明 GPU 边界安全。

MVP 要求每一侧只有一个输出 store 模板，所有必要 block 均声明为互不重叠，并且有效输出地址必须是单射的。被掩码关闭的 lane 不会改变内存。因此，逻辑域、地址、值和初始内存都相等时，可以推出最终 Output 内存相等，并由框架条件保证其他 block 不变。

## 结果状态

| 状态 | 含义 |
| --- | --- |
| `PROVED` | 所有必要证明义务都已在列出的假设下完成。 |
| `DISPROVED` | 有限地址/掩码见证或具体抽象值模型证明两者不相等。 |
| `UNKNOWN` | 因为缺少事实、模型、规则或预算，命题尚未被判定。 |

`UNKNOWN` 不是把证明失败编码成不等价。反过来，通过随机测试或性质测试也绝不会被提升为 `PROVED`。

## 证据等级

| 等级 | 用途 |
| --- | --- |
| `BOUNDED_EXHAUSTIVE` | 对固定启动域进行完整枚举。 |
| `STRUCTURAL` | 通过语法导向方法完成定义性证明义务。 |
| `CONGRUENCE` | 由相同操作符和相等 e-class 子节点推导父节点相等。 |
| `ALGEBRAIC` | 通过 Z3 UNSAT 检查准入的抽象实数重写。 |
| `TRUSTED_AXIOM` | Pair 角色、绑定、no-alias 和数值抽象。 |
| `EMPIRICAL` | 仅用于测试的证据；绝不触发生产环境合并。 |

## 可信计算与输入边界

当前结果信任以下内容：

1. 从 raw TTIR 到 Semantic TTIR JSON 的显式提升；
2. PairSpec 中的角色对应关系、固定绑定和 no-alias 声明；
3. 求值器和内存 token 实现；
4. Z3 实数算术结果和 e-graph 实现；
5. 声明的 `ABSTRACT_FLOAT` 解释。

当前最大缺口是第 1 项。只有实现一个能够解析并验证 raw TTIR、保留源码位置的 Triton/MLIR 适配器后，ETV 才能声称完成端到端 TTIR 证明。
