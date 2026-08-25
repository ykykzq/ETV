# 参数化 Add 验证全过程

本文逐步记录 `examples/add/pair_parametric.json` 的验证。输入是九齿侧 raw TTIR、
Torch 侧 Prims 图和 PairSpec；目标是在固定 rank、任意正 i32 维度且 `a*b=c` 时证明：

```text
output = input + alpha * other
```

## 1. 原始输入

| 文件 | 含义 |
| --- | --- |
| `ttir/parametric_2d_add.ttir` | 九齿形态的二维 TTIR，256 lanes/program |
| `prims/torch_add.prims.json` | Torch Prims：broadcast、mul、add |
| `pair_parametric.json` | 参数、launch、角色、指针关系、no-alias 和输出契约 |

PairSpec 声明：

```text
a,b,c in signed-i32
a >= 1, b >= 1, c >= 1
a*b = c

lhs.arg4 = a
lhs.arg5 = b
rhs.torch_dim0 = a
rhs.torch_dim1 = b

lhs programs = ceildiv(c,256)
output_numel = c
```

角色：

```text
Input  : lhs.arg0 <-> rhs.input
Other  : lhs.arg1 <-> rhs.other
Alpha  : lhs.arg2 <-> rhs.alpha[0]
Output : lhs.arg3 <-> rhs.output
```

`Input/Other/Output` 两两不别名。以上事实是条件证明的前提，不是从程序名或参数顺序
猜测出来的结论。

## 2. 两侧转换为共同 IR

### 2.1 TTIR 侧

libtriton 3.7.1 完成 parse、方言注册和 IR verify。提升后的关键表达式为：

```text
logical = pid*256 + lane
mask    = logical < a*b
offset  = (logical div b)*b + (logical rem b)

value = fadd(
  load(arg0, offset, mask, 0),
  fmul(scalar(arg2), load(arg1, offset, mask, 0)))

store(arg3, offset, mask, value)
```

### 2.2 Prims 侧

严格 JSON 中的节点是：

```text
broadcast_alpha = prims.broadcast_in_dim(alpha, [torch_dim0,torch_dim1], [])
scaled_other    = prims.mul(broadcast_alpha, other)
result          = prims.add(input, scaled_other)
```

Prims 提升器按 row-major 逻辑索引构造输入 load。rank 固定为 2，两个维度值由
PairSpec 绑定为 `a,b`。内部逻辑域为：

```text
programs = a*b
lanes = 1
logical = pid
store(rhs.output, pid, true, result(pid))
```

这只是共同 IR 的逻辑张量域，不是 Torch GPU launch。

## 3. 参数符号化

令任意输出位置为 `k`：

```text
K(k) := 0 <= k < c
```

每侧 canonical writer 为：

```text
pid_side  = k div lanes_side
lane_side = k rem lanes_side
```

定义缩写：

```text
qL(k) = (k div 256)*256 + (k rem 256)
oL(k) = (qL(k) div b)*b + (qL(k) rem b)
mL(k) = qL(k) < a*b

qR(k)   = k div 1
rowR(k) = (qR(k) div b) rem a
colR(k) = qR(k) rem b
oR(k)   = rowR(k)*b + colR(k)
```

这些表达式没有在进入 e-graph 前被求成某个具体 shape 的整数。

## 4. SMT 证明的职责

验证器先构造参数域：

```text
D := ranges(a,b,c) and a*b=c and required signed-i32 definedness
```

第一次查询检查 `D` 可满足，结果为 `SAT`，排除真空证明。之后每项查询形式为：

```text
D and counterexample
```

本次报告共 45 次 Z3 调用：1 次 `SAT` 的非空域检查，44 次必要反例查询均为
`UNSAT`。它们覆盖：

- `c` 始终为正且定义；
- 两侧 program 数合法；
- active logical index 在 `[0,c)`；
- canonical writer 对每个 `k` 有效且 active；
- logical writer 唯一；
- output address 单射、无竞争；
- 左右 output address 对应；
- 每个 load offset/mask 有定义；
- 每个被观察 load 的 mask 恒真；
- 每个 load 地址对应 canonical 逻辑位置 `k`；
- 两侧 store mask 恒真且 store address 对应 `k`。

关键变化是：这些 `UNSAT` 结果只允许相应规则进入 e-graph，不直接返回相同的 read
叶子，也不直接判定最终值相等。

## 5. 初始重写目标

代入 canonical writer 后，左根是：

```text
observe_store(lhs:arg3, oL(k), mL(k),
  fadd(
    load(lhs:arg0, oL(k), mL(k), 0),
    fmul(
      input(lhs:arg2),
      load(lhs:arg1, oL(k), mL(k), 0))))
```

右根是：

```text
observe_store(rhs:output, qR(k), true,
  fadd(
    load(rhs:input, oR(k), true, 0),
    fmul(
      load(rhs:alpha, 0, true, 0),
      load(rhs:other, oR(k), true, 0))))
```

注意五类实际差异仍存在：physical block、Alpha ABI、launch/lane 索引、二维 offset、
store mask。此时两个根不属于同一 e-class。

## 6. 本次新增的 8 条条件规则

### 6.1 左侧 load/scalar

```text
parametric_load_lhs_0:
  load(lhs:arg0,oL(k),mL(k),0) -> read(Input,k)

parametric_scalar_role_lhs_1:
  input(lhs:arg2) -> input(Alpha)

parametric_load_lhs_2:
  load(lhs:arg1,oL(k),mL(k),0) -> read(Other,k)
```

两条 load 规则要求角色映射成立、`mL(k)=true`、`oL(k)=k`，并由相应 UNSAT 查询
证明。scalar 规则来自 PairSpec 的 ABI 角色前提。

### 6.2 左侧 store

```text
parametric_store_lhs:
  observe_store(lhs:arg3,oL(k),mL(k),v)
    -> observe_store(Output,k,true,v)
```

条件是 `arg3 -> Output`、`mL(k)=true`、`oL(k)=k`。`v` 是 pattern 变量；该规则
不会绕过 value 子树的证明。

### 6.3 右侧 load

```text
parametric_load_rhs_0:
  load(rhs:input,oR(k),true,0) -> read(Input,k)

parametric_load_rhs_1:
  load(rhs:alpha,0,true,0) -> input(Alpha)

parametric_load_rhs_2:
  load(rhs:other,oR(k),true,0) -> read(Other,k)
```

Alpha 规则额外检查 scalar-block endpoint 的声明 offset 为 0。普通 load 规则利用
`torch_dim0=a`、`torch_dim1=b` 和 `a*b=c` 证明 row-major offset 对应 `k`。

### 6.4 右侧 store

```text
parametric_store_rhs:
  observe_store(rhs:output,qR(k),true,v)
    -> observe_store(Output,k,true,v)
```

其中 `qR(k)=k div 1=k` 的条件由 SMT 证明。

每条规则都同时出现在：

```text
proof.parametric_domain.fact_derived_rewrites
proof.rule_admission
proof.egraph.rule_application
```

前两个字段说明为什么允许使用，第三个字段说明是否真的匹配。

## 7. 17 条内建代数规则

验证器仍准入 `fadd_comm/fadd_assoc/fmul_comm/fmul_assoc/fadd_zero/.../fma_def`
共 17 条抽象实数规则。每条都由 Z3 Real 证明 `lhs != rhs` 不可满足。

本例的计算树在 load/scalar 规范化后结构已经相同，因此 fact-derived 阶段就闭合根，
没有启动 `ALGEBRAIC_REWRITES` 阶段。17 条规则的实际 match 为 0。这里“规则已证明”
不等于“规则已用于本次证明”。

## 8. e-graph 逐步变化

### 8.1 插入之前

```text
e-nodes   = 0
e-classes = 0
```

### 8.2 插入两侧完整根

两侧物理 load/store 和索引表达式均被保留：

```text
e-nodes   = 40
e-classes = 40
unmatched_root_pairs = 1
```

与旧实现不同，hash-consing 不会让两根提前相同。

### 8.3 fact-derived 第 1 轮

统一 ruleset 在第一轮匹配全部 8 条规则：

```text
parametric_load_lhs_0        x1
parametric_scalar_role_lhs_1 x1
parametric_load_lhs_2        x1
parametric_store_lhs         x1
parametric_load_rhs_0        x1
parametric_load_rhs_1        x1
parametric_load_rhs_2        x1
parametric_store_rhs         x1
```

概念上的合并顺序是：

1. 两侧 Input load 加入 `read(Input,k)` 所在 e-class；
2. 两侧 Alpha 表示加入 `input(Alpha)` 所在 e-class；
3. 两侧 Other load 加入 `read(Other,k)` 所在 e-class；
4. 同余闭包合并两侧 `fmul`；
5. 同余闭包合并两侧 `fadd`；
6. 两侧 store 各加入 canonical `observe_store(Output,k,true,value)`；
7. value 子类相同后，同余闭包合并最终 store 根。

egglog 同一轮批量执行匹配，因此报告不伪造逐条 wall-clock 顺序。第一轮结束：

```text
before: 40 e-nodes / 40 e-classes
after : 42 e-nodes / 34 e-classes
updated = true
unmatched_root_pairs = 0
```

报告只把这一轮的总量变化记为 `+2 e-node/-6 e-class`；具体 canonical 表示与 union
由 egglog 在同一 ruleset 中批量建立，不把聚合计数反推成虚构的逐条内部执行顺序。

### 8.4 fact-derived 第 2 轮

第二轮仍能 e-match 右侧规则，但没有产生新 union：

```text
before: 42 e-nodes / 34 e-classes
after : 42 e-nodes / 34 e-classes
updated = false
stop_reason = SATURATED
```

累计 match count 中部分右侧规则显示 x2，这是 egglog 每轮匹配统计，不表示等式被
破坏性地应用两次。图在第二轮没有变化。

最终：

```text
iterations = 2
phase = FACT_DERIVED_RELATIONAL_REWRITES
equivalent_root_pairs = 1
after_fact_rewrites.unmatched_root_pairs = 0
```

## 9. 为什么 load 等价现在是重写证明

旧路径先询问 SMT 两个 offset 是否相等，然后直接返回同一个
`read(Input,symbolic_offset)`。这样 e-graph 看不到物理 load，根会在零轮饱和时同余。

新路径中 SMT 结果只出现在规则 admission。e-graph 初始明确看到：

```text
load(lhs:arg0,oL,mL,0) != load(rhs:input,oR,true,0)
```

只有两条经过条件证明的规则都实际匹配后，它们才共享 `read(Input,k)` e-class。删除
任一必要 load/store 规则都会留下 unmatched 根，不能得到本次 `PROVED`。

## 10. 最终结论

必要 proof blocks 包括：

| Block | 证据 |
| --- | --- |
| `FRONTEND` | libtriton TTIR 检查与严格 Prims schema |
| `ABI` | PairSpec 角色前提 |
| `PARAMETER_DOMAIN` | 参数域 SAT、关系约束 |
| `INDEX/MASK/COVERAGE` | 参数化 SMT |
| `RACE_FREEDOM/ADDRESS` | 参数化 SMT |
| `LOAD` | 条件规则准入，最终由规则应用闭合 |
| `DEFINEDNESS` | 浮点结构检查 |
| `COMPUTE` | egglog congruence + 已应用关系规则 |
| `STORE` | 完整 observe_store 根相等 |

最终输出：

```text
PROVED parametric_ninetoothed_add_vs_torch_prims_add: OBSERVABLE_MEMORY_EQUIVALENT
```

该结论量化所有满足前提的 `a,b,c` 与所有 `ABSTRACT_FLOAT` 输入值。它不涵盖
IEEE-754 位级行为、动态 rank、循环/归约、多 kernel、buffer 边界或提升器自身的
形式正确性。

## 11. 复核命令

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric

jq '.proof.parametric_domain.checks' build/add_parametric/report.json
jq '.proof.parametric_domain.fact_derived_rewrites' build/add_parametric/report.json
jq '.proof.egraph.initial_state' build/add_parametric/report.json
jq '.proof.egraph.stats.iteration_trace' build/add_parametric/report.json
jq '.proof.egraph.rule_application' build/add_parametric/report.json
```

通用契约见[完整验证过程](verification_process.md)，输入来源见
[Add 验证](add_validation.md)。
