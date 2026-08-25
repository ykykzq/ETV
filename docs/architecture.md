# 系统架构

## 分层设计

ETV 的正式输入是两份 raw TT IR 和一份 PairSpec。前端、共同语义表示与证明数据结构
是三个不同层次：

```text
文件层            解析层             语义层             证明层
lhs/rhs.ttir  ->  libtriton IR  ->  ETV Program/Expr  ->  SMT + egglog e-graph
pair.json     ->  PairSpec      ->  facts/contract    ->  proof obligations
```

这一区分很重要：内部 IR 不是 e-graph。`Program`、`StoreTemplate` 和 `Expr` 是不可变、
类型化的程序语义；e-graph 是验证过程中临时建立的等价类数据结构。同一内部 IR 表达式
可以被有限枚举器、SMT 编码器、子图划分器和 egglog 后端共同消费。

## 双 TT IR 前端

`etv.schema.load_pair_spec` 强制以下条件：

1. `lhs`、`rhs` 都指向 `.ttir` 或 `.mlir`；
2. `frontends.lhs.kind` 和 `frontends.rhs.kind` 都显式为 `ttir`；
3. 两侧都声明入口 `function`；
4. 两侧都声明 host launch 的 `programs` 表达式。

TT IR 没有完整 host launch 信息，因此 ETV 不根据 kernel 内容猜 grid。缺失
`programs` 会以 `TTIR_LAUNCH_REQUIRED` 拒绝；非 TT IR 输入会以
`TTIR_PAIR_REQUIRED` 拒绝。

`etv/ttir/libtriton.py` 精确要求 Triton 3.7.1。它注册 Triton/MLIR 方言，通过
libtriton 解析文件、执行 module verifier，并遍历完整 operation/operand/result 图，
生成 `TTIRModule` 快照。快照保留类型、属性、block、region、location 和规范化 assembly，
可用于诊断；语义提升不依赖正则表达式重新解析原文件。

`etv/ttir/lift.py` 将已验证快照提升到共同 IR。目前建模的核心子集包括：

- `tt.get_program_id`、`tt.make_range`、`tt.splat`、`tt.broadcast`；
- `tt.addptr`、`tt.load`、`tt.store`；
- `arith.constant`、整数算术/比较、浮点逐点算术、`arith.select`；
- `math.sqrt`、`math.rsqrt` 等已列入语义的纯操作。

libtriton 可以完整解析但提升器尚不支持的 region、循环、归约或副作用会返回
`UNKNOWN`，不会绕过或猜测语义。

## 内部语义 IR

`etv/ir.py` 定义：

- `Sort`：`int`、`bool`、`abstract_float`；
- `Expr`：类型化不可变表达式；
- `StoreTemplate`：逻辑索引、物理 offset、mask 和 value；
- `Program`：入口、launch program 数、lane 数和 store 集合。

典型 load 表达式为：

```text
load(side:physical_block, offset(pid,lane), mask(pid,lane), default)
```

典型输出观察根为：

```text
observe_store(side:physical_output, offset(k), mask(k), value(k))
```

两侧物理名称和 side 标签始终保留。不能仅因为两个变量都叫 `arg0` 就视为相同输入；
它们必须经 PairSpec 角色映射和已准入关系规则连接。

为提高证明内核测试速度，`load_internal_pair_spec`、`load_program` 和
`verify_internal_spec` 可读取 `etv-semantic-program-v1` 测试夹具。这是 Python 内部
测试 API，不由 CLI 暴露，也不是第三种生产前端。

## PairSpec 与事实

PairSpec 提供 TT IR 本身缺少的跨程序关系：

- `roles`：物理 block/scalar/scalar-block 到逻辑 Input/Other/Alpha/Output；
- `bindings` 与 `side_bindings`：常量、shape、stride、numel 等参数；
- `parameters` 与 `constraints`：例如正 i32 的 `a,b,c` 以及 `a*b=c`；
- `disjoint`：输入与输出的 no-alias 前提；
- `contract`：观察哪个角色、观察多少逻辑元素、是否要求完整覆盖；
- `limits`：饱和迭代、e-node 和超时上限。

事实是证明前提，不是从程序中“推测”出来的结论。报告逐项记录其来源为
`TRUSTED_AXIOM`。

## 参数化与内存义务

参数化路径使用任意逻辑输出 `k`，并对每一侧写成：

```text
pid  = k div lanes
lane = k rem lanes
0 <= k < output_numel
```

Z3 检查参数域非空、整数定义性/无溢出、launch 覆盖、mask、唯一写入、地址范围和
两侧对应关系。通过后生成三类关系规则：

1. scalar role：物理标量或 scalar block 读取映射为逻辑标量；
2. load：满足地址和 mask 条件的物理读取映射为 `read(LogicalRole,k)`；
3. store：物理观察映射为规范化 `observe_store(Output,k,true,value)`。

SMT 证明的是规则条件，不直接宣布最终程序等价。规则必须随后在 egglog 中匹配。

## 等式饱和

`etv/egraph.py` 使用 `egglog==13.2.0`。每个整图或子图义务的过程是：

1. 插入两侧未归一化根；
2. 饱和事实派生的 scalar/load/store 关系规则；
3. 若根未合并，饱和经 Z3 准入的代数规则和 PairSpec 规则；
4. 若显式启用 LLM，可生成带 fact gate 的候选规则并重新准入；
5. 只有两侧根属于同一 e-class 才完成计算证明。

报告分别记录规则准入、逐规则 match 数、是否实际使用、每阶段 e-node/e-class 数与
未验证规则警告。e-graph 负责等价类和同余闭包；它不替代内部 IR、SMT 或内存模型。

## 子图划分

当 `llm.enabled=true` 且 `partition.enabled=true` 时，划分器把两侧完整内部 Program、
PairSpec 和计算根发送给 LLM 一次。模型只提出语义对应的左右路径。ETV 确定性检查：

- 每个 root family 完整覆盖；
- 同侧路径唯一且只嵌套或互不相交；
- 左右 sort 一致；
- 左右父子依赖拓扑一致；
- 分区 DAG 无环且数量在限制内。

通过检查后按子到父建立独立 e-graph。只有子图已证明，父图才能把它替换成共同的
类型化边界输入。提案无效或局部证明失败时回退整图验证，LLM 不直接提供等价公理。

## 结果和信任边界

正式报告包含输入及哈希、libtriton 版本、全部前提、SMT 查询结果、规则准入/应用、
egglog 状态、可选划分审计和反例。当前可信计算基包括：

- PairSpec 声明的角色、shape、launch 和 no-alias 事实；
- libtriton parser/verifier；
- TT IR 到 ETV IR 提升器；
- ETV 整数/内存语义与 SMT 编码；
- Z3、egglog 及 ETV 的 term 编码；
- 实际应用且未被形式验证的可信规则。

当前没有独立 proof certificate。详细判定流程见[完整验证过程](verification_process.md)。
