# 完整验证过程

本文同时定义输入契约、语义、等价判定、等式饱和和信任边界。正式目标为：在
PairSpec 前提 `P` 下，证明两份 TT IR 程序 `L`、`R` 对全部可观察输出地址产生相同
最终值：

```text
P => ObservableMemory(L) = ObservableMemory(R)
```

## 1. 输入验证

`load_pair_spec` 首先严格解析 `etv-pair-v2`，拒绝未知字段、重复物理端点、非法参数
域、未声明入口、缺失 launch 或非 TT IR 路径。生产约束为：

```text
metadata.frontends.lhs.kind = metadata.frontends.rhs.kind = ttir
metadata.lhs,metadata.rhs suffix in {.ttir,.mlir}
metadata.frontends.lhs.function and rhs.function are nonempty
metadata.frontends.lhs.programs and rhs.programs are integer expressions
```

v2 将 `metadata`、只供 LLM 阅读的 `assumptions`、形式 `predicates`、证明目标
`observation` 和外部 `rewrites` 分开。TT IR 不编码 host grid，`programs` 是必须信任的
外部配置。旧 v1 仍可读取，但会先归一化到相同的内部 `PredicateSet`。

## 2. libtriton 解析与快照

两份 TT IR 独立经过固定 Triton 3.7.1 的相同步骤：

1. 注册 Triton 和 MLIR 内置方言；
2. 解析整个 module；
3. 执行 module verifier；
4. 选择 PairSpec 指定的入口函数；
5. 遍历 operation/operand/result/block/region/attribute；
6. 生成带输入 SHA-256 和规范化 assembly 的 TTIR snapshot。

原始文本不通过正则表达式构成语义前端。正则只用于把 libtriton 已打印的单行 assembly
关联到快照节点，供诊断和当前有限属性读取；解析和 IR 合法性由 libtriton 决定。

## 3. TT IR 提升到内部 IR

提升器把入口参数表示为物理 scalar 或 block，把向量操作表示为“给定 lane 索引时的
元素表达式”。`program_id` 和 lane 保留为整数变量，指针由 block 与 offset 分离。

例如：

```text
%pid   = tt.get_program_id x
%index = %pid * 128 + tt.make_range(0,128)
%ptr   = tt.addptr %base, %index
%value = tt.load %ptr, %mask, %zero
```

提升为：

```text
load(base,
     iadd(imul(pid,128),lane),
     lt(iadd(imul(pid,128),lane),numel),
     0)
```

store 提升为 `StoreTemplate(logical_index, offset, mask, value)`。完整 Program 是普通不可变
语义对象，不包含等价类；后续有限求值、SMT、划分和 egglog 都读取它。

cast 不做前端恒等归一化。`arith.extsi/extui/trunci/index_cast/index_castui` 提升为
携带源/目标类型的 `sext/zext/trunc/index_cast/index_castui` 节点；
`arith.sitofp/uitofp` 同样保留为 `sitofp/uitofp[iN->fN]`。于是
`sext[i8->i32](x)`、`sitofp[i32->f32](x)` 与 `x` 都是不同的内部 IR term；除非显式规则
被准入并实际命中，egglog 不会合并它们。两侧完全相同的 cast 节点仍可由结构相等与
同余闭包处理。

无 `other` 的 masked `tt.load` 被提升为带 `undefined_float` 默认分支的 load，而不是
假定为零。固定规模路径仅在某个活动 store lane 实际走到该分支时返回
`TTIR_UNDEFINED_LOAD_LANE`；参数化路径则必须用 SMT 证明相应 load mask 在观察域恒真。
因此语义等价但 SSA 身份不同的 load/store mask 可以继续验证，同时越界读取不会被掩盖。

合法但未建模的 TT IR 返回 `UNKNOWN`，例如 region、循环、归约、原子操作或不支持的
类型。`UNKNOWN` 表示没有结论，而不是不等价。

## 4. 语义模式

当前只实现 `abstract_float`：

- 浮点常量解释为精确有理数；
- `fadd/fsub/fmul/fdiv/fma` 等解释为数学实数运算；
- 不模拟舍入、NaN、无穷、signed zero、flush-to-zero 或 fast-math；
- `cmpf` 的 ordered/unordered 差异在该不含 NaN 的实数域中消失；
- `maxnumf/minnumf/clampf` 降为实数比较和 select，`propagateNan` 仅支持 `none`；
- `div/sqrt/rsqrt/log/acosh/pow` 等部分函数必须证明定义域，否则返回 `UNKNOWN`；
- 其他已建模数学函数作为确定的纯函数保留，系统目前只使用结构相等和同余，不擅自加入
  超越函数代数恒等式。

整数用于地址和 shape。固定规模求值检查 signed i32 范围与除零；参数化路径通过 Z3
加入定义性/范围义务。显式 `iN` cast 按低位截断、符号扩展或零扩展解释，不会直接返回
operand。`index` 的位宽由 target data layout 决定，而当前 PairSpec 尚未声明该信息；
因此相关 cast 返回 `UNKNOWN`。除 cast 外，当前整数模型仍不是完整的 TT IR 位向量语义。

## 5. ABI 谓词与内存观察

同名物理参数不会自动对齐。PairSpec 在 `predicates.abi` 中声明：

```json
"Alpha": {
  "lhs": {"kind": "scalar", "name": "arg2"},
  "rhs": {"kind": "scalar_block", "name": "arg1", "index": 0}
}
```

这表示左侧按值标量和右侧指针下标 0 的读取具有同一逻辑角色。`block`、
`scalar_block`、`scalar` 的差异保留到关系规则实际应用。

当前内存语义只观察 `observation.output_role`。`require_disjoint` 中的角色必须被
`predicates.disjoint` 覆盖，否则无法排除写输出改变后续输入读取，结果为 `UNKNOWN`。
系统不证明实际 allocation 大小或越界安全。

## 6. 固定规模义务

当 PairSpec 没有符号参数时，ETV 枚举两侧全部 `program x lane`：

1. 对齐两侧 cast 结构；差异必须由已准入规则实际消解；
2. 求值 store 的 logical index、offset 和 mask；
3. 检查单个逻辑元素没有重复活动写；
4. 检查活动逻辑域恰好覆盖 `[0, output_numel)`；
5. 比较两侧每个逻辑元素的 output offset 和 mask；
6. 保留两侧 value 表达式进入计算等价证明。

地址或 mask 不同可直接给出具体反例。值表达式不同则交给 egglog；无法合并时，Z3
尝试构造抽象输入值模型，找到则为 `DISPROVED(COMPUTE_MISMATCH)`。

有限枚举结论只覆盖 PairSpec 中固定的 launch 和 binding，不外推其他 shape。即使某个 cast
在被枚举值上碰巧等于 operand，也不能跳过第 1 项；缺少规则时返回
`UNKNOWN(CAST_EQUIVALENCE_NOT_REWRITTEN)`。

## 7. 参数化义务

存在 `predicates.parameters` 时，ETV 对任意逻辑元素 `k` 建立两侧 writer：

```text
0 <= k < output_numel
pid  = k div lanes
lane = k rem lanes
```

SMT 依次检查：

- 参数/约束域可满足；
- launch program 数、index、offset 的整数定义性；
- 每个合法 `k` 存在且只有一个活动 writer；
- writer mask 在合法域为真；
- output offset 与逻辑地址一致；
- 所需 load 的 mask、地址和角色映射成立；
- 两侧 output 观察对应。

每项通过“寻找反例”的查询完成，`unsat` 表示该义务在全部参数域内成立。查询文本哈希、
Z3 版本和结论进入报告。

## 8. 谓词与重写规则准入

所有 ABI、binding、参数域、约束和 no-alias 信息在内部统一为 `PredicateSet`。它们是
条件验证的前提，不是验证器从程序中证明出的结论。`constraints` 与 `custom` 中的
`z3_expr` 使用当前参数化编码器支持的 Z3 整数/布尔理论；可编码只说明后续义务能在该
前提下求解，不说明前提本身已被证明。没有编码器的自定义谓词标成 `trusted`，只能
作为可信规则 gate。

`assumptions.for_llm` 与上述谓词完全分离：自然语言只帮助节点选择、规则候选生成和
子图划分，不能满足 v2 规则 gate，也不能直接改变证明状态。

内建规则位于代码维护的规则库。用户规则必须放入独立 `etv-rewrite-v1` 文件，并通过
PairSpec 的 `rewrites` 列表按本次运行加载。验证器记录文件 SHA-256，把来源强制标为
`user:<path>`，不修改 egglog 或全局规则库。

规则分三类：

### 代数规则

浮点代数规则被翻译为 Z3 Real 等式；显式 `iN` cast 规则被翻译为 Z3 Int 上的有限位宽
模数、符号扩展和零扩展公式。ETV 查询是否存在使左右不等的赋值，仅 `unsat` 才按
`ALGEBRAIC` 准入。因此无条件 `sext[i8->i32](x)=x` 会产生反例并被拒绝。依赖输入范围
才能成立的 cast 关系必须引用 PairSpec predicate 并走关系规则；`index` 位宽未知时通用
cast 规则也不会被形式准入。

### 事实派生关系规则

scalar/load/store 规则连接两侧物理表示。它们必须引用具体角色、地址、mask、shape 和
launch 事实，参数化时还要有 SMT 义务证据。它们不是通用代数恒等式。

### 未验证可信规则

布局、库语义或 LLM 提出的非代数规则如果尚无验证器，可在显式 predicate gate 和当前策略
下作为 `TRUSTED_AXIOM` 准入。这是已知缺陷。报告记录：声明、事实检查、来源、是否匹配、
使用次数和“未经形式验证”的警告。未实际使用的可信规则不污染最终证明等级；实际
使用则进入 `trusted_axioms` 和 `does_not_prove`。

LLM 不能添加无条件可信规则，不能修改谓词，也不能直接合并根。

## 9. egglog 等式饱和

ETV 把内部 IR 表达式编码为 egglog term。一次义务按阶段执行：

```text
insert(lhs_root, rhs_root)
run(predicate-derived relational rules)
if roots differ: run(validated builtin/user rules)
if roots differ and LLM enabled: admit and run LLM candidates
check(same_eclass(lhs_root, rhs_root))
```

所有当前可能导致等价的情况在统一 e-graph 内考虑：代数形态、角色对应、scalar-block
读取、不同 layout 的地址映射、不同 lane/launch、mask 和完整 store 观察。同余闭包会
把已合并子表达式传播到父表达式。

cast operator 与类型 payload 是 egglog term 的组成部分。规则库默认没有
`cast(x) -> x`；固定规模路径另有 `CAST` 义务和 `proof.cast_rewrites` 应用日志，保证有限
枚举不会在规则之外把 cast 当作恒等。参数化路径则用精确 cast SMT 公式证明地址条件，
并把结论封装为随后必须在 egglog 中匹配的关系规则。

饱和受 `max_iterations`、`max_enodes`、`timeout_ms` 限制。耗尽资源返回 `UNKNOWN`，
不会以“没有找到证明”推断不等价。

## 10. 子图划分

可选 LLM 划分发生在计算证明前。模型一次扫描左右完整 Program、形式谓词、自然语言
assumptions 和所有根，返回成对
表达式路径。ETV 重新计算覆盖、唯一性、sort、左右父子拓扑和 DAG。通过后：

1. 先验证叶子子图；
2. 已证明子图替换为左右同名类型化 boundary input；
3. 验证父图；
4. 所有根分区证明后用 `DECOMPOSITION + CONGRUENCE` 组合。

非法提议、API 失败或局部义务未证明都会回退整图验证。LLM 的语义标签不是证明。

## 11. 最终状态

`PROVED` 要求：

```text
输入/语义/ABI/定义域/覆盖/地址/mask 必要义务全部关闭
and
(所有完整 observe_store 根在同一 e-class
 or 所有依赖有序子图义务分别证明并完成组合)
```

`DISPROVED` 要求具体反例，如输出地址、活动 mask 或抽象输入值使两侧结果不同。
`UNKNOWN` 用于输入错误、语义未实现、事实不足、规则不足、定义域未证明或资源耗尽。

## 12. 信任边界与保证

`PROVED` 在声明的 PairSpec 谓词前提和 `ABSTRACT_FLOAT` 语义下建立可观察输出内存
等价。报告中的 `soundness.level=formal_under_declared_predicates` 表示证明条件化于这些
谓词；如果实际应用未验证规则，则降为 `conditional_on_unverified_rewrites`。顶层
`PROVED` 不能脱离 `soundness`、规则应用日志和输入哈希单独解释。
它不证明：

- TT IR 到 ETV IR 提升器本身的形式正确性；
- libtriton、Z3、egglog 或 ETV 编码无缺陷；
- PairSpec 角色、shape、launch、no-alias 事实真实；
- 未验证可信规则正确；
- IEEE-754/位级 GPU 等价；
- allocation bounds、并发、shared memory 或多 kernel 行为。

因此报告必须与输入哈希、事实、规则应用日志和信任警告一起解释，不能只读取顶层状态。
