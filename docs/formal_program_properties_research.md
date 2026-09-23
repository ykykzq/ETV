# Triton IR 程序对的形式化表示、对齐、唯一写与循环边界

调研与仓库核对日期：2026-09-16。ETV 基线提交为
`309d01d93427fc557265de5a2f452c5da27b20d7`；本次没有修改 benchmark artifact。

## 结论先行

1. “两个程序能分别划分为子程序，且子程序存在语义配对”最适合表示为
   **带边界关系的程序片段 + alignment automaton/product program**。片段对应可以是一对一、
   多对一，或者一侧为空的 stuttering；候选对齐可以由 trace、LLM 或 e-graph 提出，但必须
   另行证明 coverage、拓扑一致性、局部 relational triple 和 alignment adequacy。
2. “每个程序只使用一次 `tt.store`”与“每个输出元素只被写一次”不是同一性质。前者在
   ETV 的 75 算子实际语料中为假：5,204 份 TTIR 中有 835 份包含 2 或 3 条 `tt.store`，
   涉及 10 个算子。后者是动态地址注入性加覆盖性，目前不能从指令数推出，也没有在完整
   75 算子语料上被现有 ETV 证明，因而不应作为无条件全局假设。
3. 75 算子语料中的循环全部是结构化 `scf.for`：共 1,622 个，未发现 `scf.while`、
   `scf.parallel` 或 `scf.forall`。121 个循环的 trip count 可由 IR 常量求出；1,501 个循环
   具有显式 SSA 上界和静态正步长，但 trip count 依赖运行时参数。因此“每个循环都有明确
   的有限控制边界”在合法 `scf.for` 语义及无溢出前提下成立；“每个循环迭代次数固定”不成立。

## 1. 文献给出的建模方向

| 工作 | 场所 | 对 ETV 最直接的启示 |
| --- | --- | --- |
| [Translation Validation for an Optimizing Compiler](https://people.eecs.berkeley.edu/~necula/Papers/tv_pldi00.pdf) | PLDI 2000 | 验证每一次具体转换的源/目标程序，而不是先证明整个编译器；与 ETV 的 program-pair 输入一致。 |
| [GPUVerify: A Verifier for GPU Kernels](https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2012/OOPSLA.pdf) | OOPSLA 2012 | 为 GPU kernel 给出操作语义，将 race/divergence 变成验证条件；唯一写也应建模为不同动态执行实例的地址冲突查询。 |
| [Verifying GPU Kernels by Test Amplification](https://goto.ucsd.edu/~rjhala/papers/verifying_gpu_kernels_by_test_amplification.html) | PLDI 2012 | trace 可以产生候选或在受控信息流条件下推广，但一次测试本身不能证明所有 shape、数据与调度。 |
| [Semantic Program Alignment for Equivalence Checking](https://theory.stanford.edu/~aiken/publications/papers/pldi19.pdf) | PLDI 2019 | 用 trace alignment 找候选，再构造 program alignment automaton，最后用不变量与 SMT 证明；允许 many-to-many path alignment。 |
| [egg: Fast and Extensible Equality Saturation](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation) | POPL 2021 | e-graph 适合紧凑表示等价 term 和搜索重写；它本身不证明内存覆盖、循环终止或 alignment adequacy。 |
| [Alive2: Bounded Translation Validation for LLVM](https://web.ist.utl.pt/nuno.lopes/pubs.php?id=alive2-pldi21) | PLDI 2021 | SMT 化 IR 语义很有效，但 bounded unrolling 必须明确标为有界结论，不能冒充任意循环次数的证明。 |
| [An Algebra of Alignment for Relational Verification](https://popl23.sigplan.org/details/POPL-2023-popl-research-papers/20/An-Algebra-of-Alignment-for-Relational-Verification) | POPL 2023 | 用 left-only、right-only、joint step 表示对齐，并把“对齐覆盖原程序全部行为”的 adequacy 与“product 满足目标性质”分开证明。 |
| [KestRel: Relational Verification Using E-Graphs for Program Alignment](https://doi.org/10.1145/3720474) | OOPSLA 2025 | 用 e-graph 和 algebraic realignment 表示大量 product program 候选，再用执行 trace 引导抽取；与 ETV 的 e-graph/LLM 划分最接近。 |

共同结论不是“找到相似子图就足够”，而是：

```text
candidate alignment discovery
        + alignment adequacy
        + local relational proof obligations
        + compositionality
        = global relational conclusion
```

## 2. Triton IR 的基础语义模型

### 2.1 程序、状态与事件

把一次 Triton kernel launch 表示为：

```text
P = (G, CFG, Entry, Exit, Ops, Roles)
```

- `G(theta)` 是 host launch grid；它不在 TTIR 内，必须来自 PairSpec/manifest。
- `theta` 是 shape、stride、scalar、pointer base 和其他合法输入参数。
- 状态 `sigma = (pc, rho, mu, pid, lane, iota)`：`rho` 是 SSA 环境，`mu` 是按逻辑
  storage role 区分的内存，`iota` 是嵌套循环迭代向量。
- 小步关系写为 `sigma --e-->_P sigma'`。纯计算的 `e = tau`；一次活动写产生
  `e = write(role, addr, value, site, pid, lane, iota)`；mask 为假时不产生 write event。

Triton 的一条 tensor [`tt.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
可同时对应许多 lane。官方语义还规定 mask 为假时该 lane 不写内存。因此静态 operation、
动态 operation instance 和实际 write event 必须区分。

对输出 role `O`，令：

```text
Trace(P, theta) = 程序在合法输入 theta 上的事件序列
Obs_O(P, theta) = Trace 结束后 O 的可观察内存
```

程序对的最终目标可写成条件观察等价：

```text
forall theta_L, theta_R.
  Pre(theta_L, theta_R)
  => Obs_O(P_L, theta_L) = Obs_O(P_R, theta_R)
```

这里的 `Pre` 必须显式包含 ABI role 映射、shape/stride、launch grid、alias/disjoint、数值语义
和合法域，不能把自然语言说明当成形式前提。

## 3. 性质一：子程序划分与语义对齐

### 3.1 程序片段

对程序图 `G_P = (V, E_ctrl, E_data, E_mem)`，一个片段定义为：

```text
F = (V_F, Entry_F, Exit_F, In_F, Out_F, Effect_F)
```

- `V_F` 是 operation/control-region 的集合。
- `In_F`、`Out_F` 是跨片段的 typed SSA/memory boundary。
- `Effect_F` 记录读写的 logical storage role 和地址关系。
- 片段必须 single-entry/single-exit，或显式使用 region summary；对 dataflow DAG 要求凸性，
  即片段内两点之间的依赖路径不能无声明地穿过片段外部。

一个合法划分 `Part(P) = {F_1, ..., F_n}` 至少满足：

```text
Disjoint:   V_i intersect V_j = empty                     (i != j)
Coverage:   union_i V_i = observable backward slice(P)
Boundary:   cross-fragment edges appear in In/Out/Effect
Quotient:   contracted fragment graph preserves control/data/memory edges
```

循环 region 应作为一个含不变量/summary 的整体片段，或者按 iteration relation 拆分；不能把
循环体当作无环表达式直接切开。共享 SSA 子表达式可以成为 boundary input，但 operation
ownership 仍应唯一。

### 3.2 静态配对与动态 alignment automaton

静态候选配对可写为：

```text
M subseteq (Part(P_L) union {epsilon}) x (Part(P_R) union {epsilon})
```

`epsilon` 表示一侧停顿；因此 `M` 支持一对一、多对一和 stuttering。只有一张配对表还不够，
因为它没有说明不同分支、不同循环次数下如何前进。更完整的证据是 alignment automaton：

```text
A = (Q, q_entry, q_exit, Delta, Inv)
Delta subseteq Q x Path_L x Path_R x Q
```

每条 transition `(q, pi_L, pi_R, q')` 同时执行两侧 path，其中任意一侧可以是 `epsilon`。
`Inv(q)` 是两侧 boundary state 的关系。对每条 transition 证明局部 relational triple：

```text
Inv(q) and Guard(pi_L, pi_R)
  => Exec(pi_L, pi_R) establishes Inv(q')
```

还需分别证明：

1. **Adequacy**：所有满足 `Pre` 的左右程序行为都能被 `A` 覆盖；形式上至少有
   `Traces(P_L) subseteq proj_L(Traces(A))` 和右侧对应条件，并排除无限 epsilon 前进。
2. **Exit**：`Inv(q_exit) => Obs_L = Obs_R`。
3. **Topology**：fragment quotient graph 的依赖、分支和循环回边由 automaton transition 保留。
4. **Local proof**：每个非空 fragment pair 的值、mask、地址与 memory effect 关系成立。

由 transition induction 得到：若 `A` adequate、所有局部义务成立且 exit invariant 蕴含观察
相等，则全局程序对满足条件观察等价。这正是 PLDI 2019 PAA 与 POPL 2023 alignment
adequacy 思路在 TTIR 上的实例化。

### 3.3 对 ETV 的具体表示建议

建议把当前 partition proposal 提升为可审计的 `AlignmentCertificate`：

```text
fragments:
  id, side, owned_ops, entry, exit, typed_inputs, typed_outputs, effects
alignment_edges:
  source_state, target_state, lhs_fragment_or_epsilon, rhs_fragment_or_epsilon
invariants:
  state_id, predicate_ids, relation
obligations:
  coverage, boundary, topology, adequacy, local_equivalence, exit
```

ETV 已经检查 root coverage、路径唯一/嵌套、sort、父子拓扑与 partition DAG 无环；这些是良好
基础，但不是一般控制流下 adequacy 的完整替代。建议沿用现有原则：LLM/trace/e-graph 只生成
certificate 候选，确定性检查器和 SMT/e-graph proof 才能关闭义务。e-graph 适合证明局部纯
表达式等价；控制流覆盖、内存 effect 和唯一写应留在关系/SMT 层。

## 4. 性质二：`tt.store` 次数与每元素唯一写

### 4.1 两个不同的性质

静态单 store 是一个简单语法性质：

```text
OneStoreOp(P) := |{ op in Ops(P) | name(op) = "tt.store" }| = 1
```

输出单 writer 是动态语义性质。对每个静态 store site `s`，定义其动态实例域：

```text
Inst_s(theta) = LaunchStep x PID(theta) x Lane_s x Iter_s(theta)
```

对 `u in Inst_s(theta)`，`active_s(theta,u)` 是 mask/control predicate，`role_s` 是输出逻辑
storage，`addr_s(theta,u)` 是 element address。输出元素 `k` 的 writer 集合为：

```text
W(P, O, theta, k) = {
  (s,u) | role_s = O and active_s(theta,u)
          and addr_s(theta,u) = base_O(theta) + k
}
```

真正需要的性质是：

```text
SingleWriter(P, O) :=
  forall theta in Legal, forall k in [0, Numel_O(theta)).
    |W(P, O, theta, k)| = 1
```

程序有多个输出 role 时，程序级性质是对每个 `O in OutputRoles(P)` 的上述性质取合取。

它可拆成：

```text
Coverage:  forall k in output domain. exists exactly one active writer
Injective: two active dynamic instances with the same role/address are identical
InBounds:  every active write to O lies in the declared output domain
```

`OneStoreOp` 既不是 `SingleWriter` 的充分条件，也不是必要条件：

- 一条 `tt.store` 会被多个 PID、lane 和 loop iteration 执行，它们可能地址冲突；
- 多条 `tt.store` 可以写不同输出 tensors，或写同一 tensor 的互不相交切片。

### 4.2 可交给 SMT 的冲突查询

对任意两个 store site `s,t` 建立两个独立动态实例变量 `u,v`：

```text
Collision(s,t) :=
  Pre(theta)
  and Domain_s(theta,u) and Domain_t(theta,v)
  and active_s(theta,u) and active_t(theta,v)
  and role_s = role_t
  and addr_s(theta,u) = addr_t(theta,v)
  and (s,u) != (t,v)
```

所有 `(s,t)` 的 `Collision` 都为 `UNSAT` 才证明注入性。覆盖性再检查是否存在合法 `k` 没有
writer；地址反演简单时可构造 canonical writer，复杂时使用量词、分段 Presburger 关系或
有限域枚举。pointer base 的相等/不等必须来自 ABI role 与 alias/disjoint predicate，不能按
SSA 名称猜测。`tt.atomic_*` 应作为单独 effect，不能当成普通唯一写。

### 4.3 ETV 75 算子实际核对

审计范围以 `specs/ntops/operators/*.yaml` 的 75 个名字为准；benchmark 中额外的 legacy
`matmul` 目录被排除。完整机器可读结果见
[`ttir_property_audit.json`](ttir_property_audit.json)，复现脚本见
[`audit_ttir_properties.py`](../tools/audit_ttir_properties.py)。

| 指标 | 结果 |
| --- | ---: |
| TTIR 文件实例 | 5,204 |
| 精确 SHA-256 不同的文件 | 5,189 |
| libtriton 3.7.1 parse + module verify | 5,204 / 5,204 |
| 恰有 1 条 `tt.store` | 4,369 (83.95%) |
| 恰有 2 条 `tt.store` | 428 (8.22%) |
| 恰有 3 条 `tt.store` | 407 (7.82%) |
| 总 `tt.store` operation | 6,446 |
| 位于 `scf.for` 内的静态 `tt.store` | 739 |
| `tt.atomic_*` operation | 0 |

包含多条 store 的 10 个算子如下。这里按 artifact 实例计数，不能解释为 835 个不同 kernel
算法；RHS reference candidate 和不同测试 specialization 都保留为独立 artifact。

| 算子 | 多-store TTIR | 每文件最大 store 数 |
| --- | ---: | ---: |
| argsort | 60 | 2 |
| batch_norm | 48 | 3 |
| instance_norm | 384 | 3 |
| max | 31 | 2 |
| median | 30 | 2 |
| quantile | 48 | 2 |
| rotary_position_embedding | 128 | 2 |
| scaled_dot_product_attention | 39 | 3 |
| softmax | 3 | 2 |
| sort | 64 | 2 |

所以字面性质 `OneStoreOp(P)` 已被实际 IR 反驳。对于语义性质 `SingleWriter(P,O)`，本次没有
发现 atomic operation 是有利证据，但多 store、loop 内 store、vector lane 和多 PID 仍使静态
计数不足以判定。结论应为 **未在完整语料上证实，而不是已证实或已反驳**。

现有 ETV 代码也不能填补这个证据缺口：TTIR lifter 在未指定 `store_index` 时要求恰好一个
store（`etv/ttir/lift.py:978`），fixed evaluator 和 parametric verifier 同样要求单 store
（`etv/evaluator.py:304`、`etv/parametric.py:672`）。参数化路径确实已有 canonical writer、
coverage 和 address injectivity 义务（`etv/parametric.py:997`），但仅作用于已经成功提升的
单-store、无 region 程序。仓库现有 76 目录 benchmark 报告（包含本次排除的 `matmul`）中，
1,099 个 observation 有 984 个 `UNKNOWN`，其中 706 个是
`TTIR_REGION_SEMANTICS_UNSUPPORTED`，不能据此概括全部 75 个算子。

## 5. 性质三：循环边界与固定迭代次数

### 5.1 应区分三种强度

对循环 `l`，定义：

```text
Bounded(l) := forall theta in Legal. exists B_l(theta) in Nat.
                every execution performs at most B_l(theta) iterations

UniformlyBounded(l) := exists C in Nat. forall theta in Legal.
                         iterations(l,theta) <= C

FixedTrip(l) := exists C in Nat. forall theta in Legal.
                  iterations(l,theta) = C
```

“边界清晰”建议精确定义为 `Bounded` 且边界表达式可被验证器编码；“迭代次数固定”对应
`FixedTrip`。不要把“IR 语法中有 upper bound operand”自动升级为 `UniformlyBounded`。

MLIR [`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scf-forop)
使用半开区间，要求 step 严格为正；官方定义的 trip count 是：

```text
N(theta) = max(0, ceil((UB(theta) - LB(theta)) / Step(theta)))
```

并要求 `LB + N*Step` 在 induction variable 类型内可表示，否则行为未定义。因此：

- `LB/UB/Step` 都可常量求值时，得到编译期 `FixedTrip`；
- `UB` 依赖 shape/runtime scalar、`Step > 0` 可证明时，得到参数化 `Bounded`；
- 还必须证明整数定义性、无除零、无溢出和 shape domain；
- `scf.while` 或一般 CFG 回边需要 ranking function/variant，不能套用上述公式。

### 5.2 75 算子实际核对

| 指标 | 结果 |
| --- | ---: |
| 含 `scf.for` 的 TTIR 文件 | 867 / 5,204 (16.66%) |
| 涉及算子 | 18 / 75 |
| `scf.for` operation | 1,622 |
| 编译期可求 trip count | 121 (7.46%) |
| 显式 SSA bound + 静态正 step，但 trip count 运行时化 | 1,501 (92.54%) |
| `scf.while` | 0 |
| `scf.parallel` / `scf.forall` | 0 |

| 算子 | `scf.for` | 固定 trip | 参数化 bound |
| --- | ---: | ---: | ---: |
| acosh | 42 | 0 | 42 |
| addmm | 2 | 0 | 2 |
| argsort | 58 | 58 | 0 |
| atan | 17 | 0 | 17 |
| batch_norm | 56 | 8 | 48 |
| bmm | 2 | 0 | 2 |
| instance_norm | 768 | 0 | 768 |
| layer_norm | 288 | 0 | 288 |
| logsumexp | 5 | 5 | 0 |
| max | 6 | 6 | 0 |
| mean | 16 | 1 | 15 |
| median | 29 | 29 | 0 |
| mm | 2 | 0 | 2 |
| rms_norm | 128 | 0 | 128 |
| rot90 | 29 | 0 | 29 |
| scaled_dot_product_attention | 164 | 8 | 156 |
| sgn | 4 | 0 | 4 |
| softmax | 6 | 6 | 0 |

由此得到两个不同答案：

- 若假设是“循环都具有结构化、显式、可写成上式的有限边界”，实际语料支持它；结论仍以
  合法参数、正 step 和无溢出为条件。
- 若假设是“所有循环的 trip count 都在编译期固定”，实际语料明确否定它；92.54% 的循环
  依赖运行时 SSA 值。manifest 中某次 capture 的具体实参只能固定那一次 invocation，不能
  证明同一 TTIR 对所有合法输入固定次数。

## 6. 建议的 ETV 落地顺序

1. **前端语义**：继续以 libtriton operation/region 图为事实源；审计脚本的文本统计只用于
   corpus 盘点，不进入可信证明内核。
2. **扩展内部 IR**：从单个 `StoreTemplate` 扩展为全部 `StoreSite`；增加
   `LoopSummary(lb, ub, step, trip_expr, iter_args)` 和嵌套 iteration vector。
3. **先证 memory well-formedness**：按 output role 对所有 store site 生成 collision、coverage、
   in-bounds、alias/disjoint 和 integer-definedness 查询。SAT 给出
   `(site,pid,lane,iteration)` 反例，UNSAT 才记录 single-writer。
4. **循环分级**：常量 trip、参数化 `scf.for`、需要 ranking invariant 的一般循环分别处理；
   bounded unrolling 只能返回带 bound 的结论或 `UNKNOWN`。
5. **构造 fragment quotient graph**：纯 dataflow region 可按 typed boundary 划分；loop、reduce、
   multi-launch 和 memory effect 形成不可随意穿越的显式节点。
6. **生成而非信任 alignment**：LLM、trace、KestRel 风格 equality saturation 都可提出
   `AlignmentCertificate`，随后机器检查 coverage/topology/adequacy，并逐 pair 证明局部关系。
7. **组合结论**：只有 memory、loop、alignment 和局部计算义务全部关闭才返回 `PROVED`；找到
   collision/值差异返回 `DISPROVED`；前提或语义不足返回 `UNKNOWN`。

## 7. 复现与审计边界

从 ETV 根目录运行：

```bash
.venv-ttir/bin/python tools/audit_ttir_properties.py \
  --verify-libtriton \
  --output docs/ttir_property_audit.json
```

审计快照的 corpus fingerprint 为
`b60500154388259e4683ae772bef34efed955759e6e573cbb97626ceda6f69bd`。

本次审计证明的是：指定 corpus 中的 operation 数、libtriton 可解析性、store/loop 的语法结构
以及常量传播可求的 trip count。它没有证明实际 GPU 上每个输出元素唯一写，也没有把一次
numeric pass、一次 capture 或一次 trace 推广成全输入定理。后两者必须通过上文的动态实例
地址义务和参数域证明完成。
