# 参数化 Add 验证细节

## 目标

参数化样例比较两份 raw TT IR：

- `parametric_2d_add.ttir`：二维 shape 参数 `a,b`，256 lane；
- `parametric_1d_add.ttir`：线性 numel 参数 `c`，128 lane。

PairSpec 量化正 signed-i32 参数并声明：

```text
a > 0 and b > 0 and c > 0 and a*b = c
```

目标是在该域内证明 `c` 个输出元素均满足：

```text
Output[k] = Input[k] + Alpha * Other[k]
```

## 两侧物理结构

二维侧把 `k` 分解为 row/column 并用 row-major stride 形成地址，launch 为：

```text
programs_lhs = ceildiv(c,256)
index_lhs    = pid*256 + lane
row          = index_lhs div b
column       = index_lhs rem b
offset_lhs   = row*b + column
mask_lhs     = row<a and column<b
```

一维侧直接使用：

```text
programs_rhs = ceildiv(c,128)
index_rhs    = pid*128 + lane
offset_rhs   = index_rhs
mask_rhs     = index_rhs<c
```

两侧 Alpha ABI 不同：左侧按值 `arg2`，右侧为 `arg1[0]` 的 scalar block load。Input、
Other 和 Output 的物理 block 也保留 side 区分。

## SMT 义务

ETV 对任意 `0<=k<c` 令：

```text
lhs_pid=k div 256, lhs_lane=k rem 256
rhs_pid=k div 128, rhs_lane=k rem 128
```

随后以反例查询证明：

1. `ceildiv(c,lanes)` 定义且覆盖 `k`；
2. 两侧 writer 唯一；
3. `mask_lhs` 和 `mask_rhs` 在合法 `k` 上为真；
4. `row*b+column = k`，其中 `row=k div b`、`column=k rem b`；
5. 两侧 Input/Other load 地址均对应逻辑 `k`；
6. Alpha 的 value scalar 与 scalar-block index 0 对应；
7. 两侧 Output store 地址均对应逻辑 `k`；
8. 中间 signed-i32 表达式满足当前定义性条件。

每个义务都查询其否定；`unsat` 才准入对应关系规则。若删除 `a*b=c`，二维 mask 和
线性观察域之间不再有充分关系，验证不能返回 `PROVED`。

## 进入 egglog 前的根

两侧不会预先被改写成同一表达式。概念上初始根为：

```text
observe_store(lhs:arg3, lhs_offset(k), lhs_mask(k),
  fadd(load(lhs:arg0,...), fmul(input(lhs:arg2), load(lhs:arg1,...))))

observe_store(rhs:arg3, rhs_offset(k), rhs_mask(k),
  fadd(load(rhs:arg0,...), fmul(load(rhs:arg1,0,...), load(rhs:arg2,...))))
```

side、物理 block、地址、mask 和 scalar ABI 均不同，因此仅 hash-cons 不会误判等价。

## 事实派生规则

SMT 义务通过后生成：

- 两侧 Input load -> `read(Input,k)`；
- 两侧 Other load -> `read(Other,k)`；
- 左侧 scalar 和右侧 scalar block -> `input(Alpha)`；
- 两侧 store -> `observe_store(Output,k,true,value)`。

这些规则与交换律、结合律等代数规则一起进入同一个 egglog ruleset。关系规则先运行，
代数规则随后运行；最终两侧完整 store 根同 e-class 才返回 `PROVED`。

## 与固定 TorchInductor 例的区别

`torch_inductor_add.ttir` 内部硬编码 `xnumel=128`，只能支持固定例，不能借助 PairSpec
将其错误外推为任意 `c`。因此参数化右侧使用专门的符号一维 TT IR。两类样例分别证明：

- 固定例：真实 ntops 与真实 TorchInductor 产物的具体 specialization；
- 参数化例：双 TT IR 前端、布局关系、symbolic launch 和证明后端的通用能力。

## 运行

```bash
python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

成功报告应包含两侧前端均为 `libtriton`、参数域检查完整、事实关系规则实际使用，且
顶层状态为 `PROVED(OBSERVABLE_MEMORY_EQUIVALENT)`。

## 限制

该例固定 rank 为 2 对 1，只量化维度值；不支持动态 rank。当前证明使用
`ABSTRACT_FLOAT`，也不覆盖乘加融合的 IEEE-754 舍入差异、allocation bounds、并发、
循环或归约。
