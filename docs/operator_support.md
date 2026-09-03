# 算子语义支持

本文描述 Triton 3.7.1/libtriton 已解析的 MLIR operation 中，哪些已经具备 ETV 内部
语义。这里必须区分三个结论：

1. libtriton parse/verify 成功，只说明输入是合法的已注册 MLIR/Triton 方言；
2. ETV lift 成功，说明所有可达 operation 都已转换为共同语义 IR；
3. 最终 `PROVED` 还要求 PairSpec、mask、覆盖、地址、定义域和等式饱和义务全部关闭。

因此“支持某个 op”不等于“包含该 op 的任意 kernel 都能证明”。

## 已支持的逐点子集

| 方言 | operation | ETV 语义 |
| --- | --- | --- |
| `tt` | `get_program_id`、`make_range`、`splat`、`broadcast`、`expand_dims`、`reshape` | lane/program 索引与逐元素形状映射 |
| `tt` | `addptr`、`load`、`store` | block + 元素 offset 的单输出内存模型；当前 value 仅为抽象浮点 |
| `arith` | `constant` | bool/int 与有理浮点 splat constant |
| `arith` | `addi/subi/muli/divsi/remsi` | signed i32 地址/shape 算术及定义性检查 |
| `arith` | `andi/ori/xori` | `i1` 布尔运算；普通整数位运算尚不支持 |
| `arith` | `cmpi` | signed 整数 `eq/ne/slt/sle/sgt/sge` |
| `arith` | `addf/subf/mulf/divf/negf` | `ABSTRACT_FLOAT` 实数表达式 |
| `arith` | `cmpf` | 16 个 MLIR 浮点谓词；按不含 NaN/无穷/signed zero 的实数域解释 |
| `arith` | `select/maxnumf/minnumf` | typed select；极值规范化为比较 + select |
| `arith` | `extsi/extui/trunci/index_cast/index_castui` | 保留 operator 和源/目标类型的整数 cast |
| `arith` | `sitofp/uitofp` | 保留 operator 和 `iN -> fN` 类型，不作为恒等变换 |
| `arith` | `extf/truncf` | 在当前不区分浮点格式的抽象实数模式中同值提升 |
| `math` | `absf/ceil/cos/erf/exp/exp2/floor/fma/log/rsqrt/sin/sqrt` | 有类型纯函数；不自动加入超越函数恒等式 |
| `tt` | `clampf` | 仅 `propagateNan = none`，规范化为 `min(max(x,lo),hi)` |
| `tt` | `extern_elementwise` | 仅已知、`pure = true`、浮点输入和浮点结果的白名单函数 |

纯外部函数白名单当前包括：`acosh/acoshf`、`atan/atanf`、`ceilf`、`coshf`、`erff`、
`expm1f`、`floorf`、`nearbyintf`、`powf`、`rsqrtf`、`sqrtf` 和 `tanhf`。它们映射到与
相应数学操作一致的内部函数名；这是 `ABSTRACT_FLOAT` 的声明语义，不是 CUDA libdevice
位级等价证明。

## 规范化与定义域

TorchInductor 的 `maximum/minimum` helper 通常使用 `cmpf`、自比较 `une`、`ori` 和
`select` 表达 NaN 传播。因为当前语义域明确排除 NaN，`une(x,x)` 可化为 false，helper
与 `arith.maxnumf/minnumf`、`tt.clampf` 会得到同一内部树。该化简不能迁移到未来的
IEEE-754 模式。

`fdiv`、`sqrt`、`rsqrt`、`log`、`acosh` 与一般实指数 `pow` 是部分函数。当前验证器只
接受能由结构分析证明的定义域；不能证明时返回 `FLOAT_DEFINEDNESS_NOT_PROVED`。
`abs/exp/exp2/erf/sin/cos/atan/cosh/tanh/ceil/floor/nearbyint/expm1` 在实数域中按全函数
处理。相同纯函数调用可由 egglog 同余证明，但系统没有添加未经验证的函数恒等式。

masked `tt.load` 缺少 `other` 时不会默认返回零。其默认分支是显式未定义值：固定规模
路径检查该分支在活动 store lane 上是否可达，参数化路径要求 SMT 证明 load mask 恒真。

## 明确保留为 UNKNOWN

- `scf.for`、`tt.reduce` 及其他 region/归约语义；
- `tt.bitcast`，因为它需要位级浮点/整数表示和 target data layout；
- 非 `i1` 的按位整数算术、整数/布尔 load/store、跨元素类型指针观察；
- `isnan/isinf/signbit` 等依赖 IEEE 特殊值或位表示的外部函数；
- atomics、shared memory、barrier、多 store 顺序 effect 和多 kernel orchestration；
- 多维 program grid、动态 rank 和完整 TTGIR layout encoding。

这些 operation 即使能被 libtriton 解析，也会由专门错误码或
`TTIR_OP_UNSUPPORTED` 返回 `UNKNOWN`，不会被忽略。

## Benchmark 重放

2026-09-03 使用 `.venv-ttir` 中的 Triton/libtriton 3.7.1 对仓库现有 1230 个 PairSpec
只读重放。历史 `benchmark/summary.json` 中 236 个 `TTIR_OP_UNSUPPORTED`，本实现降为
67 个；这 67 个的首个未支持 operation 全部是 `tt.bitcast`。新增 op 使大量 kernel
继续进入后续验证阶段，但当前重放仍只有原有 3 个 `PROVED`，主要后续结果为：

| 结果 | 数量 | 首要原因 |
| --- | ---: | --- |
| `UNKNOWN` | 848 | region/归约语义未实现 |
| `DISPROVED` | 113 | 记录的固定 binding 下两侧活动 mask 域不同 |
| `UNKNOWN` | 71 | PairSpec 缺少被访问 block/scalar 的 ABI role |
| `UNKNOWN` | 67 | `tt.bitcast` 未实现 |
| `UNKNOWN` | 44 | 活动 store lane 可达无 `other` 的 masked load 未定义分支 |
| `UNKNOWN` | 36 | 非抽象浮点内存类型 |
| `UNKNOWN` | 32 | 两侧显式 cast 结构不同且无准入 rewrite |
| `UNKNOWN` | 8 | NaN/Inf 常量超出抽象浮点域 |
| `UNKNOWN` | 8 | `isnan` 外部符号超出抽象浮点域 |
| `PROVED` | 3 | 全部义务闭合 |

`DISPROVED(MASK_MISMATCH)` 是在 PairSpec 所记录 launch/binding 前提下的结论；它提示需要
继续审计 benchmark 的 host launch 与 scalar 捕获，而不是用 op 重写掩盖。新增的独立
端到端回归则证明了 `tt.clampf + math.exp + math.erf` 与 TorchInductor 风格
`cmpf + ori + select + extern_elementwise(__nv_erff)` 在声明的抽象语义下等价。
