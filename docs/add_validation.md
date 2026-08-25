# Add 验证

## 程序对

Add 语义为：

```text
output = input + alpha * other
```

左侧沿用原有九齿链路：

```text
ntops Add
  -> ninetoothed 0.26.0 SSA lowering
  -> Triton source
  -> Triton 3.7.1 optimized TTIR
```

右侧改为：

```text
PyTorch expression
  -> TorchRefsMode + make_fx
  -> prims.broadcast_in_dim
  -> prims.mul
  -> prims.add
  -> etv-prims-program-v1
```

右侧不调用 TorchInductor，也不生成 Triton/TTIR。

## 资产

| 文件 | 作用 |
| --- | --- |
| `examples/add/ttir/ntops_add.ttir` | 真实九齿侧固定 specialization |
| `examples/add/ttir/parametric_2d_add.ttir` | 固定 rank 2 的符号 shape TTIR |
| `examples/add/prims/torch_add.prims.json` | 固定 rank 2 的符号 Prims 图 |
| `examples/add/pair.json` | `[8,16]` singleton 参数域 |
| `examples/add/pair_parametric.json` | 任意正 i32 `a,b,c` 且 `a*b=c` |
| `examples/add/provenance.json` | 上游提交、工具版本和输入 SHA-256 |

`pair.json` 虽对应固定 `[8,16]`，仍将 shape 声明为：

```text
a in [8,8]
b in [16,16]
c in [128,128]
a*b=c
```

这样真实 TTIR/Prims 程序对始终走符号规则路径，不依靠固定 lane 的直接值比较。

## Prims 图

实际 Torch 图包含：

```text
alpha: []
  -> prims.broadcast_in_dim(shape=[torch_dim0,torch_dim1], dims=[])
  -> prims.mul(other)
  -> prims.add(input)
  -> output
```

PairSpec 将：

```text
torch_dim0 = a
torch_dim1 = b
```

rank 2 固定；`a,b` 的值被符号化。Prims 前端把输出域表示为 `a*b` 个逻辑元素。

## ABI 对应

| 逻辑角色 | 九齿 TTIR | Torch Prims |
| --- | --- | --- |
| `Input` | block `arg0` | tensor `input` |
| `Other` | block `arg1` | tensor `other` |
| `Alpha` | scalar `arg2` | rank-0 tensor `alpha[0]` |
| `Output` | block `arg3` | tensor `output` |

`Input/Other/Output` 的 no-alias 关系由 PairSpec 声明。角色对应是验证前提，不从指针
名称猜测。

## 参数化输入

九齿侧 launch：

```text
programs = ceildiv(c, 256)
lanes    = 256
logical  = pid*256 + lane
mask     = logical < a*b
offset   = (logical div b)*b + logical rem b
```

Torch Prims 逻辑域：

```text
programs = a*b
lanes    = 1
logical  = pid
mask     = true
offset   = row_major(index(pid,[a,b]))
```

这些表达式保持符号化；验证器不选择一个示例 shape 替代整个域。

## 验证方式

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

预期结果：

```text
PROVED parametric_ninetoothed_add_vs_torch_prims_add: OBSERVABLE_MEMORY_EQUIVALENT
```

证明分为两部分：

1. Z3 在 `a*b=c` 和正 i32 域中证明 launch、coverage、mask、地址、无竞争以及每条
   load/store 重写的条件；
2. egglog 实际应用物理端点到逻辑端点的规则，使两侧完整 `observe_store` 根合并。

初始根不是同一个 e-class。报告必须显示 load/store 关系规则有非零 match，且
`after_fact_rewrites.unmatched_root_pairs=0`。这防止重新引入“先由 SMT 归一化 load，
再发现两侧根天然相同”的旧路径。

详细执行轨迹见[参数化 Add 验证全过程](add_parametric_verification_details.md)。

## 重新提取

```bash
python tools/extract_add_pair.py \
  --ntops-repo /path/to/ntops \
  --ninetoothed-repo /path/to/ninetoothed \
  --output examples/add
```

工具检查固定 Git HEAD、ntops 关键源码哈希、ninetoothed/PyTorch/Triton 版本；只对
左侧调用 Triton 编译。右侧检查实际 Prims 调用序列后写出严格 JSON。所有输出哈希
写入 `provenance.json`。

## 证明边界

本例证明声明参数域上、`ABSTRACT_FLOAT` 语义下的最终 Output 内存等价。它不证明
IEEE-754 位级行为、运行时事实真实性、buffer 边界、动态 rank、循环/归约或前端
提升器自身的形式正确性。
