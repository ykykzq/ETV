# 真实 Add 验证

## 程序来源

固定样例验证：

```text
output = input + alpha * other
shape = [8,16], dtype = float32, numel = 128
```

左侧来源链：

```text
InfiniTensor/ntops add.py
-> ninetoothed 0.26.0 lowering
-> Triton 3.7.1 AST frontend / TTIR passes
-> examples/add/ttir/ntops_add.ttir
```

右侧来源链：

```text
PyTorch input + alpha * other
-> FakeTensor make_fx
-> TorchInductor GraphLowering/codegen
-> generated @triton.jit pointwise kernel
-> Triton 3.7.1 AST frontend / TTIR passes
-> examples/add/ttir/torch_inductor_add.ttir
```

右侧不是手写 TT IR，也不是 Prims 替代物。`sources/torch_inductor_add.generated.py`
保留 TorchInductor 原始 kernel，`torch_inductor_add.triton.py` 仅移除 host launch decorator，
保留原 `@triton.jit` 函数用于离线 AST 编译。提取时使用固定 A100/sm80 属性，不执行
CUDA kernel。

## 固定版本

`provenance.json` 固定并哈希：

- ntops commit `9ae4166ad342e4745f0eed13a5a20d069e994fc0`；
- ninetoothed commit `efe519d1b12a820e7aa605d775af3d52c8b0d605`；
- ninetoothed 0.26.0；
- PyTorch 2.8.0；
- Triton/libtriton 3.7.1；
- 两侧源文件、TT IR 和 PairSpec 的 SHA-256。

本提交先完成双 TT IR 前端重构，仍保留上述已验证的固定上游版本。最新版 ntops 和
全算子 TorchInductor TT IR 可提取性矩阵属于下一阶段，不能把一次网络 checkout 当作
已经完成的来源升级。

## ABI 对应

| 逻辑角色 | ntops TT IR | TorchInductor TT IR |
| --- | --- | --- |
| Input | `arg0` block | `arg0` block |
| Alpha | `arg2` value scalar | `arg1[0]` scalar block |
| Other | `arg1` block | `arg2` block |
| Output | `arg3` block | `arg3` block |

虽然参数编号部分相同，ETV 仍保留 side 标签，并只依据 PairSpec 映射建立逻辑对应。

左侧还接收 shape/stride ABI 参数 `arg4..arg15`；右侧 `arg4` 是 `xnumel=128`。这些值
位于 `facts.side_bindings`。两侧 launch 均为一个 program，但左侧编译为 256 lane，
右侧为 128 lane。

## 实际验证轨迹

1. schema 确认两侧均为 TT IR、入口和 launch 完整；
2. libtriton 分别 parse/verify 两个 module；
3. 提升器构建两个内部 `Program`；
4. 枚举左侧 256 lane 和右侧 128 lane；
5. 左侧后 128 lane 被 mask，活动逻辑域均为 `[0,128)`；
6. PairSpec 角色把两侧 input/other/alpha/output 对齐；
7. 地址和 mask 义务通过；
8. egglog 使用角色关系与代数规则合并：

```text
lhs: input[k] + alpha * other[k]
rhs: input[k] + load(alpha_block,0) * other[k]
```

9. `scalar_block(arg1,0)` 依据 Alpha 角色映射为共同逻辑标量；
10. 完整 `observe_store` 根进入同一 e-class，结果为
    `PROVED(OBSERVABLE_MEMORY_EQUIVALENT)`。

## 运行与重现

验证已提交输入：

```bash
python -m etv check examples/add/pair.json --out build/add
```

重新提取需要固定 checkout 和额外依赖：

```bash
python tools/extract_add_pair.py \
  --ntops-repo /path/to/ntops \
  --ninetoothed-repo /path/to/ninetoothed \
  --output examples/add
```

工具检查 Git HEAD、tracked worktree、ntops 关键源文件哈希和 Python 包版本。生成后
更新 `provenance.json`；任何文件变化都会使 provenance 测试失败。

## 结论边界

该固定例只证明 `[8,16]`、float32 ABI、声明 launch 和 `ABSTRACT_FLOAT` 下的输出
内存等价，不证明任意 shape 或 IEEE-754 位级一致。任意正 shape 的前端/证明回归由
参数化双 TT IR 样例单独承担。
