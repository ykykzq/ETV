# ETV

[English](README.md)

ETV 验证两份 raw TT IR kernel 在给定 PairSpec 前提下是否具有相同的可观察输出内存。
两侧输入统一为 `.ttir`/`.mlir`：

```text
lhs.ttir -> libtriton parse/verify -> TTIR snapshot -> ETV IR --+
                                                               +-> SMT + egglog -> verdict
rhs.ttir -> libtriton parse/verify -> TTIR snapshot -> ETV IR --+
```

ETV IR 是面向验证的共同语义表示，描述 launch、逐 lane 地址、mask、load、算术和
store；它不是 e-graph。证明阶段才把 ETV IR 中的表达式编码进 egglog e-graph，利用
重写与同余闭包判断两侧 `observe_store` 是否进入同一个 e-class。

## 结果

- `PROVED`：在 PairSpec 全部前提下，可观察 Output 内存等价；
- `DISPROVED`：找到可重放的地址、mask 或抽象数值反例；
- `UNKNOWN`：输入超出语义子集，或事实、定义域、规则、资源不足。

当前浮点模式 `ABSTRACT_FLOAT` 使用精确数学值，不代表 IEEE-754、容差或 GPU 位级
等价。当前 MVP 面向固定 rank 的无环逐点程序。固定 specialization 可以在一侧声明
有序多 launch 序列，另一侧保持为单 kernel，并在对应语义边界上进行划分。

整数 cast 在 TT IR 提升后仍是带源/目标类型的显式 IR operator。验证器没有默认的
`cast(x)=x` 规则；两侧 cast 结构不一致时必须由已准入 rewrite 实际消解，否则结果为
`UNKNOWN(CAST_EQUIVALENCE_NOT_REWRITTEN)`。目标相关 `index` 位宽未声明时同样返回
`UNKNOWN`。

逐点前端还支持浮点比较、min/max/clamp、常用 `math` 运算、白名单内的纯 libdevice
`extern_elementwise` 调用，以及保留类型的有符号/无符号整数到浮点 cast。位级操作、
region、归约和非浮点内存语义仍返回 `UNKNOWN`；完整矩阵见
[算子语义支持](docs/operator_support.md)。

## 安装

```bash
uv sync --locked --python 3.12 --extra dev
uv run --locked --extra dev pytest -q
```

该核心环境会跳过 raw TT IR 集成测试。项目要求 uv 0.12.x；Linux 完整环境使用
`uv sync --locked --extra dev --extra ttir`。macOS 使用
`tools/setup_uv_ttir.sh` 建立独立 `.venv-ttir` 环境并源码构建固定
Triton 3.7.1/libtriton。
`extraction` 与 `ttir` 被显式声明为互斥，因为 Linux 上 PyTorch 2.8.0 会选择
Triton 3.4.0。详见[依赖与环境](docs/dependencies.md)。

下文命令使用 `.venv/bin/python`；macOS 源码构建环境应替换为
`.venv-ttir/bin/python`。

## 使用

固定 `[8,16]` 的真实 Add 程序对：

```bash
.venv/bin/python -m etv check examples/add/pair.json --out build/add
```

左侧来自 ntops/ninetoothed，右侧来自 PyTorch/TorchInductor；两侧都是真实生成的
TT IR。预期结果：

```text
PROVED ntops_add_vs_torch_inductor_add: OBSERVABLE_MEMORY_EQUIVALENT
```

参数化的 256-lane 二维 TT IR 对 128-lane 一维 TT IR：

```bash
.venv/bin/python -m etv check examples/add/pair_parametric.json \
  --out build/add_parametric
```

该例在 `a>0, b>0, c>0, a*b=c` 下证明地址、mask、覆盖和计算等价。

解析单个 TT IR 并导出稳定快照：

```bash
.venv/bin/python -m etv parse examples/add/ttir/ntops_add.ttir \
  --out build/ntops_add.snapshot.json --no-assembly
.venv/bin/python -m etv inspect examples/add/ttir/torch_inductor_add.ttir
```

`check` 和 `inspect` 不接受 Semantic JSON 或 Prims。`tests/fixtures/semantic` 中的
JSON 只供证明内核单元测试使用，通过 Python 专用 API `verify_internal_spec` 进入，
不属于用户输入协议；raw TT IR 集成夹具位于 `tests/fixtures/ttir`。

## PairSpec

PairSpec v2 把配置、自然语言上下文、形式谓词、观察目标和重写库严格分开：

```json
{
  "format": "etv-pair-v2",
  "metadata": {
    "pair_id": "lhs_vs_rhs",
    "lhs": "lhs.ttir",
    "rhs": "rhs.ttir",
    "semantic_mode": "abstract_float",
    "frontends": {
      "lhs": {"kind": "ttir", "function": "lhs_kernel", "programs": 1},
      "rhs": {"kind": "ttir", "function": "rhs_kernel", "programs": 1}
    }
  },
  "assumptions": {"for_llm": []},
  "predicates": {
    "abi": {
      "Output": {
        "lhs": {"kind": "block", "name": "arg0"},
        "rhs": {"kind": "block", "name": "arg0"}
      }
    },
    "bindings": {"n": 128},
    "disjoint": [],
    "custom": []
  },
  "observation": {
    "output_role": "Output",
    "output_numel": {"var": "n"},
    "require_full_coverage": true,
    "require_disjoint": []
  },
  "rewrites": []
}
```

TT IR 本身不包含 host launch grid，因此 `programs` 必须显式给出，也可以是
`ceildiv(c, 256)` 之类的符号表达式。`assumptions.for_llm` 不参与形式证明；ABI、shape、
跨程序关系和 no-alias 信息必须写成 `predicates`。完整格式见
[TT IR 输入格式](docs/ttir_input.md)与[完整验证过程](docs/verification_process.md)。

## 等式饱和与 LLM

内建浮点代数规则先由 Z3 Real 证明，显式位宽整数 cast 规则使用 Z3 整数公式验证。
布局、地址、mask、标量 ABI 等关系规则由 ETV 根据 PairSpec 谓词生成，并由参数化 SMT
查询证明适用条件；只有规则在 egglog 中实际匹配并合并根，才会关闭等价目标。用户规则
位于独立的 `etv-rewrite-v1` 文件，加载后带 `user:<path>` 来源与文件 SHA-256。用户或
LLM 新增的非代数规则可按当前策略作为可信规则准入，但若实际使用，报告会标为
`admitted_unverified`，且 `soundness.level` 会降为
`conditional_on_unverified_rewrites`。

可选子图划分让 LLM 扫描完整左右程序并提出对应子图边界。ETV 自己检查根覆盖、路径
唯一性、左右依赖拓扑、无环性和类型，再按依赖顺序分别验证；LLM 不决定等价结论。

多 launch 一侧通过 `metadata.launches.lhs` 或 `.rhs` 声明执行 step、每个 launch 的局部
ABI、binding 和选定 store。ETV 把这些 launch 视为固定的预划分子图，根据逻辑存储角色
与精确 offset 重建中间内存边；LLM 只能选择单 kernel 一侧的对应表达式路径。同一 step
中的 store 并发提交，只有后续 step 可以读取。提案非法时，验证器回退到机器组合后的
整图证明。该路径目前只支持固定规模、仅一侧为 launch 序列，且每个序列项观察一个
store。

## 仓库结构

```text
etv/ir.py                     TT IR 提升后的共同语义 IR
etv/casts.py                  显式 cast 元数据与有限位宽整数语义
etv/ttir/libtriton.py         固定版本 libtriton 解析、验证与快照
etv/ttir/lift.py              TT IR 到 ETV IR 的语义提升
etv/schema.py                 TT IR PairSpec 与内部测试夹具 schema
etv/multilaunch.py            有序 launch 内存组合与固定子图边界
etv/parametric.py             参数化 SMT 义务和谓词派生规则
etv/rules.py                  内建规则库与 predicate requirement
etv/egraph.py                 egglog 编码、饱和与应用日志
etv/partition.py              LLM 成对子图提议与机器检查
etv/observability.py          结构化运行日志与 run 关联
etv/verify.py                 端到端证明编排
tools/extract_add_pair.py     ntops/TorchInductor 双 TT IR 提取
tools/extract_ntops_test_specs.py  可审计的 ntops 测试 Spec 证据抽取
tools/materialize_ntops_operator_specs.py  结构化 YAML Spec 物化
tools/setup_uv_ttir.sh        可重复的 uv/源码构建环境
tools/test_ntops_torch_ttir.py  全算子 TorchInductor TTIR 探针
specs/ntops_test_evidence.json  固定版本测试语法与断言证据
specs/ntops_LICENSE             再分发 ntops 测试片段的许可证
specs/ntops/operators/          75 份独立的仅输入前提算子 YAML
examples/add/                 程序对、PairSpec、源代码与来源哈希
```

进一步阅读：[架构](docs/architecture.md)、[完整验证过程](docs/verification_process.md)、
[实现状态](docs/implementation_status.md)、[uv 与全算子测试](docs/operator_testing.md)、
[算子 Spec 格式](docs/spec_format.md)、[75 个测试派生 ntops Spec](docs/ntops_test_specs.md)、
[运行日志](docs/logging.md)和
[真实 Add 验证](docs/add_validation.md)。
