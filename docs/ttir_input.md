# TT IR 与 PairSpec v2 输入格式

ETV 的两个程序输入必须是 raw TT IR 文本，扩展名为 `.ttir` 或 `.mlir`。正式 PairSpec
使用 `etv-pair-v2`，顶层由五个互不替代的部分组成：

| 部分 | 作用 | 是否参与证明 |
| --- | --- | --- |
| `metadata` | 文件、入口、launch、资源限制、LLM 和划分配置 | 配置会决定证明范围 |
| `assumptions` | 给 LLM 阅读的自然语言背景 | 否，不能作为规则 gate |
| `predicates` | ABI、shape、参数关系、no-alias 和自定义谓词 | 是，默认作为前提成立 |
| `observation` | 要证明相等的输出范围和内存条件 | 是，这是证明目标 |
| `rewrites` | 本次运行加载的外部用户规则文件 | 规则通过准入后参与 |

完整骨架如下：

```json
{
  "format": "etv-pair-v2",
  "metadata": {
    "pair_id": "lhs_vs_rhs",
    "lhs": "ttir/lhs.ttir",
    "rhs": "ttir/rhs.ttir",
    "semantic_mode": "abstract_float",
    "frontends": {
      "lhs": {"kind": "ttir", "function": "lhs_kernel", "programs": 1},
      "rhs": {
        "kind": "ttir",
        "function": "rhs_kernel",
        "programs": {"op": "ceildiv", "args": [{"var": "n"}, 128]}
      }
    },
    "limits": {"max_iterations": 8, "max_enodes": 20000, "timeout_ms": 5000},
    "rule_policy": {
      "algebraic_validation": "required",
      "non_algebraic_validation": "trusted"
    },
    "llm": {"enabled": false},
    "partition": {"enabled": false}
  },
  "assumptions": {
    "for_llm": ["两侧程序来自同一个逐元素算子，但使用不同布局。"]
  },
  "predicates": {
    "abi": {
      "Output": {
        "lhs": {"kind": "block", "name": "arg0"},
        "rhs": {"kind": "block", "name": "arg1"}
      }
    },
    "bindings": {"n": 128},
    "side_bindings": {"lhs": {}, "rhs": {}},
    "parameters": {},
    "constraints": [],
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

## metadata

`metadata.frontends.lhs` 和 `rhs` 都必须包含：

| 字段 | 含义 |
| --- | --- |
| `kind` | 必须为 `ttir` |
| `function` | TT IR module 内的入口 `tt.func` 名称 |
| `programs` | host 启动的一维 program 数，可为整数表达式 |

TT IR 不携带完整 host launch grid，所以 ETV 不允许省略 `programs`。`limits` 控制
egglog 迭代、e-node 数和超时。`llm` 与 `partition` 只控制辅助流程；API 输出本身不构成
证明。启用 `partition.enabled` 必须同时显式启用 `llm.enabled`。

## assumptions

`assumptions.for_llm` 是自然语言字符串列表，只会进入 LLM 的规则生成和成对子图划分
提示。验证器不会把这些文本翻译成 Z3 公式，也不会允许 v2 规则使用文本
`assumption` gate。因此，文本写着“两个布局等价”不会使程序获得等价关系。

旧 `etv-pair-v1` 仍可读，用于兼容已有夹具；其中 `facts.assumptions` 保留旧的可信
fact-gate 行为。新输入应使用 v2，避免混淆自然语言和形式前提。

## predicates

`predicates` 是默认正确的前提集合。验证结论形式为：

```text
all declared predicates => observation(lhs) = observation(rhs)
```

它包括：

- `abi`：两个程序物理参数到同一逻辑角色的对应；
- `bindings`：共享整数常量或整数表达式；
- `side_bindings`：仅在某一侧成立的 shape、stride、numel 等绑定；
- `parameters`：全称量化 signed-i32 参数及闭区间；
- `constraints`：使用 ETV 表达式语言描述的布尔关系；
- `disjoint`：逻辑 block 的 no-alias 组；
- `custom`：带稳定 ID 的扩展谓词。

ABI endpoint 支持 `block`、`scalar` 和 `scalar_block`。`scalar_block` 还必须给出读取
`index`。同名物理参数不会自动对应，只能通过 `abi` 建立关系。

验证器为结构谓词生成稳定 ID，例如 `abi.Output`、`binding.n`、
`binding.lhs.arg4`、`parameter.n`、`constraint.0` 和 `disjoint.0`。外部规则可以用
`{"kind":"predicate","id":"abi.Output"}` 精确引用。

### 自定义谓词

自定义谓词有两种编码：

```json
{"id": "shape.same_numel", "kind": "shape", "encoder": "z3_expr",
 "formula": {"op": "eq", "args": [{"var": "a"}, {"var": "b"}]}}
```

`z3_expr` 公式会作为 SMT 前提进入参数化义务。当前 schema 接受由 signed-i32 参数、
整数算术/比较、布尔连接和 `select` 组成的表达式；浮点、内存读取或未实现理论会在
加载时拒绝。公式本身仍由用户声明为真，不会因为可编码就自动变成已证明结论。

```json
{"id": "layout.vendor_equivalent", "kind": "relation", "encoder": "trusted"}
```

`trusted` 用于当前还没有编码器的领域谓词。它只能作为规则 gate；其正确性完全属于
用户信任边界。PairSpec 不允许声明 `smt_proved` 来自行提升证据等级。

## observation

`observation` 不是前提，而是证明目标：`output_role` 指定可观察逻辑 block，
`output_numel` 指定逻辑元素数，`require_full_coverage` 要求完整写覆盖，
`require_disjoint` 要求对应角色的两两 no-alias 谓词已经声明。

## 外部 rewrite 文件

用户规则不再内嵌于 v2 PairSpec。PairSpec 仅引用相对路径：

```json
"rewrites": [{"file": "rules/layout.json"}]
```

规则文件格式为：

```json
{
  "format": "etv-rewrite-v1",
  "rules": [
    {
      "id": "vendor_layout_identity",
      "kind": "trusted_predicate",
      "lhs": {"op": "fadd", "args": [{"match": "a"}, {"float": "0"}]},
      "rhs": {"match": "a"},
      "requires": [{"kind": "predicate", "id": "layout.vendor_equivalent"}]
    }
  ]
}
```

加载后，规则来源被强制标记为 `user:<绝对路径>`，文件 SHA-256 进入报告。规则不会写回
全局库，也不会污染后续验证。代数规则默认必须经 Z3 Real 证明；`trusted_predicate` 必须有
满足的形式 gate，并以 `admitted_unverified` 记录。自然语言 assumptions 不是形式 gate。

## libtriton 输入检查

ETV 固定使用 Triton/libtriton 3.7.1。每份输入依次注册方言、解析 module、执行 MLIR
verifier、确认入口、遍历完整 IR、生成快照，再把已支持语义提升到 `etv.ir.Program`。

常见错误码：

| 错误码 | 含义 |
| --- | --- |
| `TTIR_PAIR_REQUIRED` | 任一侧不是 TT IR 或前端声明不是 `ttir` |
| `TTIR_FRONTEND_UNAVAILABLE` | 无法导入固定 libtriton |
| `TTIR_VERSION_MISMATCH` | Triton 版本不是 3.7.1 |
| `TTIR_PARSE_ERROR` | MLIR/TT IR 语法解析失败 |
| `TTIR_VERIFY_ERROR` | module verifier 失败 |
| `TTIR_FUNCTION_REQUIRED` | module 无唯一入口且 PairSpec 未指定 |
| `TTIR_FUNCTION_NOT_FOUND` | 指定入口不存在 |
| `TTIR_LAUNCH_REQUIRED` | 缺少 `programs` |

`tests/fixtures/semantic` 下的 Semantic JSON 仅供证明核心测试；CLI 不接受该格式。
