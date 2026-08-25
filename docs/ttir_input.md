# TT IR 输入格式

ETV 的两个程序输入都必须是 raw TT IR 文本，扩展名为 `.ttir` 或 `.mlir`。PairSpec
使用 `etv-pair-v1`：

```json
{
  "format": "etv-pair-v1",
  "pair_id": "lhs_vs_rhs",
  "lhs": "ttir/lhs.ttir",
  "rhs": "ttir/rhs.ttir",
  "frontends": {
    "lhs": {"kind": "ttir", "function": "lhs_kernel", "programs": 1},
    "rhs": {
      "kind": "ttir",
      "function": "rhs_kernel",
      "programs": {"op": "ceildiv", "args": [{"var": "n"}, 128]}
    }
  },
  "semantic_mode": "abstract_float",
  "roles": {},
  "facts": {},
  "contract": {},
  "limits": {}
}
```

## 必填前端字段

| 字段 | 含义 |
| --- | --- |
| `kind` | 必须为 `ttir` |
| `function` | TT IR module 内的入口 `tt.func` 名称 |
| `programs` | host 启动的一维 program 数，可为整数表达式 |

TT IR 不携带完整 host launch grid，因此 ETV 不允许省略 `programs`。当前内部 Program
只建模一维 `program_id x` 和固定编译期 lane 数；不支持的多维 launch 会返回
`UNKNOWN`。

## libtriton 验证

ETV 固定使用 Triton/libtriton 3.7.1。每份输入依次执行：

1. 加载 Triton 与内置 MLIR 方言；
2. `parse_mlir_module`；
3. MLIR module verifier；
4. 确认入口函数存在；
5. 遍历 operation、operand、result、block、region 和属性；
6. 生成 `etv-ttir-snapshot-v1`；
7. 将支持的语义提升到 `etv.ir.Program`。

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

## 角色映射

两侧 ABI 可不同，PairSpec 用逻辑角色对齐：

```json
"Alpha": {
  "lhs": {"kind": "scalar", "name": "arg2"},
  "rhs": {"kind": "scalar_block", "name": "arg1", "index": 0}
}
```

支持的 endpoint 类型：

- `block`：指针参数；
- `scalar_block`：指针参数指定下标处的标量；
- `scalar`：按值传入的标量。

角色映射本身是可信 PairSpec 前提。提升器不会因参数名称相同而自动对齐。

## 内部夹具不是输入格式

仓库中 `tests/fixtures/semantic` 下的 JSON 用于隔离测试 SMT、egglog、规则和反例逻辑。
CLI 明确拒绝这些文件；`tests/fixtures/ttir` 则继续走正式 libtriton 前端。生产系统
不存在 `auto`、`prims` 或 `semantic_json` 用户前端。
