# 运行日志

ETV 将证明报告和运行日志分开：

- `report.json`/`report.md` 是可保存、可审阅的证明和审计结果；
- 运行日志记录 schema、TTIR 前端、规则准入、Z3、egglog、LLM、子图划分和最终状态等执行事件。

日志使用 Python 标准库 `logging`，不增加运行时依赖。每次 CLI 运行都有一个 `run_id`，
验证 PairSpec 后还会记录 `pair_id` 和阶段 `phase`，便于在并行或多次验证时关联事件。

## CLI

```bash
etv check examples/add/pair.json \
  --out build/add-report \
  --log-level INFO \
  --log-format json
```

`check --out DIR` 未显式指定 `--log-file` 时，会自动写入 `DIR/etv.log`。其他命令需要
通过 `--log-file PATH` 显式保存日志：

```bash
etv parse kernel.ttir --log-level DEBUG --log-file build/parse.log
```

日志默认写到 stderr；`check --out` 自动创建的日志文件额外保存 `INFO` 级别事件。显式
指定 `--log-file` 时，文件与 stderr 使用同一日志级别。stdout 保留给 CLI 的机器输出和最终摘要，因此
`etv check --json` 的 stdout 仍然是合法 JSON。`--log-format text` 适合交互查看，
`--log-format json` 输出 JSON Lines，适合采集系统。

支持的级别为 `DEBUG`、`INFO`、`WARNING`、`ERROR` 和 `CRITICAL`。默认级别是 `WARNING`，
所以普通库调用不会产生过程噪声；设置 `INFO` 可以看到阶段事件，设置 `DEBUG` 可以看到
规则列表和更细的后端信息。

## 事件和边界

常见事件包括：

- `verification_started`、`spec_loaded`、`verification_result`；
- `ttir_parse_started`、`ttir_module_verified`、`ttir_parse_finished`；
- `rewrite_admission_started`、`rewrite_validation_finished`；
- `egraph_saturation_started`、`egraph_saturation_finished`；
- `llm_request_started`、`llm_request_finished`、`llm_rule_proposal_finished`；
- `partition_request_finished`、`partition_fallback`。
- `launch_sequence_started`、`launch_sequence_built`；
- `launch_partition_started`、`launch_partition_proved`、`launch_partition_fallback`。

日志不是证明证据，也不会改变验证结论。完整规则、谓词、SMT 查询摘要、输入哈希和
soundness 仍以正式报告为准。日志不会记录 API key 或完整 LLM payload，只记录用途、模型、
prompt/response 哈希和统计字段。

库调用者可以通过 `etv.observability.configure_logging` 配置 handler，也可以直接配置
`etv` logger。未配置时，ETV 不主动创建 stdout handler。
