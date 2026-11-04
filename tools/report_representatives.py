"""Generate an auditable report without dropping failed representatives."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--corpus", default="frozen-ttir-v2")
    parser.add_argument("--analysis", default="analysis-v4")
    args = parser.parse_args()
    root = args.root
    corpus = root / args.corpus
    summary = read(root / args.analysis / "summary.json")
    lock = read(corpus / "corpus.lock.json")
    suite = read(corpus / "suite.lock.json")
    rows = summary["cases"]
    assert len(rows) == len(suite["selected"]) == lock["selected"]
    assert {r["nodeid"] for r in rows} == set(suite["selected"])
    for file, digest in lock["files"].items():
        assert hashlib.sha256((corpus / file).read_bytes()).hexdigest() == digest, file
    capture_counts = Counter((r.get("capture") or {}).get("status", "no_hook") for r in rows)
    process_counts = Counter((r.get("process") or {}).get("returncode") for r in rows)
    reasons = Counter(r.get("reason") for r in rows if r["status"] == "UNKNOWN")
    ready = sum(r["status"] in ("PROVED", "DISPROVED", "UNKNOWN") for r in rows)
    proved = [r for r in rows if r["status"] == "PROVED"]
    dtype = []
    for row in rows:
        if "storage role has inconsistent element types" not in row.get("diagnostics", []):
            continue
        c = read(corpus / row["case_id"] / "capture.json")
        types = {}
        for side in ("lhs", "rhs"):
            ids = c["outputs"][side]
            types[side] = sorted(
                {
                    slot["runtime"]["dtype"]
                    for launch in c["launches"]
                    if launch["side"] == side
                    for slot in launch["slots"]
                    if slot["runtime"].get("storage_id") in ids
                }
            )
        dtype.append({"nodeid": row["nodeid"], **types})
    controls = []
    for path in sorted((root / "controls").glob("*/report.json")):
        d = read(path)
        controls.append(
            {
                "case": path.parent.name,
                "status": d["status"],
                "reason": d["reason"],
                "counterexample": d["counterexample"],
            }
        )
    mutant = root / "controls/frozen_add_sub_mutant/validation/report.json"
    if mutant.exists():
        d = read(mutant)
        controls.append(
            {
                "case": "frozen_add_sub_mutant",
                "status": d["status"],
                "reason": d["reason"],
                "counterexample": d["counterexample"],
            }
        )
    metrics = {
        "test_inventory": len(suite["all_cases"]),
        "representatives": len(rows),
        "test_modules": len({r["nodeid"].split("::")[0] for r in rows}),
        "ttir_files": sum(r["ttir_files"] for r in lock["cases"]),
        "capture_counts": dict(capture_counts),
        "pytest_exit_codes": dict(process_counts),
        "verdicts": summary["verdicts"],
        "unknown_reasons": dict(reasons),
        "dtype_contract_differences": dtype,
        "controls": controls,
        "analysis_wall_seconds": summary["wall_seconds"],
        "corpus_sha256": hashlib.sha256((corpus / "corpus.lock.json").read_bytes()).hexdigest(),
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# ETV 固定代表集实验（2026-11-04）",
        "",
        "本实验在用户指定远程 L20C 机器上执行。结果只覆盖下列固定参数、ABI 与数学实数语义，不能外推为全参数或 IEEE-754 正确性。",
        "",
        "## 版本与范围",
        "",
        f"- ntops 提交：`{suite['source_commit']}`。未修改上游测试源文件。",
        f"- 完整 pytest 清单：{len(suite['all_cases'])} 个参数化用例，锁定 {len(suite['files'])} 个测试源文件的哈希。",
        f"- 执行代表集：{len(rows)} 个测试函数，来自 {metrics['test_modules']} 个测试模块。每函数优先 float32、非原地、最小 shape，最后按 nodeid 排序。",
        "- 每个隔离进程按完整测试目录收集，恢复上游模块导入顺序；执行前逐项校验锁定参数。原测试自带模块/用例 SHA-256 随机种子。",
        f"- 固定 TTIR：{metrics['ttir_files']} 份；集合哈希 `{metrics['corpus_sha256']}`。采集失败和部分采集仍保留在分母中。",
        "- 采集环境：Python 3.12.13、PyTorch 2.8.0、Triton 3.4.0、ninetoothed 0.26.0；验证环境：Triton 3.7.1、egglog 13.2.0、Z3 4.16.0.0。依赖受 uv.lock 和下载哈希固定。",
        "- 采集关闭 Inductor 静态 C++ launcher，以保留实际选中内核的 TTIR；未改动原数值测试。CUDA graph、外部库内核和 eager 后处理不能被当成已采集完整。",
        "",
        "## 量化结果",
        "",
        "| 指标 | 数量 |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in summary["verdicts"].items()],
        f"| 进入有效形式验证 | {ready} |",
        f"| 全代表集证明率 | {len(proved)}/{len(rows)} = {len(proved) / len(rows):.2%} |",
        f"| 有效契约内证明率 | {len(proved)}/{ready} = {len(proved) / ready:.2%} |",
        f"| 验证批次墙钟时间（8 个 worker） | {summary['wall_seconds']:.3f} 秒 |",
        "",
        f"采集状态：`{dict(capture_counts)}`。pytest 退出码分布：`{dict(process_counts)}`。退出码 0 只表示原数值测试通过，不代表形式证明通过。",
        "",
        "### UNKNOWN 的原因",
        "",
        "| 原因 | 数量 |",
        "|---|---:|",
        *[f"| `{k}` | {v} |" for k, v in sorted(reasons.items())],
        "",
        "`NOT_VERIFIED` 表示采集、ABI 配对或验证进程未能形成有效结果。它们没有被当成反例，也没有被移出 109 个代表用例的分母。完整细节在 summary.json 和原始 pytest 日志中。",
        "",
        "## 验证通过的代表用例",
        "",
        "| 测试函数 | Case ID |",
        "|---|---|",
        *[f"| `{r['nodeid'].split('[')[0]}` | `{r['case_id']}` |" for r in proved],
        "",
        "具体 shape、stride、dtype、标量参数、launch grid 均在对应 case 的 case_status.json/capture.json 中；同一算子的其他参数未据此宣称通过。",
        "",
        "## 不等价与反例",
        "",
        f"自然测试集合中，已重放的形式数值反例为 {summary['verdicts'].get('DISPROVED', 0)} 个；发现 {len(dtype)} 个输出 dtype 契约差异（独立于数值等价判定）。",
        "",
        "| 测试函数 | ntops 输出 | PyTorch 输出 |",
        "|---|---|---|",
        *[
            f"| `{r['nodeid'].split('[')[0]}` | {', '.join(r['lhs'])} | {', '.join(r['rhs'])} |"
            for r in dtype
        ],
        "",
        "下面是明确标注的人工缺陷对照，不能计作自然发现的 ntops 错误。frozen_add_sub_mutant 从固定 add 内核复制，将右侧唯一 arith.addf 改为 arith.subf；mutation.json 记录原始/变异哈希。",
        "",
        "| 对照 | 结论 | 原因 |",
        "|---|---|---|",
        *[f"| `{r['case']}` | {r['status']} | `{r['reason']}` |" for r in controls],
        "",
    ]
    for control in controls:
        if control["counterexample"]:
            lines += [
                f"**{control['case']} 的可重放证据**",
                "",
                "```json",
                json.dumps(control["counterexample"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
    lines += [
        "## 系统验证流程",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[固定版本、参数与运行环境] --> B[采集实际选中的双方 TTIR]",
        "  B --> C[官方 MLIR 解析验证与 ABI 配对]",
        "  C --> D[带类型标量 IR 和多 launch 内存组合]",
        "  D --> E[Z3 检查地址、mask、覆盖、别名和定义域]",
        "  E --> F[Z3 审核重写规则，再由 egglog 合并观察根]",
        "  F --> G[PROVED / UNKNOWN]",
        "  F --> H[精确 SMT 反例 + 独立求值器重放]",
        "  H --> I[DISPROVED / UNKNOWN]",
        "```",
        "",
        "1. 保存完整 pytest 清单、代表选择、源码与依赖哈希。输入复制保留 storage、stride、offset 和别名关系。",
        "2. 记录实际 selected launch，排除 autotuning 候选。保存 grid、函数、signature、constexpr、标量 ABI 和物理 buffer。",
        "3. 用 Triton 3.7.1 官方 MLIR parser/verifier 构造结构化快照，再生成严格 etv-pair-v3。角色类型冲突、缺失输出写入、eager 临时输入等一律阻止配对。",
        "4. 提升到不可变、带类型的标量表达式。按 launch step 组合内存；固定规模枚举全部 program/lane，检查覆盖、写冲突、地址、未定义读和分离前提。",
        "5. 代数重写先经 Z3 证明准入，再在资源上限内运行 egglog。只有必要义务关闭且全部观察根合并才输出 PROVED；本批次未启用 LLM 或未验证规则。",
        "6. 根未合并时尝试精确 SMT 反例；再以独立求值器重放有理数/整数模型。无法精确求值的超越函数不会用于宣称数值反例。",
        "",
        "## 本轮实现与边界",
        "",
        "新增有限整数/布尔内存及位运算、标量浮点 ABI 的精确有理数绑定、受限 bool 指针 byte-store、静态单输入结合律 tt.reduce、有限实数域 isnan，以及经 Z3 审核的 min/max/select 重写。修正了 cmpf 无序比较枚举映射，并以 16 种谓词 × 3 组输入回归。早期枚举错误产生的伪反例全部废弃，不纳入本报告。",
        "",
        "分支/循环、非有限常量、浮点位级重解释、非规范布尔 byte-load、多输出 API、部分 host/eager 后处理及外部库内核仍存在支持缺口。静态归约不等同于任意 region 支持。即使不含循环，当前也不是全部代表用例均可证明的系统。",
        "",
        "证明的浮点语义是数学实数；不包含 IEEE 舍入、NaN/Inf、signed zero 或误差容限。形状、ABI、host 调度和 storage 分离是声明前提；外部 buffer 的实际分配上界不是本系统推导的结论。",
        "",
        "## 逐用例结果",
        "",
        "| 测试 ID | 结论 | 原因/阻塞 |",
        "|---|---|---|",
    ]
    for row in rows:
        reason = row.get("reason") or "; ".join(row.get("diagnostics", []))
        if not reason:
            reason = (
                (row.get("capture") or {}).get("error")
                or row.get("analysis_error")
                or "参见进程日志/采集记录"
            )
        reason = reason.splitlines()[0].replace("|", "\\|")[:220]
        lines.append(f"| `{row['nodeid']}` | {row['status']} | {reason} |")
    lines += [
        "",
        "## 复核与重跑",
        "",
        "```bash",
        "# 使用已构建官方适配器的独立验证环境；不需要 GPU 重新验证 TTIR",
        "PYTHONPATH=src .venv-verify/bin/python tools/analyze_representatives.py \\",
        f"  experiments/20261104/{args.corpus}/suite.lock.json \\",
        f"  --captures experiments/20261104/{args.corpus} \\",
        "  --python .venv-verify/bin/python --out build/recheck --workers 8",
        "```",
        "",
        "分析脚本在运行前后复核 corpus.lock.json 中全部冻结文件的哈希。重新采集会生成新的版本目录，不能覆盖已冻结集合。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in ("controls", "dtype_contract_differences")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
