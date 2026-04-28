from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
KUANT_DIR = PROJECT_ROOT / "KuantAgent"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(KUANT_DIR) not in sys.path:
    sys.path.insert(0, str(KUANT_DIR))

from algorithm_evaluator import summarize_batch_results
from batch_baseline_runner import run_directory_baseline, save_batch_results


def _build_compact_rows(batch_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in batch_results:
        baseline = result["baseline_result"]
        decision = result["analysis_output"]["decision_features"]
        rows.append(
            {
                "asset": baseline["asset"],
                "timeframe": baseline["timeframe"],
                "sample_file": baseline["sample_file"],
                "predicted": baseline["dominant_side"],
                "raw_predicted": decision.get("raw_dominant_side"),
                "true_direction": baseline["true_direction"],
                "correct": baseline["correct"],
                "horizon_correct_count": baseline.get("horizon_correct_count"),
                "horizon_total_count": baseline.get("horizon_total_count"),
                "horizon_step_accuracy": baseline.get("horizon_step_accuracy"),
                "future_step_directions": baseline.get("future_step_directions"),
                "confidence": baseline["confidence"],
                "is_neutral_prediction": int(baseline["dominant_side"] == "NEUTRAL"),
                "consensus_level": baseline["consensus_level"],
                "risk_level": baseline["risk_level"],
                "future_return_pct": baseline["future_return_pct"],
                "indicator_long_score": decision.get("indicator_long_score"),
                "indicator_short_score": decision.get("indicator_short_score"),
                "pattern_long_score": decision.get("pattern_long_score"),
                "pattern_short_score": decision.get("pattern_short_score"),
                "trend_long_score": decision.get("trend_long_score"),
                "trend_short_score": decision.get("trend_short_score"),
                "long_score": decision.get("long_score"),
                "short_score": decision.get("short_score"),
                "score_gap": decision.get("score_gap"),
                "signal_gate": decision.get("signal_gate"),
                "gate_reasons": " | ".join(decision.get("gate_reasons", [])),
                "supporting_reasons": " | ".join(decision.get("supporting_reasons", [])),
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _generate_markdown_report(
    benchmark_dir: Path,
    timeframe: str,
    window_size: int,
    future_horizon: int,
    summary: Dict[str, Any],
    compact_df: pd.DataFrame,
) -> str:
    accuracy_report = summary.get("accuracy_report", {})
    confidence_buckets = summary.get("confidence_buckets", {})
    bucket_accuracy = confidence_buckets.get("accuracy", {})
    bucket_counts = confidence_buckets.get("counts", {})
    asset_accuracy = accuracy_report.get("asset_accuracy", {})

    lines: List[str] = []
    lines.append("# 纯算法层实验报告")
    lines.append("")
    lines.append("## 1. 实验配置")
    lines.append("")
    lines.append(f"- Benchmark 目录：`{benchmark_dir}`")
    lines.append(f"- 时间粒度：`{timeframe}`")
    lines.append(f"- 输入窗口长度：`{window_size}` 根 K 线")
    lines.append(f"- 未来标签长度：`{future_horizon}` 根 K 线")
    lines.append("")
    lines.append("## 2. 总体结果")
    lines.append("")
    lines.append(f"- 样本数量：`{accuracy_report.get('num_samples', 0)}`")
    lines.append(f"- 总体方向准确率：`{_format_number(accuracy_report.get('accuracy'))}`")
    lines.append(f"- 剔除中性小波动后的准确率：`{_format_number(accuracy_report.get('filtered_accuracy_ex_neutral'))}`")
    lines.append(f"- 平均置信度：`{_format_number(accuracy_report.get('avg_confidence'))}`")
    lines.append(f"- 中性小波动样本数：`{accuracy_report.get('num_neutral_moves', 0)}`")
    lines.append("")
    lines.append("## 3. 如何理解这些指标")
    lines.append("")
    lines.append("- `accuracy`：纯算法层直接判断 `LONG/SHORT` 的总体正确率。")
    lines.append("- `avg_confidence`：算法层对自身结论的平均把握程度。")
    lines.append("- `filtered_accuracy_ex_neutral`：剔除未来波动极小的边缘样本后，再看方向正确率。")
    lines.append("- `confidence_buckets`：检查高置信度信号是否真的更可靠。")
    lines.append("- `future_return_pct`：未来标签窗口末尾收盘价，相对输入窗口最后收盘价的涨跌幅。")
    lines.append("")
    lines.append("## 4. 分资产准确率")
    lines.append("")
    if asset_accuracy:
        for asset, accuracy in asset_accuracy.items():
            lines.append(f"- `{asset}`：`{_format_number(accuracy)}`")
    else:
        lines.append("- 当前批次只有单一资产，或暂时没有足够样本形成分资产统计。")
    lines.append("")
    lines.append("## 5. 置信度分桶")
    lines.append("")
    lines.append("如果 `high` 桶显著好于 `medium/low`，说明算法层已经具备一定的信号质量排序能力。")
    lines.append("")
    for bucket in ["high", "medium", "low"]:
        lines.append(
            f"- `{bucket}`：准确率 `{bucket_accuracy.get(bucket)}`，样本数 `{bucket_counts.get(bucket, 0)}`"
        )
    lines.append("")
    lines.append("## 6. 建议优先观察什么")
    lines.append("")
    lines.append("- 如果总体准确率一般，但高置信度桶明显更强，优先保留高质量信号。")
    lines.append("- 如果高低置信度差异不明显，说明置信度校准还需要继续优化。")
    lines.append("- 如果某一资产明显更差，优先检查该资产上的趋势识别、位置判断和假突破过滤。")
    lines.append("")
    lines.append("## 7. 样本预览")
    lines.append("")

    preview_df = compact_df.head(12)
    if preview_df.empty:
        lines.append("当前没有可展示的样本。")
    else:
        lines.append("| asset | predicted | true | correct | confidence | future_return_pct |")
        lines.append("|---|---|---|---:|---:|---:|")
        for _, row in preview_df.iterrows():
            lines.append(
                "| "
                f"{row['asset']} | "
                f"{row['predicted']} | "
                f"{row['true_direction']} | "
                f"{row['correct']} | "
                f"{_format_number(row['confidence'])} | "
                f"{_format_number(row['future_return_pct'])} |"
            )

    lines.append("")
    lines.append("## 8. 输出文件说明")
    lines.append("")
    lines.append("- `batch_results.json`：完整逐样本结果，适合深度排错和后续分析。")
    lines.append("- `compact_results.csv`：压缩结果表，适合直接查看、排序和做论文图表。")
    lines.append("- `summary.json`：汇总统计结果，适合程序继续读取。")
    lines.append("- `report.md`：当前这份说明报告，适合你快速理解实验表现。")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "一键运行 KuantAgent 纯算法层实验，并导出 JSON、CSV 与 Markdown 结果。"
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Benchmark sample directory, for example benchmark/1h/btc",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="时间粒度标签，例如 1h、4h、1d",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=45,
        help="输入窗口长度，表示每个样本允许算法层看到的 K 线根数",
    )
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=3,
        help="未来标签长度，用于生成真实方向标签",
    )
    parser.add_argument(
        "--neutral-threshold-pct",
        type=float,
        default=0.15,
        help="若未来涨跌幅绝对值低于该阈值，则视为中性小波动样本，用于补充评估统计",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="结果输出目录；若不提供，将自动创建时间戳目录",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = CURRENT_DIR / "results" / f"pure_algo_{args.timeframe}_{timestamp}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_results = run_directory_baseline(
        benchmark_dir=benchmark_dir,
        timeframe=args.timeframe,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        neutral_threshold_pct=args.neutral_threshold_pct,
    )
    summary = summarize_batch_results(batch_results)
    compact_df = _build_compact_rows(batch_results)

    save_batch_results(batch_results, output_dir / "batch_results.json")
    compact_df.to_csv(output_dir / "compact_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md = _generate_markdown_report(
        benchmark_dir=benchmark_dir,
        timeframe=args.timeframe,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        summary=summary,
        compact_df=compact_df,
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    accuracy_report = summary.get("accuracy_report", {})

    print("纯算法层实验已完成。")
    print(f"样本数量: {accuracy_report.get('num_samples', 0)}")
    print(f"总体方向准确率: {_format_number(accuracy_report.get('accuracy'))}")
    print(f"剔除中性小波动后的准确率: {_format_number(accuracy_report.get('filtered_accuracy_ex_neutral'))}")
    print(f"平均置信度: {_format_number(accuracy_report.get('avg_confidence'))}")
    print(f"结果目录: {output_dir}")
    print("已导出: batch_results.json / compact_results.csv / summary.json / report.md")


if __name__ == "__main__":
    main()
