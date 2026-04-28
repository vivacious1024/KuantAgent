from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_summary(results_dir: Path) -> Dict[str, Any]:
    return json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))


def _load_compact(results_dir: Path) -> pd.DataFrame:
    return pd.read_csv(results_dir / "compact_results.csv")


def _top_reason_tokens(values: pd.Series, top_k: int = 12) -> List[str]:
    counter: Counter[str] = Counter()
    for value in values.dropna():
        for reason in str(value).split(" | "):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return [f"{reason}: {count}" for reason, count in counter.most_common(top_k)]


def _summarize_prediction_mix(df: pd.DataFrame) -> Dict[str, int]:
    return {
        key: int(value)
        for key, value in df["predicted"].value_counts(dropna=False).to_dict().items()
    }


def _compare_runs(previous_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    previous_map = {row["sample_file"]: row for _, row in previous_df.iterrows()}
    current_map = {row["sample_file"]: row for _, row in current_df.iterrows()}

    improved: List[Dict[str, Any]] = []
    worsened: List[Dict[str, Any]] = []

    for sample_file, current_row in current_map.items():
        previous_row = previous_map.get(sample_file)
        if previous_row is None:
            continue
        old_correct = int(previous_row["correct"])
        new_correct = int(current_row["correct"])
        if old_correct == 0 and new_correct == 1:
            improved.append(
                {
                    "sample_file": sample_file,
                    "old_predicted": previous_row["predicted"],
                    "new_predicted": current_row["predicted"],
                }
            )
        elif old_correct == 1 and new_correct == 0:
            worsened.append(
                {
                    "sample_file": sample_file,
                    "old_predicted": previous_row["predicted"],
                    "new_predicted": current_row["predicted"],
                }
            )

    return {
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "improved_samples": improved[:10],
        "worsened_samples": worsened[:10],
    }


def build_diagnostic_report(current_dir: Path, previous_dir: Path | None = None) -> str:
    current_summary = _load_summary(current_dir)
    current_df = _load_compact(current_dir)

    accuracy_report = current_summary["accuracy_report"]
    wrong_df = current_df[current_df["correct"] == 0]
    neutral_df = current_df[current_df["predicted"] == "NEUTRAL"] if "predicted" in current_df.columns else current_df.iloc[0:0]
    actionable_df = current_df[current_df["predicted"] != "NEUTRAL"] if "predicted" in current_df.columns else current_df

    lines: List[str] = []
    lines.append("# 纯算法层诊断报告")
    lines.append("")
    lines.append(f"- 结果目录：`{current_dir}`")
    lines.append(f"- 总体准确率：`{accuracy_report.get('accuracy')}`")
    lines.append(f"- 可执行样本准确率：`{accuracy_report.get('actionable_accuracy')}`")
    lines.append(f"- 可执行样本覆盖率：`{accuracy_report.get('actionable_coverage')}`")
    lines.append(f"- 中性预测数量：`{accuracy_report.get('num_neutral_predictions')}`")
    lines.append("")
    lines.append("## 1. 预测分布")
    lines.append("")
    for key, value in _summarize_prediction_mix(current_df).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## 2. 错误样本最常见理由")
    lines.append("")
    for item in _top_reason_tokens(wrong_df.get("supporting_reasons", pd.Series(dtype=str))):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## 3. 中性预测最常见门控理由")
    lines.append("")
    if "gate_reasons" in neutral_df.columns and not neutral_df.empty:
        for item in _top_reason_tokens(neutral_df["gate_reasons"]):
            lines.append(f"- {item}")
    else:
        lines.append("- 当前没有中性预测，或未记录门控理由。")

    lines.append("")
    lines.append("## 4. 高置信度错误样本")
    lines.append("")
    high_conf_wrong = wrong_df.sort_values("confidence", ascending=False).head(10)
    if high_conf_wrong.empty:
        lines.append("- 当前没有错误样本。")
    else:
        lines.append("| sample | predicted | raw_predicted | true | confidence | gate |")
        lines.append("|---|---|---|---|---:|---|")
        for _, row in high_conf_wrong.iterrows():
            lines.append(
                f"| {Path(str(row['sample_file'])).name} | {row.get('predicted')} | {row.get('raw_predicted')} | "
                f"{row.get('true_direction')} | {row.get('confidence')} | {row.get('signal_gate')} |"
            )

    lines.append("")
    lines.append("## 5. 组件分数失衡观察")
    lines.append("")
    if not actionable_df.empty:
        component_lines = [
            f"- 平均 `indicator_long_score`: `{round(float(actionable_df['indicator_long_score'].mean()), 4) if 'indicator_long_score' in actionable_df.columns else 'N/A'}`",
            f"- 平均 `indicator_short_score`: `{round(float(actionable_df['indicator_short_score'].mean()), 4) if 'indicator_short_score' in actionable_df.columns else 'N/A'}`",
            f"- 平均 `pattern_long_score`: `{round(float(actionable_df['pattern_long_score'].mean()), 4) if 'pattern_long_score' in actionable_df.columns else 'N/A'}`",
            f"- 平均 `pattern_short_score`: `{round(float(actionable_df['pattern_short_score'].mean()), 4) if 'pattern_short_score' in actionable_df.columns else 'N/A'}`",
            f"- 平均 `trend_long_score`: `{round(float(actionable_df['trend_long_score'].mean()), 4) if 'trend_long_score' in actionable_df.columns else 'N/A'}`",
            f"- 平均 `trend_short_score`: `{round(float(actionable_df['trend_short_score'].mean()), 4) if 'trend_short_score' in actionable_df.columns else 'N/A'}`",
        ]
        lines.extend(component_lines)
    else:
        lines.append("- 当前没有可执行方向样本。")

    if previous_dir is not None and previous_dir.exists():
        previous_df = _load_compact(previous_dir)
        comparison = _compare_runs(previous_df, current_df)
        lines.append("")
        lines.append("## 6. 与上一轮对比")
        lines.append("")
        lines.append(f"- 改好样本数：`{comparison['improved_count']}`")
        lines.append(f"- 改坏样本数：`{comparison['worsened_count']}`")
        if comparison["worsened_samples"]:
            lines.append("- 代表性改坏样本：")
            for item in comparison["worsened_samples"]:
                lines.append(
                    f"- `{Path(item['sample_file']).name}`: `{item['old_predicted']} -> {item['new_predicted']}`"
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断纯算法层实验结果，输出可读的故障分析报告。")
    parser.add_argument("--results-dir", required=True, help="当前实验结果目录")
    parser.add_argument("--previous-results-dir", default=None, help="上一轮结果目录，可选")
    parser.add_argument("--output", default=None, help="诊断报告输出路径，可选")
    args = parser.parse_args()

    current_dir = Path(args.results_dir).resolve()
    previous_dir = Path(args.previous_results_dir).resolve() if args.previous_results_dir else None
    report = build_diagnostic_report(current_dir=current_dir, previous_dir=previous_dir)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = current_dir / "diagnostic_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"诊断报告已生成: {output_path}")


if __name__ == "__main__":
    main()
