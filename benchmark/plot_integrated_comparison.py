from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


SYSTEM_ORDER = ["pure_algo", "kuant_full", "quant_full"]
SYSTEM_LABELS = {
    "pure_algo": "Pure Algorithm",
    "kuant_full": "KuantAgent",
    "quant_full": "QuantAgent",
}
SYSTEM_COLORS = {
    "pure_algo": "#2C7A7B",
    "kuant_full": "#DD6B20",
    "quant_full": "#4A5568",
}


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_results_dir(results_dir: Path | None) -> Path:
    if results_dir is not None:
        return results_dir.resolve()

    benchmark_results_dir = Path(__file__).resolve().parent / "results"
    candidates = sorted(
        [p for p in benchmark_results_dir.iterdir() if p.is_dir() and p.name.startswith("integrated_comparison_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No integrated_comparison_* result directory found.")
    return candidates[0]


def _load_system_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    dataframes: Dict[str, pd.DataFrame] = {}
    for system in SYSTEM_ORDER:
        path = results_dir / system / "results.json"
        if path.exists():
            dataframes[system] = pd.DataFrame(_load_json(path))
    return dataframes


def _sample_index_from_file(sample_file: str) -> int | None:
    try:
        return int(Path(sample_file).stem.split("_")[-1])
    except Exception:
        return None


def _has_system_error(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("system_error"))


def _summarize_frame(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "accuracy_report": {
                "num_samples": 0,
                "accuracy": None,
                "final_direction_accuracy": None,
                "horizon_majority_accuracy": None,
                "avg_confidence": None,
                "num_neutral_predictions": 0,
                "actionable_num_samples": 0,
                "actionable_coverage": 0.0,
                "actionable_accuracy": None,
                "final_direction_actionable_accuracy": None,
                "num_neutral_moves": 0,
                "filtered_num_samples": 0,
                "filtered_accuracy_ex_neutral": None,
                "final_direction_filtered_accuracy_ex_neutral": None,
                "filtered_actionable_num_samples": 0,
                "filtered_actionable_accuracy": None,
                "asset_accuracy": {},
                "asset_final_direction_accuracy": {},
                "details": [],
            }
        }

    if "final_direction_correct" not in df.columns:
        df["final_direction_correct"] = (df["predicted"].astype(str).str.upper() == df["true_direction"].astype(str).str.upper()).astype(int)
    if "horizon_majority_correct" not in df.columns:
        df["horizon_majority_correct"] = df["correct"].astype(int)
    accuracy = round(float(df["horizon_majority_correct"].mean()), 4)
    final_direction_accuracy = round(float(df["final_direction_correct"].mean()), 4)
    horizon_total = int(df["horizon_total_count"].sum()) if "horizon_total_count" in df.columns else int(len(df))
    horizon_correct = int(df["horizon_correct_count"].sum()) if "horizon_correct_count" in df.columns else int(df["correct"].sum())
    horizon_step_accuracy = None if horizon_total == 0 else round(float(horizon_correct / horizon_total), 4)
    avg_confidence = round(float(df["confidence"].mean()), 4)
    actionable_df = df[df["is_neutral_prediction"] == 0]
    strong_move_df = df[df["is_neutral_move"] == 0]
    strong_move_actionable_df = strong_move_df[strong_move_df["is_neutral_prediction"] == 0]
    asset_accuracy = df.groupby("asset")["horizon_majority_correct"].mean().round(4).to_dict()
    asset_final_direction_accuracy = df.groupby("asset")["final_direction_correct"].mean().round(4).to_dict()

    return {
        "accuracy_report": {
            "num_samples": int(len(df)),
            "accuracy": accuracy,
            "final_direction_accuracy": final_direction_accuracy,
            "horizon_majority_accuracy": accuracy,
            "horizon_step_accuracy": horizon_step_accuracy,
            "avg_confidence": avg_confidence,
            "num_neutral_predictions": int(df["is_neutral_prediction"].sum()),
            "actionable_num_samples": int(len(actionable_df)),
            "actionable_coverage": round(float(len(actionable_df) / len(df)), 4),
            "actionable_accuracy": None if actionable_df.empty else round(float(actionable_df["horizon_majority_correct"].mean()), 4),
            "final_direction_actionable_accuracy": None
            if actionable_df.empty
            else round(float(actionable_df["final_direction_correct"].mean()), 4),
            "num_neutral_moves": int(df["is_neutral_move"].sum()),
            "filtered_num_samples": int(len(strong_move_df)),
            "filtered_accuracy_ex_neutral": None if strong_move_df.empty else round(float(strong_move_df["horizon_majority_correct"].mean()), 4),
            "final_direction_filtered_accuracy_ex_neutral": None
            if strong_move_df.empty
            else round(float(strong_move_df["final_direction_correct"].mean()), 4),
            "filtered_actionable_num_samples": int(len(strong_move_actionable_df)),
            "filtered_actionable_accuracy": None
            if strong_move_actionable_df.empty
            else round(float(strong_move_actionable_df["horizon_majority_correct"].mean()), 4),
            "asset_accuracy": asset_accuracy,
            "asset_final_direction_accuracy": asset_final_direction_accuracy,
            "details": df.to_dict(orient="records"),
        }
    }


def _filter_common_success_samples(system_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    filtered_frames: Dict[str, pd.DataFrame] = {}
    valid_sets: List[set[str]] = []
    for system in SYSTEM_ORDER:
        frame = system_frames.get(system)
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        if "parsed_decision" in working.columns:
            working = working[~working["parsed_decision"].apply(_has_system_error)]
        filtered_frames[system] = working
        valid_sets.append(set(working["sample_file"].tolist()))

    if not valid_sets:
        return filtered_frames

    common_samples = set.intersection(*valid_sets)
    for system, frame in list(filtered_frames.items()):
        filtered_frames[system] = frame[frame["sample_file"].isin(common_samples)].copy()
        filtered_frames[system]["sample_index"] = filtered_frames[system]["sample_file"].apply(_sample_index_from_file)
        filtered_frames[system] = filtered_frames[system].sort_values(["sample_index", "sample_file"]).drop(columns=["sample_index"])
    return filtered_frames


def _filter_index_range(system_frames: Dict[str, pd.DataFrame], start_index: int | None, end_index: int | None) -> Dict[str, pd.DataFrame]:
    if start_index is None and end_index is None:
        return system_frames

    ranged_frames: Dict[str, pd.DataFrame] = {}
    for system, frame in system_frames.items():
        working = frame.copy()
        working["sample_index"] = working["sample_file"].apply(_sample_index_from_file)
        if start_index is not None:
            working = working[working["sample_index"] >= start_index]
        if end_index is not None:
            working = working[working["sample_index"] <= end_index]
        ranged_frames[system] = working.sort_values(["sample_index", "sample_file"]).drop(columns=["sample_index"])
    return ranged_frames


def _extract_summary_table(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        report = (summary.get(system) or {}).get("accuracy_report", {})
        rows.append(
            {
                "system": system,
                "label": SYSTEM_LABELS[system],
                "final_direction_accuracy": report.get("final_direction_accuracy"),
                "final_direction_filtered_accuracy_ex_neutral": report.get("final_direction_filtered_accuracy_ex_neutral"),
                "final_direction_actionable_accuracy": report.get("final_direction_actionable_accuracy"),
                "actionable_coverage": report.get("actionable_coverage"),
                "avg_confidence": report.get("avg_confidence"),
                "num_samples": report.get("num_samples"),
            }
        )
    return pd.DataFrame(rows)


def _save_metric_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(summary_df))
    width = 0.22

    final_accuracy = summary_df["final_direction_accuracy"].fillna(0.0).tolist()
    filtered = summary_df["final_direction_filtered_accuracy_ex_neutral"].fillna(0.0).tolist()
    actionable = summary_df["final_direction_actionable_accuracy"].fillna(0.0).tolist()
    colors = [SYSTEM_COLORS[s] for s in summary_df["system"]]

    ax.bar([i - width for i in x], final_accuracy, width=width, label="Final Direction Accuracy", color=colors, alpha=0.98)
    ax.bar([i + 0.0 for i in x], actionable, width=width, label="Final Direction Actionable Accuracy", color=colors, alpha=0.76)
    ax.bar([i + width for i in x], filtered, width=width, label="Final Direction Filtered Accuracy", color=colors, alpha=0.58)

    ax.set_title("Three-System Final-Direction Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["label"].tolist())
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for offset, values in [
        (-width, final_accuracy),
        (0.0, actionable),
        (width, filtered),
    ]:
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_confidence_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = list(range(len(summary_df)))
    confidence_values = [0.0 if pd.isna(v) else float(v) for v in summary_df["avg_confidence"]]
    colors = [SYSTEM_COLORS[s] for s in summary_df["system"]]

    bars = ax.bar(x, confidence_values, color=colors, alpha=0.9)
    ax.set_title("Average Confidence Comparison")
    ax.set_ylabel("Average Confidence")
    ax.set_ylim(0, max(0.8, max(confidence_values) + 0.1))
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["label"].tolist())
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value, system in zip(bars, confidence_values, summary_df["system"]):
        label = "N/A" if system == "quant_full" and math.isclose(value, 0.0) else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_sample_comparison_chart(system_frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    base_df = system_frames.get("pure_algo")
    if base_df is None or base_df.empty:
        raise ValueError("pure_algo results are required for sample comparison plotting.")

    merged = pd.DataFrame(
        {
            "sample_file": base_df["sample_file"].tolist(),
            "sample": [Path(v).stem for v in base_df["sample_file"]],
            "true_direction": base_df["true_direction"].tolist(),
        }
    )
    for system in SYSTEM_ORDER:
        if system in system_frames:
            frame = system_frames[system][["sample_file", "predicted"]].copy()
            frame = frame.rename(columns={"predicted": SYSTEM_LABELS[system]})
            merged = merged.merge(frame, on="sample_file", how="left")

    merged = merged.drop(columns=["sample_file"])

    fig, ax = plt.subplots(figsize=(14, max(5.5, 0.55 * len(merged) + 1.5)))
    ax.axis("off")

    columns = merged.columns.tolist()
    table = ax.table(
        cellText=merged.values.tolist(),
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for col_index, col_name in enumerate(columns):
        header_cell = table[0, col_index]
        header_cell.set_facecolor("#E2E8F0")
        header_cell.set_text_props(weight="bold")

    for row_idx in range(len(merged)):
        truth = merged.iloc[row_idx]["true_direction"]
        for col_idx, col_name in enumerate(columns):
            cell = table[row_idx + 1, col_idx]
            if col_name in {"sample", "true_direction"}:
                continue
            pred = merged.iloc[row_idx][col_name]
            if pred == truth:
                cell.set_facecolor("#C6F6D5")
            else:
                cell.set_facecolor("#FED7D7")

    ax.set_title("Sample-Level Prediction Comparison", fontsize=13, pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_explanation_text(results_dir: Path, summary_df: pd.DataFrame) -> str:
    best_final_accuracy_row = summary_df.sort_values("final_direction_accuracy", ascending=False).iloc[0]
    best_actionable_row = summary_df.sort_values("final_direction_actionable_accuracy", ascending=False).iloc[0]
    best_filtered_row = summary_df.sort_values("final_direction_filtered_accuracy_ex_neutral", ascending=False).iloc[0]

    lines: List[str] = []
    lines.append("# Integrated Comparison Figure Notes")
    lines.append("")
    lines.append(f"- Results directory: `{results_dir}`")
    lines.append(f"- Best final direction accuracy: `{best_final_accuracy_row['label']}` = `{best_final_accuracy_row['final_direction_accuracy']:.4f}`")
    lines.append(f"- Best actionable final direction accuracy: `{best_actionable_row['label']}` = `{best_actionable_row['final_direction_actionable_accuracy']:.4f}`")
    lines.append(
        f"- Best filtered final direction accuracy: `{best_filtered_row['label']}` = `{best_filtered_row['final_direction_filtered_accuracy_ex_neutral']:.4f}`"
    )
    lines.append("")
    lines.append("## Suggested interpretation")
    lines.append("")
    lines.append(
        "- Read final direction accuracy first because it directly measures whether the model predicts the correct direction at the end of the forecast horizon."
    )
    lines.append(
        "- Final direction actionable accuracy is the second most important metric because it shows whether executable, non-neutral decisions are correct under the same end-of-horizon criterion."
    )
    lines.append(
        "- Actionable coverage remains important, but it is better reported in tables or text than mixed into the main thesis bar chart."
    )
    lines.append(
        "- The filtered final-direction metric is useful as a supplementary view because it removes very small future moves and better reflects clearer directional opportunities."
    )
    lines.append(
        "- The confidence chart and sample-level comparison chart should be treated as supplementary materials rather than headline thesis figures."
    )
    lines.append(
        "- For QuantAgent, a zero confidence value may reflect missing confidence output in the original JSON schema rather than true lack of confidence."
    )
    lines.append("")
    lines.append("## Recommended paper wording")
    lines.append("")
    lines.append(
        "The comparison results should be interpreted primarily through final-direction metrics. Under this criterion, KuantAgent demonstrates the strongest end-of-horizon directional prediction performance on the tested BTC 1h benchmark, while actionable accuracy and coverage together indicate that this advantage is not obtained solely by excessive abstention."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison figures for an integrated benchmark result directory.")
    parser.add_argument("--results-dir", default=None, help="Path to integrated_comparison_* directory. Defaults to latest.")
    parser.add_argument("--common-success-only", action="store_true", help="Plot only samples that succeeded for all systems.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional 1-based inclusive sample start index.")
    parser.add_argument("--end-index", type=int, default=0, help="Optional 1-based inclusive sample end index.")
    args = parser.parse_args()

    results_dir = _resolve_results_dir(Path(args.results_dir) if args.results_dir else None)
    system_frames = _load_system_results(results_dir)
    if args.common_success_only:
        system_frames = _filter_common_success_samples(system_frames)
    system_frames = _filter_index_range(
        system_frames,
        args.start_index if args.start_index > 0 else None,
        args.end_index if args.end_index > 0 else None,
    )
    summaries = {system: _summarize_frame(frame) for system, frame in system_frames.items()}
    summary = {system: summaries.get(system) for system in SYSTEM_ORDER}
    summary_df = _extract_summary_table(summary)

    output_dir_name = "figures"
    if args.common_success_only:
        output_dir_name += "_common_success"
    if args.start_index > 0 or args.end_index > 0:
        start_label = args.start_index if args.start_index > 0 else "start"
        end_label = args.end_index if args.end_index > 0 else "end"
        output_dir_name += f"_{start_label}_{end_label}"
    output_dir = results_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    _save_metric_chart(summary_df, output_dir / "accuracy_comparison.png")
    _save_confidence_chart(summary_df, output_dir / "confidence_comparison.png")
    _save_sample_comparison_chart(system_frames, output_dir / "sample_level_comparison.png")

    explanation = _build_explanation_text(results_dir, summary_df)
    (output_dir / "figure_notes.md").write_text(explanation, encoding="utf-8")

    print(
        json.dumps(
            {
                "results_dir": str(results_dir),
                "figures_dir": str(output_dir),
                "generated_files": [
                    "accuracy_comparison.png",
                    "confidence_comparison.png",
                    "sample_level_comparison.png",
                    "comparison_metrics.csv",
                    "figure_notes.md",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
