from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from plot_integrated_comparison import (
    _extract_summary_table,
    _save_confidence_chart,
    _save_metric_chart,
    _save_sample_comparison_chart,
    _build_explanation_text,
)


SYSTEM_ORDER = ["pure_algo", "kuant_full", "quant_full"]
SYSTEM_LABELS = {
    "pure_algo": "Pure Algorithm Baseline",
    "kuant_full": "KuantAgent Full System",
    "quant_full": "QuantAgent Original System",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_index_from_file(sample_file: str) -> int | None:
    try:
        return int(Path(sample_file).stem.split("_")[-1])
    except Exception:
        return None


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    sample_file = str(normalized.get("sample_file", ""))
    normalized["sample_file"] = str(Path(sample_file).resolve()) if sample_file else sample_file
    return normalized


def _sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            _sample_index_from_file(str(item.get("sample_file", ""))) is None,
            _sample_index_from_file(str(item.get("sample_file", ""))) or 0,
            str(item.get("sample_file", "")),
        ),
    )


def _summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
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
                "final_direction_filtered_actionable_accuracy": None,
                "hard_case_num_samples": 0,
                "hard_case_accuracy": None,
                "final_direction_hard_case_accuracy": None,
                "hard_case_actionable_num_samples": 0,
                "hard_case_actionable_accuracy": None,
                "final_direction_hard_case_actionable_accuracy": None,
                "algorithm_only_num_samples": 0,
                "algorithm_only_coverage": 0.0,
                "system_failure_count": 0,
                "successful_num_samples": 0,
                "successful_coverage": 0.0,
                "asset_accuracy": {},
                "asset_final_direction_accuracy": {},
                "details": [],
            }
        }

    df = pd.DataFrame(records)
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
    hard_case_df = df[df["is_hard_case"] == 1]
    hard_case_actionable_df = hard_case_df[hard_case_df["is_neutral_prediction"] == 0]
    successful_df = df[df["is_system_failure"] == 0]
    asset_accuracy = df.groupby("asset")["horizon_majority_correct"].mean().round(4).to_dict()
    asset_final_direction_accuracy = df.groupby("asset")["final_direction_correct"].mean().round(4).to_dict()

    return {
        "accuracy_report": {
            "num_samples": int(len(df)),
            "accuracy": accuracy,
            "final_direction_accuracy": final_direction_accuracy,
            "horizon_majority_accuracy": accuracy,
            "horizon_step_accuracy": horizon_step_accuracy,
            "horizon_correct_count": horizon_correct,
            "horizon_total_count": horizon_total,
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
            "final_direction_filtered_actionable_accuracy": None
            if strong_move_actionable_df.empty
            else round(float(strong_move_actionable_df["final_direction_correct"].mean()), 4),
            "hard_case_num_samples": int(len(hard_case_df)),
            "hard_case_accuracy": None if hard_case_df.empty else round(float(hard_case_df["horizon_majority_correct"].mean()), 4),
            "final_direction_hard_case_accuracy": None
            if hard_case_df.empty
            else round(float(hard_case_df["final_direction_correct"].mean()), 4),
            "hard_case_actionable_num_samples": int(len(hard_case_actionable_df)),
            "hard_case_actionable_accuracy": None
            if hard_case_actionable_df.empty
            else round(float(hard_case_actionable_df["horizon_majority_correct"].mean()), 4),
            "final_direction_hard_case_actionable_accuracy": None
            if hard_case_actionable_df.empty
            else round(float(hard_case_actionable_df["final_direction_correct"].mean()), 4),
            "algorithm_only_num_samples": int((df["decision_route"] == "algorithm_only").sum()),
            "algorithm_only_coverage": round(float((df["decision_route"] == "algorithm_only").mean()), 4),
            "system_failure_count": int(df["is_system_failure"].sum()),
            "successful_num_samples": int(len(successful_df)),
            "successful_coverage": round(float(len(successful_df) / len(df)), 4),
            "asset_accuracy": asset_accuracy,
            "asset_final_direction_accuracy": asset_final_direction_accuracy,
            "details": df.to_dict(orient="records"),
        }
    }


def _write_system_outputs(system_dir: Path, raw_results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "results.json").write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(raw_results).to_csv(system_dir / "results.csv", index=False, encoding="utf-8-sig")
    (system_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_report(
    output_dir: Path,
    source_dirs: List[Path],
    summaries: Dict[str, Dict[str, Any] | None],
) -> str:
    lines: List[str] = []
    lines.append("# Merged Integrated Comparison Report")
    lines.append("")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("- Source result directories:")
    for source_dir in source_dirs:
        lines.append(f"  - `{source_dir}`")
    lines.append("")

    for system in SYSTEM_ORDER:
        lines.append(f"## {SYSTEM_LABELS[system]}")
        summary = summaries.get(system)
        if summary is None:
            lines.append("- Not available")
            lines.append("")
            continue
        report = summary["accuracy_report"]
        lines.append(f"- Num samples: `{report['num_samples']}`")
        lines.append(f"- Final direction accuracy: `{report.get('final_direction_accuracy')}`")
        lines.append(f"- Horizon majority accuracy: `{report.get('horizon_majority_accuracy', report['accuracy'])}`")
        lines.append(f"- Horizon step accuracy: `{report.get('horizon_step_accuracy')}`")
        lines.append(f"- Avg confidence: `{report['avg_confidence']}`")
        lines.append(f"- Final direction actionable accuracy: `{report.get('final_direction_actionable_accuracy')}`")
        lines.append(f"- Horizon majority actionable accuracy: `{report['actionable_accuracy']}`")
        lines.append(f"- Actionable coverage: `{report['actionable_coverage']}`")
        lines.append(f"- Final direction filtered accuracy ex neutral: `{report.get('final_direction_filtered_accuracy_ex_neutral')}`")
        lines.append(f"- Horizon majority filtered accuracy ex neutral: `{report['filtered_accuracy_ex_neutral']}`")
        lines.append(f"- Final direction hard-case accuracy: `{report.get('final_direction_hard_case_accuracy')}`")
        lines.append(f"- Horizon majority hard-case accuracy: `{report['hard_case_accuracy']}`")
        lines.append(f"- Final direction hard-case actionable accuracy: `{report.get('final_direction_hard_case_actionable_accuracy')}`")
        lines.append(f"- Horizon majority hard-case actionable accuracy: `{report['hard_case_actionable_accuracy']}`")
        lines.append(f"- Algorithm-only coverage: `{report['algorithm_only_coverage']}`")
        lines.append(f"- System failure count: `{report['system_failure_count']}`")
        lines.append(f"- Successful coverage: `{report['successful_coverage']}`")
        lines.append("")

    pure_summary = summaries.get("pure_algo")
    kuant_summary = summaries.get("kuant_full")
    quant_summary = summaries.get("quant_full")

    if pure_summary and kuant_summary:
        pure_acc = pure_summary["accuracy_report"].get("final_direction_accuracy")
        kuant_acc = kuant_summary["accuracy_report"].get("final_direction_accuracy")
        lines.append("## KuantAgent vs Pure Algorithm")
        if pure_acc is not None and kuant_acc is not None:
            lines.append(f"- Final direction accuracy delta: `{round(kuant_acc - pure_acc, 4)}`")
        if kuant_summary["accuracy_report"]["system_failure_count"] > 0:
            lines.append("- Fairness note: merged KuantAgent results still contain system failures.")
        lines.append("")

    if kuant_summary and quant_summary:
        kuant_acc = kuant_summary["accuracy_report"].get("final_direction_accuracy")
        quant_acc = quant_summary["accuracy_report"].get("final_direction_accuracy")
        lines.append("## KuantAgent vs QuantAgent")
        if kuant_acc is not None and quant_acc is not None:
            lines.append(f"- Final direction accuracy delta: `{round(kuant_acc - quant_acc, 4)}`")
        if (
            kuant_summary["accuracy_report"]["system_failure_count"] > 0
            or quant_summary["accuracy_report"]["system_failure_count"] > 0
        ):
            lines.append("- Fairness note: merged full-system results still contain execution failures.")
        lines.append("")

    return "\n".join(lines)


def _merge_system_records(source_dirs: List[Path], system: str) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source_dir in source_dirs:
        results_path = source_dir / system / "results.json"
        if not results_path.exists():
            continue
        records = _load_json(results_path)
        if not isinstance(records, list):
            continue
        for raw_record in records:
            record = _normalize_record(raw_record)
            sample_file = str(record.get("sample_file", ""))
            if not sample_file:
                continue
            merged[sample_file] = record
    return _sort_records(list(merged.values()))


def _collect_system_failures(source_dirs: List[Path]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for source_dir in source_dirs:
        failure_path = source_dir / "system_failures.json"
        if not failure_path.exists():
            continue
        try:
            payload = _load_json(failure_path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict):
                failures.append(
                    {
                        "source_results_dir": str(source_dir),
                        **item,
                    }
                )
    return failures


def _generate_figures(output_dir: Path, summaries: Dict[str, Dict[str, Any] | None]) -> None:
    summary_df = _extract_summary_table(summaries)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(figures_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    _save_metric_chart(summary_df, figures_dir / "accuracy_comparison.png")
    _save_confidence_chart(summary_df, figures_dir / "confidence_comparison.png")

    system_frames: Dict[str, pd.DataFrame] = {}
    for system in SYSTEM_ORDER:
        results_path = output_dir / system / "results.json"
        if results_path.exists():
            system_frames[system] = pd.DataFrame(_load_json(results_path))
    if "pure_algo" in system_frames and not system_frames["pure_algo"].empty:
        _save_sample_comparison_chart(system_frames, figures_dir / "sample_level_comparison.png")

    notes = _build_explanation_text(output_dir, summary_df)
    (figures_dir / "figure_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple integrated benchmark result directories into one consolidated result package."
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        required=True,
        help="One integrated_comparison_* result directory. Repeat this argument for multiple directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to benchmark/results/integrated_comparison_merged_<timestamp>.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label inserted into the auto-generated merged directory name.",
    )
    args = parser.parse_args()

    source_dirs = [Path(path).resolve() for path in args.results_dir]
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {source_dir}")

    benchmark_results_dir = Path(__file__).resolve().parent / "results"
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"_{args.label}" if args.label else ""
        output_dir = benchmark_results_dir / f"integrated_comparison_merged{label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_records: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any] | None] = {}
    for system in SYSTEM_ORDER:
        records = _merge_system_records(source_dirs, system)
        merged_records[system] = records
        summary = _summarize_records(records) if records else None
        summaries[system] = summary
        if summary is not None:
            _write_system_outputs(output_dir / system, records, summary)

    combined_summary = {system: summaries.get(system) for system in SYSTEM_ORDER}
    (output_dir / "summary.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_text = _build_report(output_dir, source_dirs, summaries)
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")

    merge_manifest = {
        "source_results_dirs": [str(path) for path in source_dirs],
        "merged_at": datetime.now().isoformat(),
        "systems": {
            system: {
                "num_records": len(merged_records.get(system, [])),
            }
            for system in SYSTEM_ORDER
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "mode": "merge_existing_results",
                "results_dirs": [str(path) for path in source_dirs],
                "label": args.label,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "merge_manifest.json").write_text(json.dumps(merge_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    system_failures = _collect_system_failures(source_dirs)
    if system_failures:
        (output_dir / "system_failures.json").write_text(
            json.dumps(system_failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _generate_figures(output_dir, combined_summary)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_results_dirs": [str(path) for path in source_dirs],
                "systems": {
                    system: {
                        "num_records": len(merged_records.get(system, [])),
                        "accuracy": (summaries.get(system) or {}).get("accuracy_report", {}).get("accuracy"),
                    }
                    for system in SYSTEM_ORDER
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
