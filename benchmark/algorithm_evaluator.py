from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_label_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Load a label manifest for offline evaluation.

    Expected columns:
    - sample_file
    - true_direction
    Optional columns:
    - future_return
    - asset
    - timeframe
    """

    return pd.read_csv(manifest_path)


def evaluate_directional_accuracy(
    batch_results: List[Dict[str, Any]],
    label_manifest: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    Evaluate pure algorithmic predictions against labeled directions.

    Why this matters:
    If the deterministic layer cannot produce stable directional quality on its
    own, then later AI improvements may only be masking weak foundations.
    """

    rows: List[Dict[str, Any]] = []
    manifest_map = (
        {str(row["sample_file"]): row for _, row in label_manifest.iterrows()}
        if label_manifest is not None
        else {}
    )

    for result in batch_results:
        baseline_result = result["baseline_result"]
        sample_file = str(baseline_result["sample_file"])
        predicted = str(baseline_result["dominant_side"])
        if label_manifest is not None:
            if sample_file not in manifest_map:
                continue
            target = manifest_map[sample_file]
            true_direction = str(target["true_direction"])
            asset = target.get("asset", baseline_result.get("asset"))
            timeframe = target.get("timeframe", baseline_result.get("timeframe"))
        else:
            true_direction = str(baseline_result["true_direction"])
            asset = baseline_result.get("asset")
            timeframe = baseline_result.get("timeframe")
        confidence = float(baseline_result.get("confidence", 0.0) or 0.0)

        if label_manifest is None:
            correct = int(baseline_result.get("correct", int(predicted == true_direction)) or 0)
        else:
            correct = int(predicted == true_direction)

        rows.append(
            {
                "sample_file": sample_file,
                "predicted": predicted,
                "true_direction": true_direction,
                "confidence": confidence,
                "correct": correct,
                "is_neutral_prediction": int(predicted == "NEUTRAL"),
                "asset": asset,
                "timeframe": timeframe,
                "future_return_pct": baseline_result.get("future_return_pct"),
                "horizon_correct_count": int(baseline_result.get("horizon_correct_count", baseline_result.get("correct", 0)) or 0),
                "horizon_total_count": int(baseline_result.get("horizon_total_count", 1) or 1),
                "horizon_step_accuracy": float(baseline_result.get("horizon_step_accuracy", baseline_result.get("correct", 0)) or 0.0),
                "is_neutral_move": int(baseline_result.get("is_neutral_move", 0) or 0),
            }
        )

    evaluation_df = pd.DataFrame(rows)
    if evaluation_df.empty:
        return {
            "num_samples": 0,
            "accuracy": None,
            "avg_confidence": None,
            "details": [],
        }

    accuracy = float(evaluation_df["correct"].mean())
    horizon_correct = int(evaluation_df["horizon_correct_count"].sum()) if "horizon_correct_count" in evaluation_df.columns else int(evaluation_df["correct"].sum())
    horizon_total = int(evaluation_df["horizon_total_count"].sum()) if "horizon_total_count" in evaluation_df.columns else int(len(evaluation_df))
    horizon_step_accuracy = None if horizon_total == 0 else round(float(horizon_correct / horizon_total), 4)
    avg_confidence = float(evaluation_df["confidence"].mean())

    asset_accuracy = (
        evaluation_df.groupby("asset")["correct"].mean().round(4).to_dict()
        if "asset" in evaluation_df.columns
        else {}
    )
    actionable_df = evaluation_df[evaluation_df["is_neutral_prediction"] == 0]
    actionable_accuracy = None
    actionable_coverage = 0.0
    actionable_num_samples = 0
    if not actionable_df.empty:
        actionable_accuracy = round(float(actionable_df["correct"].mean()), 4)
        actionable_num_samples = int(len(actionable_df))
        actionable_coverage = round(float(len(actionable_df) / len(evaluation_df)), 4)

    strong_move_df = evaluation_df[evaluation_df["is_neutral_move"] == 0]
    filtered_accuracy = None
    filtered_num_samples = 0
    if not strong_move_df.empty:
        filtered_accuracy = round(float(strong_move_df["correct"].mean()), 4)
        filtered_num_samples = int(len(strong_move_df))

    strong_move_actionable_df = strong_move_df[strong_move_df["is_neutral_prediction"] == 0]
    filtered_actionable_accuracy = None
    filtered_actionable_num_samples = 0
    if not strong_move_actionable_df.empty:
        filtered_actionable_accuracy = round(float(strong_move_actionable_df["correct"].mean()), 4)
        filtered_actionable_num_samples = int(len(strong_move_actionable_df))

    return {
        "num_samples": int(len(evaluation_df)),
        "accuracy": round(accuracy, 4),
        "horizon_step_accuracy": horizon_step_accuracy,
        "horizon_correct_count": horizon_correct,
        "horizon_total_count": horizon_total,
        "avg_confidence": round(avg_confidence, 4),
        "num_neutral_predictions": int(evaluation_df["is_neutral_prediction"].sum()),
        "actionable_num_samples": actionable_num_samples,
        "actionable_coverage": actionable_coverage,
        "actionable_accuracy": actionable_accuracy,
        "num_neutral_moves": int(evaluation_df["is_neutral_move"].sum()),
        "filtered_num_samples": filtered_num_samples,
        "filtered_accuracy_ex_neutral": filtered_accuracy,
        "filtered_actionable_num_samples": filtered_actionable_num_samples,
        "filtered_actionable_accuracy": filtered_actionable_accuracy,
        "asset_accuracy": asset_accuracy,
        "details": evaluation_df.to_dict(orient="records"),
    }


def evaluate_confidence_buckets(
    batch_results: List[Dict[str, Any]],
    label_manifest: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    Check whether higher-confidence algorithmic signals are actually more accurate.

    This is important because a good baseline is not only directional, but also
    calibrated: strong signals should be better than weak ones.
    """

    base_eval = evaluate_directional_accuracy(batch_results, label_manifest)
    details = base_eval.get("details", [])
    if not details:
        return {
            "low": None,
            "medium": None,
            "high": None,
        }

    df = pd.DataFrame(details)
    if df.empty:
        return {"low": None, "medium": None, "high": None}

    def bucket(conf: float) -> str:
        if conf >= 0.66:
            return "high"
        if conf >= 0.33:
            return "medium"
        return "low"

    df["bucket"] = df["confidence"].apply(bucket)
    bucket_accuracy = (
        df.groupby("bucket")["correct"]
        .mean()
        .round(4)
        .to_dict()
    )
    bucket_counts = df.groupby("bucket").size().to_dict()
    return {
        "accuracy": bucket_accuracy,
        "counts": bucket_counts,
    }


def summarize_batch_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quick summary for the pure algorithm layer without any external labels.

    This works because baseline results now carry internally generated future
    labels from the benchmark sample split.
    """

    accuracy_report = evaluate_directional_accuracy(batch_results, label_manifest=None)
    confidence_report = evaluate_confidence_buckets(batch_results, label_manifest=None)

    return {
        "accuracy_report": accuracy_report,
        "confidence_buckets": confidence_report,
    }
