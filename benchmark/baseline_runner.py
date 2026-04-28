from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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

from analysis_io import (
    analysis_output_to_dict,
    prepare_benchmark_sample,
    run_algorithm_analysis,
)


@dataclass(frozen=True)
class BaselineResult:
    """Deterministic baseline result used for automated evaluation."""

    asset: str
    timeframe: str
    sample_file: str
    dominant_side: str
    confidence: float
    consensus_level: str
    risk_level: str
    future_horizon: int
    true_direction: str
    future_return_pct: float
    horizon_correct_count: int
    horizon_total_count: int
    horizon_step_accuracy: float
    future_step_directions: List[str]
    neutral_threshold_pct: float
    is_neutral_move: int
    correct: int
    justification: List[str]


def load_benchmark_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a benchmark sample file into a DataFrame."""

    return pd.read_csv(csv_path)


def run_algorithmic_baseline(
    csv_path: str | Path,
    asset: str,
    timeframe: str,
    window_size: int = 45,
    future_horizon: int = 3,
    neutral_threshold_pct: float = 0.15,
) -> Dict[str, Any]:
    """
    Run a pure algorithmic baseline on one benchmark sample.

    Why this matters:
    Before comparing KuantAgent against QuantAgent, we should first know how far
    the deterministic algorithm layer can go on its own. This gives us a clean
    lower-bound baseline and supports ablation experiments later.
    """

    raw_df = load_benchmark_csv(csv_path)
    benchmark_sample = prepare_benchmark_sample(
        raw_df=raw_df,
        asset=asset,
        timeframe=timeframe,
        window_size=window_size,
        future_horizon=future_horizon,
    )
    analysis_output = run_algorithm_analysis(benchmark_sample.analysis_input)
    decision_context = analysis_output.decision_features
    input_close = float(benchmark_sample.analysis_input.kline_data["Close"][-1])
    future_close = float(benchmark_sample.future_df.iloc[-1]["Close"])
    future_return_pct = float((future_close / input_close - 1.0) * 100.0)
    true_direction = "LONG" if future_return_pct >= 0 else "SHORT"
    comparison_closes = [input_close] + [float(value) for value in benchmark_sample.future_df["Close"].tolist()]
    future_step_directions: List[str] = []
    for prev_close, next_close in zip(comparison_closes, comparison_closes[1:]):
        future_step_directions.append("LONG" if next_close >= prev_close else "SHORT")
    predicted_side = str(decision_context.get("dominant_side", "neutral")).upper()
    horizon_correct_count = int(sum(1 for direction in future_step_directions if predicted_side == direction))
    horizon_total_count = len(future_step_directions)
    horizon_step_accuracy = 0.0 if horizon_total_count == 0 else horizon_correct_count / horizon_total_count
    is_neutral_move = int(abs(future_return_pct) < neutral_threshold_pct)
    correct = int(predicted_side == true_direction)
    if future_horizon > 1 and horizon_total_count > 0:
        correct = int(horizon_correct_count >= ((horizon_total_count // 2) + 1))

    baseline_result = BaselineResult(
        asset=asset,
        timeframe=timeframe,
        sample_file=str(csv_path),
        dominant_side=predicted_side,
        confidence=float(decision_context.get("algorithmic_confidence", 0.0) or 0.0),
        consensus_level=str(decision_context.get("consensus_level", "weak")),
        risk_level=str(decision_context.get("risk_level", "unknown")),
        future_horizon=future_horizon,
        true_direction=true_direction,
        future_return_pct=round(future_return_pct, 4),
        horizon_correct_count=horizon_correct_count,
        horizon_total_count=horizon_total_count,
        horizon_step_accuracy=round(horizon_step_accuracy, 4),
        future_step_directions=future_step_directions,
        neutral_threshold_pct=neutral_threshold_pct,
        is_neutral_move=is_neutral_move,
        correct=correct,
        justification=list(decision_context.get("supporting_reasons", [])),
    )

    return {
        "baseline_result": baseline_result.__dict__,
        "analysis_output": analysis_output_to_dict(analysis_output),
    }


def save_baseline_result(result: Dict[str, Any], output_path: str | Path) -> None:
    """Persist a baseline result to JSON for later comparison or reporting."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
