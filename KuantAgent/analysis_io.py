from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd

from data_processing import (
    OHLCWindowConfig,
    dataframe_to_ohlc_dict,
    normalize_ohlc_dataframe,
    prepare_market_data,
)
from feature_extraction import FeatureConfig, extract_market_features


@dataclass(frozen=True)
class AnalysisInput:
    """Canonical analysis input shared by online inference and offline experiments."""

    asset: str
    timeframe: str
    window_size: int
    kline_data: Dict[str, List[Any]]
    forecast_horizon_bars: int = 1
    macro_timeframe: str | None = None
    macro_kline_data: Dict[str, List[Any]] | None = None
    case_context: Dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisOutput:
    """Canonical analysis output for benchmarking, evaluation, and logging."""

    asset: str
    timeframe: str
    window_size: int
    forecast_horizon_bars: int
    indicator_features: Dict[str, Any]
    pattern_features: Dict[str, Any]
    trend_features: Dict[str, Any]
    risk_features: Dict[str, Any]
    decision_features: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkSample:
    """Offline benchmark sample split into input window and future horizon."""

    analysis_input: AnalysisInput
    future_df: pd.DataFrame
    future_horizon: int


def prepare_analysis_input(
    raw_df: pd.DataFrame,
    asset: str,
    timeframe: str,
    window_config: OHLCWindowConfig | None = None,
) -> AnalysisInput:
    """
    Convert raw market data into the standardized input object used by KuantAgent.

    Why this matters:
    A stable experiment pipeline needs one canonical input format so that web
    inference, batch evaluation, and baseline strategies all operate on the same
    sample definition.
    """

    window_config = window_config or OHLCWindowConfig(window_size=45, keep_volume=False)
    prepared = prepare_market_data(raw_df, config=window_config)
    return AnalysisInput(
        asset=asset,
        timeframe=timeframe,
        window_size=window_config.window_size,
        forecast_horizon_bars=1,
        kline_data=prepared["ohlc_dict"],
    )


def prepare_benchmark_sample(
    raw_df: pd.DataFrame,
    asset: str,
    timeframe: str,
    window_size: int = 45,
    future_horizon: int = 1,
) -> BenchmarkSample:
    """
    Split one benchmark CSV into an input window and a future label horizon.

    Convention:
    - first `window_size` rows are the observable past
    - next `future_horizon` rows are the evaluation future

    Why this matters:
    Offline evaluation needs a clean separation between what the algorithm is
    allowed to see and what is held out as the future target.
    """

    normalized_df = normalize_ohlc_dataframe(raw_df, keep_volume=False)
    if len(normalized_df) < window_size + future_horizon:
        raise ValueError(
            f"Not enough rows for benchmark split: need at least {window_size + future_horizon}, got {len(normalized_df)}"
        )

    input_df = normalized_df.iloc[:window_size].reset_index(drop=True)
    future_df = normalized_df.iloc[window_size : window_size + future_horizon].reset_index(drop=True)
    window_config = OHLCWindowConfig(window_size=window_size, keep_volume=False)

    analysis_input = AnalysisInput(
        asset=asset,
        timeframe=timeframe,
        window_size=window_size,
        forecast_horizon_bars=future_horizon,
        kline_data=dataframe_to_ohlc_dict(input_df, config=window_config),
    )
    return BenchmarkSample(
        analysis_input=analysis_input,
        future_df=future_df,
        future_horizon=future_horizon,
    )


def run_algorithm_analysis(
    analysis_input: AnalysisInput,
    feature_config: FeatureConfig | None = None,
) -> AnalysisOutput:
    """
    Run the full deterministic analysis layer without invoking any LLM.

    This function is the core building block for future baselines because it
    lets us evaluate the algorithm layer on its own.
    """

    feature_config = feature_config or FeatureConfig()
    ohlc_df = pd.DataFrame(analysis_input.kline_data).copy()
    ohlc_df = normalize_ohlc_dataframe(ohlc_df, keep_volume=False)
    features = extract_market_features(ohlc_df, config=feature_config)
    decision_features = dict(features["decision_features"])
    decision_features["forecast_horizon_bars"] = analysis_input.forecast_horizon_bars

    return AnalysisOutput(
        asset=analysis_input.asset,
        timeframe=analysis_input.timeframe,
        window_size=analysis_input.window_size,
        forecast_horizon_bars=analysis_input.forecast_horizon_bars,
        indicator_features=features["indicator_features"],
        pattern_features=features["pattern_features"],
        trend_features=features["trend_features"],
        risk_features=features["risk_features"],
        decision_features=decision_features,
    )


def analysis_output_to_dict(output: AnalysisOutput) -> Dict[str, Any]:
    """Serialize the canonical output object into a plain dict."""

    return asdict(output)
