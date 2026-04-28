"""
Algorithm analysis node for KuantAgent.

This node formalizes the second layer of the thesis architecture inside the
LangGraph workflow. It computes structured market features before any agent
starts its semantic reasoning.
"""

import pandas as pd

from context_enrichment import build_case_context
from feature_extraction import extract_market_features


def create_algorithm_analysis_node():
    """Create the node that transforms OHLC input into structured features."""

    def algorithm_analysis_node(state):
        kline_data = state["kline_data"]
        forecast_horizon_bars = int(state.get("forecast_horizon_bars", 1) or 1)
        ohlc_df = pd.DataFrame(kline_data).copy()
        ohlc_df["Datetime"] = pd.to_datetime(ohlc_df["Datetime"], errors="coerce")

        market_features = extract_market_features(
            ohlc_df,
            forecast_horizon_bars=forecast_horizon_bars,
        )
        decision_features = dict(market_features["decision_features"])
        decision_features["forecast_horizon_bars"] = forecast_horizon_bars
        case_context = build_case_context(
            asset=str(state.get("stock_name", "")),
            timeframe=str(state.get("time_frame", "")),
            pattern_features=market_features["pattern_features"],
            trend_features=market_features["trend_features"],
            risk_features=market_features["risk_features"],
            decision_features=decision_features,
            macro_timeframe=str(state.get("macro_timeframe", "") or ""),
            macro_kline_data=state.get("macro_kline_data"),
        )

        return {
            "indicator_features": market_features["indicator_features"],
            "pattern_features": market_features["pattern_features"],
            "trend_features": market_features["trend_features"],
            "risk_features": market_features["risk_features"],
            "decision_features": decision_features,
            "case_context": case_context,
        }

    return algorithm_analysis_node
