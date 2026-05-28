from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import talib


@dataclass(frozen=True)
class FeatureConfig:
    """Shared configuration for the algorithm analysis layer."""

    rsi_period: int = 14
    roc_period: int = 10
    stoch_period: int = 14
    willr_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    bollinger_period: int = 20
    ma_fast_period: int = 10
    ma_slow_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    pivot_order: int = 2
    pattern_tolerance: float = 0.02
    neckline_breakout_buffer: float = 0.003


def _safe_last(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _safe_prev(series: pd.Series) -> float | None:
    valid = series.dropna()
    if len(valid) < 2:
        return None
    return float(valid.iloc[-2])


def _safe_mean(series: pd.Series, window: int) -> float | None:
    valid = series.dropna()
    if len(valid) < window:
        return None
    return float(valid.iloc[-window:].mean())


def _detect_cross(
    prev_a: float | None,
    curr_a: float | None,
    prev_b: float | None,
    curr_b: float | None,
) -> str:
    if None in {prev_a, curr_a, prev_b, curr_b}:
        return "none"
    if prev_a <= prev_b and curr_a > curr_b:
        return "bullish_cross"
    if prev_a >= prev_b and curr_a < curr_b:
        return "bearish_cross"
    return "none"


def _classify_rsi(rsi_value: float | None) -> str:
    if rsi_value is None:
        return "unknown"
    if rsi_value >= 70:
        return "overbought"
    if rsi_value <= 30:
        return "oversold"
    if rsi_value >= 55:
        return "bullish"
    if rsi_value <= 45:
        return "bearish"
    return "neutral"


def _classify_willr(willr_value: float | None) -> str:
    if willr_value is None:
        return "unknown"
    if willr_value >= -20:
        return "overbought"
    if willr_value <= -80:
        return "oversold"
    return "neutral"


def _classify_stoch(k_value: float | None, d_value: float | None) -> str:
    if k_value is None or d_value is None:
        return "unknown"
    if k_value >= 80 and d_value >= 80:
        return "overbought"
    if k_value <= 20 and d_value <= 20:
        return "oversold"
    if k_value > d_value:
        return "bullish"
    if k_value < d_value:
        return "bearish"
    return "neutral"


def _classify_adx(adx_value: float | None) -> str:
    if adx_value is None:
        return "unknown"
    if adx_value >= 30:
        return "strong"
    if adx_value >= 20:
        return "moderate"
    return "weak"


def _classify_volatility(atr_pct: float | None) -> str:
    if atr_pct is None:
        return "unknown"
    if atr_pct >= 3:
        return "high"
    if atr_pct >= 1.5:
        return "medium"
    return "low"


def _classify_location_ratio(location_ratio: float | None) -> str:
    """Describe where price sits inside the recent support-resistance range."""

    if location_ratio is None:
        return "unknown"
    if location_ratio <= 0.2:
        return "near_support"
    if location_ratio >= 0.8:
        return "near_resistance"
    return "mid_range"


def _classify_breakout_state(
    latest_close: float,
    support_level: float,
    resistance_level: float,
    buffer_pct: float = 0.0025,
) -> str:
    """Classify whether price is breaking out of the recent range."""

    if latest_close > resistance_level * (1 + buffer_pct):
        return "bullish_breakout"
    if latest_close < support_level * (1 - buffer_pct):
        return "bearish_breakdown"
    if latest_close > resistance_level * (1 - buffer_pct):
        return "testing_resistance"
    if latest_close < support_level * (1 + buffer_pct):
        return "testing_support"
    return "inside_range"


def _count_level_touches(
    values: pd.Series,
    level: float,
    tolerance_ratio: float = 0.004,
) -> int:
    """Count how many recent points interacted with a target level."""

    if level == 0:
        return 0
    tolerance = abs(level) * tolerance_ratio
    return int((values.sub(level).abs() <= tolerance).sum())


def _detect_simple_divergence(
    price_series: pd.Series,
    oscillator_series: pd.Series,
) -> str:
    """
    Detect a lightweight divergence signal.

    Bullish divergence:
    price makes a lower recent low while the oscillator makes a higher low.
    Bearish divergence:
    price makes a higher recent high while the oscillator makes a lower high.
    """

    clean_osc = oscillator_series.dropna()
    if len(price_series) < 8 or len(clean_osc) < 8:
        return "none"

    recent_price = price_series.tail(8).reset_index(drop=True)
    recent_osc = clean_osc.tail(8).reset_index(drop=True)

    price_low_1 = float(recent_price.iloc[:4].min())
    price_low_2 = float(recent_price.iloc[4:].min())
    osc_low_1 = float(recent_osc.iloc[:4].min())
    osc_low_2 = float(recent_osc.iloc[4:].min())

    price_high_1 = float(recent_price.iloc[:4].max())
    price_high_2 = float(recent_price.iloc[4:].max())
    osc_high_1 = float(recent_osc.iloc[:4].max())
    osc_high_2 = float(recent_osc.iloc[4:].max())

    if price_low_2 < price_low_1 and osc_low_2 > osc_low_1:
        return "bullish_divergence"
    if price_high_2 > price_high_1 and osc_high_2 < osc_high_1:
        return "bearish_divergence"
    return "none"


def _compute_regression_stats(series: pd.Series) -> Dict[str, float]:
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    fitted = slope * x + intercept
    residual = series - fitted
    total_var = np.sum((series - series.mean()) ** 2)
    residual_var = np.sum(residual**2)
    r_squared = 0.0 if total_var == 0 else float(1 - residual_var / total_var)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "residual_std": float(np.std(residual)),
    }


def _safe_pct_change(base: float, current: float) -> float:
    if abs(base) <= 1e-8:
        return 0.0
    return float((current / base - 1.0) * 100.0)


def _compute_shape_score(left_value: float, right_value: float) -> float:
    scale = max(abs(left_value), abs(right_value), 1e-8)
    return float(max(0.0, 1.0 - abs(left_value - right_value) / scale))


def _compute_candle_geometry(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> Dict[str, float | str]:
    total_range = max(high_price - low_price, 1e-8)
    body = abs(close_price - open_price)
    upper_wick = max(0.0, high_price - max(open_price, close_price))
    lower_wick = max(0.0, min(open_price, close_price) - low_price)
    direction = "bullish" if close_price > open_price else "bearish" if close_price < open_price else "neutral"
    return {
        "candle_direction": direction,
        "body_ratio": round(body / total_range, 4),
        "upper_wick_ratio": round(upper_wick / total_range, 4),
        "lower_wick_ratio": round(lower_wick / total_range, 4),
        "range_pct": round(float(total_range / max(abs(close_price), 1e-8) * 100.0), 4),
    }


def _compute_line_stats(points: List[Dict[str, float]]) -> Dict[str, float | str]:
    if len(points) < 2:
        return {
            "slope": 0.0,
            "fit_r2": 0.0,
            "direction": "unknown",
        }
    x = pd.Series([point["index"] for point in points], dtype=float)
    y = pd.Series([point["value"] for point in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total_var = np.sum((y - y.mean()) ** 2)
    residual_var = np.sum((y - fitted) ** 2)
    r_squared = 0.0 if total_var == 0 else float(1 - residual_var / total_var)
    direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
    return {
        "slope": round(float(slope), 6),
        "fit_r2": round(r_squared, 4),
        "direction": direction,
    }


def _count_consecutive_direction(close: pd.Series, direction: str, lookback: int = 6) -> int:
    values = close.tail(lookback + 1).reset_index(drop=True)
    if len(values) < 2:
        return 0
    count = 0
    for idx in range(len(values) - 1, 0, -1):
        diff = float(values.iloc[idx] - values.iloc[idx - 1])
        if direction == "up" and diff > 0:
            count += 1
        elif direction == "down" and diff < 0:
            count += 1
        else:
            break
    return count


def _pivot_points(series: pd.Series, order: int) -> Dict[str, List[Dict[str, float]]]:
    values = series.reset_index(drop=True)
    highs: List[Dict[str, float]] = []
    lows: List[Dict[str, float]] = []

    for idx in range(order, len(values) - order):
        window = values.iloc[idx - order : idx + order + 1]
        current = values.iloc[idx]

        if current == window.max() and (window == current).sum() == 1:
            highs.append({"index": idx, "value": float(current)})
        if current == window.min() and (window == current).sum() == 1:
            lows.append({"index": idx, "value": float(current)})

    return {"pivot_highs": highs, "pivot_lows": lows}


def _infer_high_low_structure(
    pivot_highs: List[Dict[str, float]],
    pivot_lows: List[Dict[str, float]],
) -> str:
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "insufficient_structure"

    high_1, high_2 = pivot_highs[-2], pivot_highs[-1]
    low_1, low_2 = pivot_lows[-2], pivot_lows[-1]

    higher_high = high_2["value"] > high_1["value"]
    lower_high = high_2["value"] < high_1["value"]
    higher_low = low_2["value"] > low_1["value"]
    lower_low = low_2["value"] < low_1["value"]

    if higher_high and higher_low:
        return "higher_high_higher_low"
    if lower_high and lower_low:
        return "lower_high_lower_low"
    if lower_high and higher_low:
        return "compression"
    if higher_high and lower_low:
        return "expansion"
    return "mixed"


def _merge_recent_swings(
    pivot_highs: List[Dict[str, float]],
    pivot_lows: List[Dict[str, float]],
    limit: int = 6,
) -> List[Dict[str, float | str]]:
    swings: List[Dict[str, float | str]] = []
    for pivot in pivot_highs[-limit:]:
        swings.append(
            {
                "type": "high",
                "index": int(pivot["index"]),
                "value": float(pivot["value"]),
            }
        )
    for pivot in pivot_lows[-limit:]:
        swings.append(
            {
                "type": "low",
                "index": int(pivot["index"]),
                "value": float(pivot["value"]),
            }
        )
    swings.sort(key=lambda item: int(item["index"]))
    return swings[-limit:]


def _compute_swing_geometry(
    pivot_highs: List[Dict[str, float]],
    pivot_lows: List[Dict[str, float]],
    latest_close: float,
) -> Dict[str, object]:
    recent_swings = _merge_recent_swings(pivot_highs, pivot_lows, limit=6)
    recent_highs = pivot_highs[-3:]
    recent_lows = pivot_lows[-3:]

    higher_highs = 0
    lower_highs = 0
    higher_lows = 0
    lower_lows = 0

    for left, right in zip(recent_highs[:-1], recent_highs[1:]):
        if right["value"] > left["value"]:
            higher_highs += 1
        elif right["value"] < left["value"]:
            lower_highs += 1
    for left, right in zip(recent_lows[:-1], recent_lows[1:]):
        if right["value"] > left["value"]:
            higher_lows += 1
        elif right["value"] < left["value"]:
            lower_lows += 1

    swing_bias = "mixed"
    if higher_highs >= 1 and higher_lows >= 1:
        swing_bias = "bullish"
    elif lower_highs >= 1 and lower_lows >= 1:
        swing_bias = "bearish"
    elif higher_lows >= 1 and lower_highs >= 1:
        swing_bias = "compression"
    elif higher_highs >= 1 and lower_lows >= 1:
        swing_bias = "expansion"

    amplitudes: List[float] = []
    for left, right in zip(recent_swings[:-1], recent_swings[1:]):
        left_value = float(left["value"])
        right_value = float(right["value"])
        amplitudes.append(abs(_safe_pct_change(left_value, right_value)))

    swing_amplitude_pct_avg = float(np.mean(amplitudes)) if amplitudes else 0.0
    compression_score = 0.0
    if len(amplitudes) >= 2:
        early_avg = float(np.mean(amplitudes[: max(1, len(amplitudes) // 2)]))
        late_avg = float(np.mean(amplitudes[max(1, len(amplitudes) // 2) :]))
        if early_avg > 1e-8:
            compression_score = max(0.0, min(1.0, 1.0 - late_avg / early_avg))

    swing_quality_score = 0.0
    if swing_bias == "bullish":
        swing_quality_score = min(1.0, 0.35 + 0.18 * higher_highs + 0.18 * higher_lows)
    elif swing_bias == "bearish":
        swing_quality_score = min(1.0, 0.35 + 0.18 * lower_highs + 0.18 * lower_lows)
    elif swing_bias == "compression":
        swing_quality_score = min(1.0, 0.32 + 0.28 * compression_score)

    structure_break_state = "none"
    structure_break_bias = "neutral"
    structure_break_distance_pct = 0.0
    if recent_highs and latest_close > float(recent_highs[-1]["value"]):
        structure_break_state = "bullish_break"
        structure_break_bias = "bullish"
        structure_break_distance_pct = round(
            _safe_pct_change(float(recent_highs[-1]["value"]), latest_close), 4
        )
    elif recent_lows and latest_close < float(recent_lows[-1]["value"]):
        structure_break_state = "bearish_break"
        structure_break_bias = "bearish"
        structure_break_distance_pct = round(
            _safe_pct_change(latest_close, float(recent_lows[-1]["value"])), 4
        )

    return {
        "swing_sequence": recent_swings,
        "swing_bias": swing_bias,
        "swing_quality_score": round(swing_quality_score, 4),
        "swing_compression_score": round(compression_score, 4),
        "swing_amplitude_pct_avg": round(swing_amplitude_pct_avg, 4),
        "higher_high_count": higher_highs,
        "higher_low_count": higher_lows,
        "lower_high_count": lower_highs,
        "lower_low_count": lower_lows,
        "structure_break_state": structure_break_state,
        "structure_break_bias": structure_break_bias,
        "structure_break_distance_pct": structure_break_distance_pct,
    }


def _compute_breakout_microstructure(
    ohlc_df: pd.DataFrame,
    support_level: float,
    resistance_level: float,
) -> Dict[str, object]:
    if len(ohlc_df) < 5:
        return {
            "latest_close_near_high_ratio": 0.0,
            "latest_close_near_low_ratio": 0.0,
            "recent_breakout_followthrough_score": 0.0,
            "recent_breakout_failure_score": 0.0,
            "level_reclaim_state": "none",
            "rejection_wick_bias": "neutral",
        }

    recent = ohlc_df.tail(5).reset_index(drop=True)
    latest = recent.iloc[-1]
    prev = recent.iloc[-2]
    latest_open = float(latest["Open"])
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    latest_close = float(latest["Close"])
    prev_close = float(prev["Close"])
    total_range = max(latest_high - latest_low, 1e-8)
    latest_close_near_high_ratio = max(0.0, min(1.0, (latest_close - latest_low) / total_range))
    latest_close_near_low_ratio = max(0.0, min(1.0, (latest_high - latest_close) / total_range))

    followthrough_score = 0.0
    failure_score = 0.0
    level_reclaim_state = "none"

    if prev_close <= resistance_level and latest_close > resistance_level:
        followthrough_score += min(1.0, max(0.0, _safe_pct_change(resistance_level, latest_close) / 0.45))
        if latest_open <= resistance_level:
            level_reclaim_state = "bullish_reclaim"
    elif prev_close >= support_level and latest_close < support_level:
        followthrough_score += min(1.0, max(0.0, _safe_pct_change(latest_close, support_level) / 0.45))
        if latest_open >= support_level:
            level_reclaim_state = "bearish_reclaim"

    recent_closes = recent["Close"].astype(float).tolist()
    if max(recent_closes[:-1]) > resistance_level and latest_close < resistance_level:
        failure_score += 0.55
        level_reclaim_state = "failed_bullish_breakout_reentry"
    if min(recent_closes[:-1]) < support_level and latest_close > support_level:
        failure_score += 0.55
        level_reclaim_state = "failed_bearish_breakdown_reentry"

    rejection_wick_bias = "neutral"
    candle_geometry = _compute_candle_geometry(latest_open, latest_high, latest_low, latest_close)
    if (
        float(candle_geometry["upper_wick_ratio"]) >= 0.45
        and latest_close < latest_open
        and latest_high >= resistance_level * 0.998
    ):
        rejection_wick_bias = "bearish"
        failure_score += 0.18
    elif (
        float(candle_geometry["lower_wick_ratio"]) >= 0.45
        and latest_close > latest_open
        and latest_low <= support_level * 1.002
    ):
        rejection_wick_bias = "bullish"
        failure_score += 0.18

    return {
        "latest_close_near_high_ratio": round(latest_close_near_high_ratio, 4),
        "latest_close_near_low_ratio": round(latest_close_near_low_ratio, 4),
        "recent_breakout_followthrough_score": round(min(1.0, followthrough_score), 4),
        "recent_breakout_failure_score": round(min(1.0, failure_score), 4),
        "level_reclaim_state": level_reclaim_state,
        "rejection_wick_bias": rejection_wick_bias,
    }


def _detect_double_bottom(
    close: pd.Series,
    pivot_lows: List[Dict[str, float]],
    pivot_highs: List[Dict[str, float]],
    tolerance: float,
    breakout_buffer: float,
) -> Dict[str, object] | None:
    if len(pivot_lows) < 2:
        return None

    left_low, right_low = pivot_lows[-2], pivot_lows[-1]
    if right_low["index"] <= left_low["index"]:
        return None

    similar_lows = abs(left_low["value"] - right_low["value"]) / max(left_low["value"], 1e-8)
    if similar_lows > tolerance:
        return None

    neckline_candidates = [
        pivot for pivot in pivot_highs if left_low["index"] < pivot["index"] < right_low["index"]
    ]
    if not neckline_candidates:
        return None

    neckline = max(candidate["value"] for candidate in neckline_candidates)
    current_close = float(close.iloc[-1])
    breakout_confirmed = current_close > neckline * (1 + breakout_buffer)
    completion = min(1.0, max(0.0, 1.0 - similar_lows / max(tolerance, 1e-8)))

    return {
        "pattern": "double_bottom",
        "pattern_bias": "bullish",
        "pattern_family": "classic",
        "pattern_confidence": round(0.55 + 0.35 * completion + (0.1 if breakout_confirmed else 0.0), 4),
        "pattern_completed": True,
        "breakout_confirmed": breakout_confirmed,
        "neckline_level": float(neckline),
        "support_level": float(min(left_low["value"], right_low["value"])),
        "pattern_span": int(right_low["index"] - left_low["index"]),
        "symmetry_score": round(_compute_shape_score(left_low["value"], right_low["value"]), 4),
        "neckline_distance_pct": round(_safe_pct_change(current_close, neckline), 4),
        "breakout_margin_pct": round(_safe_pct_change(neckline, current_close), 4),
    }


def _detect_double_top(
    close: pd.Series,
    pivot_highs: List[Dict[str, float]],
    pivot_lows: List[Dict[str, float]],
    tolerance: float,
    breakout_buffer: float,
) -> Dict[str, object] | None:
    if len(pivot_highs) < 2:
        return None

    left_high, right_high = pivot_highs[-2], pivot_highs[-1]
    if right_high["index"] <= left_high["index"]:
        return None

    similar_highs = abs(left_high["value"] - right_high["value"]) / max(left_high["value"], 1e-8)
    if similar_highs > tolerance:
        return None

    neckline_candidates = [
        pivot for pivot in pivot_lows if left_high["index"] < pivot["index"] < right_high["index"]
    ]
    if not neckline_candidates:
        return None

    neckline = min(candidate["value"] for candidate in neckline_candidates)
    current_close = float(close.iloc[-1])
    breakout_confirmed = current_close < neckline * (1 - breakout_buffer)
    completion = min(1.0, max(0.0, 1.0 - similar_highs / max(tolerance, 1e-8)))

    return {
        "pattern": "double_top",
        "pattern_bias": "bearish",
        "pattern_family": "classic",
        "pattern_confidence": round(0.55 + 0.35 * completion + (0.1 if breakout_confirmed else 0.0), 4),
        "pattern_completed": True,
        "breakout_confirmed": breakout_confirmed,
        "neckline_level": float(neckline),
        "resistance_level": float(max(left_high["value"], right_high["value"])),
        "pattern_span": int(right_high["index"] - left_high["index"]),
        "symmetry_score": round(_compute_shape_score(left_high["value"], right_high["value"]), 4),
        "neckline_distance_pct": round(_safe_pct_change(neckline, current_close), 4),
        "breakout_margin_pct": round(_safe_pct_change(current_close, neckline), 4),
    }


def _detect_triangle(
    pivot_highs: List[Dict[str, float]],
    pivot_lows: List[Dict[str, float]],
) -> Dict[str, object] | None:
    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None

    high_values = [pivot["value"] for pivot in pivot_highs[-3:]]
    low_values = [pivot["value"] for pivot in pivot_lows[-3:]]

    high_reg = _compute_regression_stats(pd.Series(high_values))
    low_reg = _compute_regression_stats(pd.Series(low_values))

    descending_highs = high_reg["slope"] < 0
    ascending_lows = low_reg["slope"] > 0

    if descending_highs and ascending_lows:
        return {
            "pattern": "symmetrical_triangle",
            "pattern_bias": "neutral",
            "pattern_family": "classic",
            "pattern_confidence": round(min(0.85, 0.55 + 0.15 * (high_reg["r_squared"] + low_reg["r_squared"])), 4),
            "pattern_completed": False,
            "breakout_confirmed": False,
            "upper_slope": float(high_reg["slope"]),
            "lower_slope": float(low_reg["slope"]),
        }

    return None


def _detect_v_reversal(
    close: pd.Series,
) -> Dict[str, object] | None:
    """Detect a lightweight V-shaped reversal or exhaustion turn."""

    if len(close) < 12:
        return None

    recent = close.tail(12).reset_index(drop=True)
    pivot_idx = int(recent.idxmin())
    pivot_value = float(recent.iloc[pivot_idx])
    first_close = float(recent.iloc[0])
    last_close = float(recent.iloc[-1])

    if 2 <= pivot_idx <= 8:
        decline_pct = 0.0 if first_close == 0 else (pivot_value / first_close - 1.0) * 100.0
        rebound_pct = 0.0 if pivot_value == 0 else (last_close / pivot_value - 1.0) * 100.0
        right_leg_strength = rebound_pct - abs(min(decline_pct, 0.0))
        if decline_pct <= -0.7 and rebound_pct >= 1.0:
            confidence = 0.58
            if rebound_pct >= 1.8:
                confidence += 0.08
            if last_close > float(recent.iloc[-3]):
                confidence += 0.05
            if right_leg_strength >= -0.2:
                confidence += 0.04
            return {
                "pattern": "v_shaped_reversal",
                "pattern_bias": "bullish",
                "pattern_family": "reversal",
                "pattern_confidence": round(min(0.84, confidence), 4),
                "pattern_completed": True,
                "breakout_confirmed": rebound_pct >= 1.5,
                "pivot_level": pivot_value,
                "pattern_span": 12 - pivot_idx,
                "left_leg_pct": round(abs(decline_pct), 4),
                "right_leg_pct": round(rebound_pct, 4),
                "recovery_ratio": round(0.0 if abs(decline_pct) <= 1e-8 else rebound_pct / abs(decline_pct), 4),
                "leg_symmetry_score": round(_compute_shape_score(abs(decline_pct), rebound_pct), 4),
            }

    pivot_idx = int(recent.idxmax())
    pivot_value = float(recent.iloc[pivot_idx])
    if 2 <= pivot_idx <= 8:
        rally_pct = 0.0 if first_close == 0 else (pivot_value / first_close - 1.0) * 100.0
        fallback_pct = 0.0 if pivot_value == 0 else (last_close / pivot_value - 1.0) * 100.0
        right_leg_strength = abs(fallback_pct) - max(rally_pct, 0.0)
        if rally_pct >= 0.7 and fallback_pct <= -1.0:
            confidence = 0.58
            if fallback_pct <= -1.8:
                confidence += 0.08
            if last_close < float(recent.iloc[-3]):
                confidence += 0.05
            if right_leg_strength >= -0.2:
                confidence += 0.04
            return {
                "pattern": "inverted_v_reversal",
                "pattern_bias": "bearish",
                "pattern_family": "reversal",
                "pattern_confidence": round(min(0.84, confidence), 4),
                "pattern_completed": True,
                "breakout_confirmed": fallback_pct <= -1.5,
                "pivot_level": pivot_value,
                "pattern_span": 12 - pivot_idx,
                "left_leg_pct": round(rally_pct, 4),
                "right_leg_pct": round(abs(fallback_pct), 4),
                "recovery_ratio": round(0.0 if rally_pct <= 1e-8 else abs(fallback_pct) / rally_pct, 4),
                "leg_symmetry_score": round(_compute_shape_score(rally_pct, abs(fallback_pct)), 4),
            }

    return None


def _detect_level_reaction(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Dict[str, object] | None:
    """Detect short-horizon support bounce / resistance rejection candidates."""

    if len(close) < 8:
        return None

    recent_close = close.tail(8).reset_index(drop=True)
    recent_high = high.tail(8).reset_index(drop=True)
    recent_low = low.tail(8).reset_index(drop=True)
    support_level = float(recent_low.min())
    resistance_level = float(recent_high.max())
    latest_close = float(recent_close.iloc[-1])
    prev_close = float(recent_close.iloc[-2])

    support_dist_pct = 0.0 if latest_close == 0 else abs(latest_close - support_level) / latest_close * 100.0
    resistance_dist_pct = 0.0 if latest_close == 0 else abs(resistance_level - latest_close) / latest_close * 100.0
    last_two_return_pct = 0.0 if float(recent_close.iloc[-3]) == 0 else (latest_close / float(recent_close.iloc[-3]) - 1.0) * 100.0

    if support_dist_pct <= 0.35 and latest_close > prev_close and last_two_return_pct >= 0.25:
        return {
            "pattern": "support_bounce",
            "pattern_bias": "bullish",
            "pattern_family": "level_reaction",
            "pattern_confidence": round(min(0.78, 0.54 + last_two_return_pct * 0.18), 4),
            "pattern_completed": True,
            "breakout_confirmed": last_two_return_pct >= 0.6,
            "support_level": support_level,
            "pattern_span": 3,
            "distance_to_level_pct": round(support_dist_pct, 4),
            "reaction_return_pct": round(last_two_return_pct, 4),
        }
    if resistance_dist_pct <= 0.35 and latest_close < prev_close and last_two_return_pct <= -0.25:
        return {
            "pattern": "resistance_rejection",
            "pattern_bias": "bearish",
            "pattern_family": "level_reaction",
            "pattern_confidence": round(min(0.78, 0.54 + abs(last_two_return_pct) * 0.18), 4),
            "pattern_completed": True,
            "breakout_confirmed": last_two_return_pct <= -0.6,
            "resistance_level": resistance_level,
            "pattern_span": 3,
            "distance_to_level_pct": round(resistance_dist_pct, 4),
            "reaction_return_pct": round(abs(last_two_return_pct), 4),
        }

    return None


def _detect_flag_continuation(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Dict[str, object] | None:
    """Detect a lightweight bullish/bearish flag style continuation structure."""

    if len(close) < 18:
        return None

    recent_close = close.tail(18).reset_index(drop=True)
    recent_high = high.tail(18).reset_index(drop=True)
    recent_low = low.tail(18).reset_index(drop=True)

    impulse = recent_close.iloc[:8].reset_index(drop=True)
    flag = recent_close.iloc[8:].reset_index(drop=True)

    impulse_start = float(impulse.iloc[0])
    impulse_end = float(impulse.iloc[-1])
    latest_close = float(recent_close.iloc[-1])
    flag_high = float(recent_high.iloc[8:].max())
    flag_low = float(recent_low.iloc[8:].min())
    impulse_return_pct = 0.0 if impulse_start == 0 else (impulse_end / impulse_start - 1.0) * 100.0
    retrace_pct = 0.0 if impulse_end == 0 else abs(latest_close - impulse_end) / impulse_end * 100.0
    flag_reg = _compute_regression_stats(flag)
    flag_width_pct = 0.0 if latest_close == 0 else (flag_high - flag_low) / latest_close * 100.0

    if (
        impulse_return_pct >= 1.0
        and flag_reg["slope"] >= -0.28 * abs(_compute_regression_stats(impulse)["slope"])
        and retrace_pct <= 1.8
        and latest_close >= flag.mean()
        and flag_width_pct <= 3.2
    ):
        confidence = 0.58
        if latest_close >= flag_high * 0.997:
            confidence += 0.1
        if flag_reg["slope"] >= 0:
            confidence += 0.05
        return {
            "pattern": "bullish_flag",
            "pattern_bias": "bullish",
            "pattern_family": "continuation",
            "pattern_confidence": round(min(0.84, confidence), 4),
            "pattern_completed": True,
            "breakout_confirmed": latest_close >= flag_high * 0.999,
            "resistance_level": flag_high,
            "support_level": flag_low,
            "pattern_span": 10,
            "impulse_return_pct": round(impulse_return_pct, 4),
            "retrace_pct": round(retrace_pct, 4),
            "flag_width_pct": round(flag_width_pct, 4),
            "flag_slope": round(float(flag_reg["slope"]), 6),
        }

    if (
        impulse_return_pct <= -1.0
        and flag_reg["slope"] <= 0.28 * abs(_compute_regression_stats(impulse)["slope"])
        and retrace_pct <= 1.8
        and latest_close <= flag.mean()
        and flag_width_pct <= 3.2
    ):
        confidence = 0.58
        if latest_close <= flag_low * 1.003:
            confidence += 0.1
        if flag_reg["slope"] <= 0:
            confidence += 0.05
        return {
            "pattern": "bearish_flag",
            "pattern_bias": "bearish",
            "pattern_family": "continuation",
            "pattern_confidence": round(min(0.84, confidence), 4),
            "pattern_completed": True,
            "breakout_confirmed": latest_close <= flag_low * 1.001,
            "resistance_level": flag_high,
            "support_level": flag_low,
            "pattern_span": 10,
            "impulse_return_pct": round(abs(impulse_return_pct), 4),
            "retrace_pct": round(retrace_pct, 4),
            "flag_width_pct": round(flag_width_pct, 4),
            "flag_slope": round(float(flag_reg["slope"]), 6),
        }

    return None


def _detect_hidden_base(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Dict[str, object] | None:
    """Detect a tight consolidation followed by a directional release."""

    if len(close) < 14:
        return None

    recent_close = close.tail(14).reset_index(drop=True)
    recent_high = high.tail(14).reset_index(drop=True)
    recent_low = low.tail(14).reset_index(drop=True)
    base_close = recent_close.iloc[:10].reset_index(drop=True)
    release_close = recent_close.iloc[10:].reset_index(drop=True)

    base_high = float(recent_high.iloc[:10].max())
    base_low = float(recent_low.iloc[:10].min())
    base_mid = float(base_close.mean())
    latest_close = float(recent_close.iloc[-1])
    prev_close = float(recent_close.iloc[-2])
    base_width_pct = 0.0 if base_mid == 0 else float((base_high - base_low) / base_mid * 100.0)
    breakout_margin_pct = round(_safe_pct_change(base_high, latest_close), 4)
    breakdown_margin_pct = round(_safe_pct_change(latest_close, base_low), 4)
    release_return_pct = round(_safe_pct_change(float(base_close.iloc[-1]), latest_close), 4)
    release_reg = _compute_regression_stats(release_close)

    # Bullish hidden base: tight sideways base, then impulsive close above range.
    if (
        base_width_pct <= 1.9
        and latest_close > base_high * 1.0015
        and prev_close >= base_mid
        and release_reg["slope"] > 0
    ):
        confidence = 0.6
        if breakout_margin_pct >= 0.25:
            confidence += 0.08
        if release_return_pct >= 0.55:
            confidence += 0.06
        return {
            "pattern": "hidden_base_breakout",
            "pattern_bias": "bullish",
            "pattern_family": "continuation",
            "pattern_confidence": round(min(0.86, confidence), 4),
            "pattern_completed": True,
            "breakout_confirmed": True,
            "support_level": base_low,
            "resistance_level": base_high,
            "pattern_span": 14,
            "base_width_pct": round(base_width_pct, 4),
            "breakout_margin_pct": breakout_margin_pct,
            "release_return_pct": release_return_pct,
            "base_tightness_score": round(max(0.0, 1.0 - base_width_pct / 2.5), 4),
        }

    # Bearish symmetric case for downside release.
    if (
        base_width_pct <= 1.9
        and latest_close < base_low * 0.9985
        and prev_close <= base_mid
        and release_reg["slope"] < 0
    ):
        confidence = 0.6
        if breakdown_margin_pct >= 0.25:
            confidence += 0.08
        if abs(release_return_pct) >= 0.55:
            confidence += 0.06
        return {
            "pattern": "hidden_distribution_breakdown",
            "pattern_bias": "bearish",
            "pattern_family": "continuation",
            "pattern_confidence": round(min(0.86, confidence), 4),
            "pattern_completed": True,
            "breakout_confirmed": True,
            "support_level": base_low,
            "resistance_level": base_high,
            "pattern_span": 14,
            "base_width_pct": round(base_width_pct, 4),
            "breakout_margin_pct": breakdown_margin_pct,
            "release_return_pct": round(abs(release_return_pct), 4),
            "base_tightness_score": round(max(0.0, 1.0 - base_width_pct / 2.5), 4),
        }

    return None


def _compute_breakout_authenticity(
    ohlc_df: pd.DataFrame,
    pattern_features: Dict[str, object],
) -> Dict[str, object]:
    """Score whether the latest breakout looks real enough to deserve authority."""

    if ohlc_df.empty:
        return {
            "breakout_authenticity_score": 0.0,
            "breakout_body_ratio": 0.0,
            "breakout_extension_pct": 0.0,
            "breakout_retest_quality": "unknown",
        }

    latest = ohlc_df.iloc[-1]
    prev = ohlc_df.iloc[-2] if len(ohlc_df) >= 2 else latest
    latest_geom = _compute_candle_geometry(
        open_price=float(latest["Open"]),
        high_price=float(latest["High"]),
        low_price=float(latest["Low"]),
        close_price=float(latest["Close"]),
    )
    current_close = float(latest["Close"])
    support_level = pattern_features.get("support_level")
    resistance_level = pattern_features.get("resistance_level")
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))

    extension_pct = 0.0
    retest_quality = "none"
    if breakout_confirmed and pattern_bias == "bullish" and resistance_level is not None:
        extension_pct = round(_safe_pct_change(float(resistance_level), current_close), 4)
        retest_quality = (
            "healthy"
            if float(prev["Low"]) >= float(resistance_level) * 0.998
            else "unproven"
        )
    elif breakout_confirmed and pattern_bias == "bearish" and support_level is not None:
        extension_pct = round(_safe_pct_change(current_close, float(support_level)), 4)
        retest_quality = (
            "healthy"
            if float(prev["High"]) <= float(support_level) * 1.002
            else "unproven"
        )

    authenticity = 0.0
    authenticity += min(0.4, float(latest_geom["body_ratio"]) * 0.5)
    authenticity += min(0.28, max(0.0, extension_pct) * 0.45)
    if retest_quality == "healthy":
        authenticity += 0.18
    elif breakout_confirmed:
        authenticity += 0.08

    return {
        "breakout_authenticity_score": round(min(1.0, authenticity), 4),
        "breakout_body_ratio": latest_geom["body_ratio"],
        "breakout_extension_pct": round(max(0.0, extension_pct), 4),
        "breakout_retest_quality": retest_quality,
    }


def _score_indicator_bias(indicator_features: Dict[str, object]) -> Dict[str, object]:
    """Convert raw indicator states into directional evidence scores."""

    long_score = 0.0
    short_score = 0.0
    reasons: List[str] = []

    if indicator_features.get("macd_cross") == "bullish_cross":
        long_score += 1.0
        reasons.append("MACD bullish cross")
    elif indicator_features.get("macd_cross") == "bearish_cross":
        short_score += 1.0
        reasons.append("MACD bearish cross")

    if indicator_features.get("ma_cross") == "bullish_cross":
        long_score += 0.8
        reasons.append("Fast MA crossed above slow MA")
    elif indicator_features.get("ma_cross") == "bearish_cross":
        short_score += 0.8
        reasons.append("Fast MA crossed below slow MA")

    rsi_state = indicator_features.get("rsi_state")
    if rsi_state == "bullish":
        long_score += 0.5
        reasons.append("RSI in bullish zone")
    elif rsi_state == "bearish":
        short_score += 0.5
        reasons.append("RSI in bearish zone")
    elif rsi_state == "oversold":
        long_score += 0.6
        reasons.append("RSI indicates oversold rebound potential")
    elif rsi_state == "overbought":
        short_score += 0.6
        reasons.append("RSI indicates overbought pullback risk")

    stoch_state = indicator_features.get("stoch_state")
    if stoch_state == "bullish":
        long_score += 0.3
    elif stoch_state == "bearish":
        short_score += 0.3
    elif stoch_state == "oversold":
        long_score += 0.4
    elif stoch_state == "overbought":
        short_score += 0.4

    roc = indicator_features.get("roc")
    if isinstance(roc, (float, int)):
        if roc > 0:
            long_score += min(0.6, abs(float(roc)) / 5)
        elif roc < 0:
            short_score += min(0.6, abs(float(roc)) / 5)

    divergence = indicator_features.get("rsi_divergence")
    if divergence == "bullish_divergence":
        long_score += 0.7
        reasons.append("Bullish RSI divergence")
    elif divergence == "bearish_divergence":
        short_score += 0.7
        reasons.append("Bearish RSI divergence")

    return {
        "indicator_long_score": round(long_score, 4),
        "indicator_short_score": round(short_score, 4),
        "indicator_bias": "bullish" if long_score > short_score else "bearish" if short_score > long_score else "neutral",
        "indicator_reasons": reasons[:4],
    }


def _score_pattern_bias(pattern_features: Dict[str, object]) -> Dict[str, object]:
    """Translate pattern detection results into directional decision hints."""

    confidence = float(pattern_features.get("pattern_confidence", 0.0) or 0.0)
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    bias = pattern_features.get("pattern_bias", "neutral")
    family = str(pattern_features.get("pattern_family", "classic"))

    long_score = 0.0
    short_score = 0.0
    reasons: List[str] = []

    if bias == "bullish":
        # An unconfirmed pattern should still contribute directional evidence,
        # just not as aggressively as a confirmed breakout.
        weight = 1.0 if breakout_confirmed else 0.55
        if family == "level_reaction":
            weight = 0.78 if breakout_confirmed else 0.62
        elif family == "reversal":
            weight = 0.9 if breakout_confirmed else 0.7
        elif family == "continuation":
            weight = 0.95 if breakout_confirmed else 0.76
        long_score = confidence * weight
        reasons.append(f"Detected bullish pattern: {pattern_features.get('pattern')}")
        if breakout_confirmed:
            reasons.append("Bullish breakout confirmed")
        elif pattern_features.get("pattern_completed"):
            reasons.append("Pattern structure is completed but not yet confirmed")
    elif bias == "bearish":
        weight = 1.0 if breakout_confirmed else 0.55
        if family == "level_reaction":
            weight = 0.78 if breakout_confirmed else 0.62
        elif family == "reversal":
            weight = 0.9 if breakout_confirmed else 0.7
        elif family == "continuation":
            weight = 0.95 if breakout_confirmed else 0.76
        short_score = confidence * weight
        reasons.append(f"Detected bearish pattern: {pattern_features.get('pattern')}")
        if breakout_confirmed:
            reasons.append("Bearish breakdown confirmed")
        elif pattern_features.get("pattern_completed"):
            reasons.append("Pattern structure is completed but not yet confirmed")
    elif pattern_features.get("pattern") not in {None, "none"}:
        reasons.append(f"Pattern present but directional edge is weak: {pattern_features.get('pattern')}")

    return {
        "pattern_long_score": round(long_score, 4),
        "pattern_short_score": round(short_score, 4),
        "pattern_family": family,
        "pattern_reasons": reasons[:3],
    }


def _score_trend_bias(trend_features: Dict[str, object]) -> Dict[str, object]:
    """Turn trend structure into directional evidence."""

    direction = trend_features.get("trend_direction")
    strength = float(trend_features.get("trend_strength_score", 0.0) or 0.0)
    regime = trend_features.get("market_regime")
    structure = trend_features.get("high_low_structure")
    breakout_state = trend_features.get("breakout_state")
    location_state = trend_features.get("location_state")
    support_touches = int(trend_features.get("support_touch_count", 0) or 0)
    resistance_touches = int(trend_features.get("resistance_touch_count", 0) or 0)

    long_score = 0.0
    short_score = 0.0
    reasons: List[str] = []

    if direction == "uptrend":
        # Trend direction should matter, but it should not overwhelm every
        # other component by default. We scale the base contribution by the
        # measured trend strength instead of forcing a large floor.
        long_score += 0.35 + 0.65 * strength
        reasons.append("Trend direction is upward")
    elif direction == "downtrend":
        short_score += 0.35 + 0.65 * strength
        reasons.append("Trend direction is downward")

    if structure == "higher_high_higher_low":
        long_score += 0.45
        reasons.append("Higher highs and higher lows")
    elif structure == "lower_high_lower_low":
        short_score += 0.45
        reasons.append("Lower highs and lower lows")
    elif structure == "compression":
        reasons.append("Price structure is compressed")

    if breakout_state == "bullish_breakout":
        long_score += 0.8
        reasons.append("Price is breaking above recent resistance")
    elif breakout_state == "bearish_breakdown":
        short_score += 0.8
        reasons.append("Price is breaking below recent support")
    elif breakout_state == "testing_support":
        reasons.append("Price is testing support")
    elif breakout_state == "testing_resistance":
        reasons.append("Price is testing resistance")

    if location_state == "near_support" and long_score >= short_score:
        long_score += 0.2
        reasons.append("Price is positioned near support")
    elif location_state == "near_resistance" and short_score >= long_score:
        short_score += 0.2
        reasons.append("Price is positioned near resistance")

    # Position conflict should mostly affect confidence sizing, not flip a
    # directional read by itself.
    if location_state == "near_resistance" and long_score > short_score:
        reasons.append("Long setup is approaching resistance")
    elif location_state == "near_support" and short_score > long_score:
        reasons.append("Short setup is approaching support")

    if support_touches >= 2 and long_score >= short_score:
        long_score += 0.15
        reasons.append("Support has been tested multiple times")
    if resistance_touches >= 2 and short_score >= long_score:
        short_score += 0.15
        reasons.append("Resistance has been tested multiple times")

    # A trend that is already pressing into a nearby barrier should not keep
    # the same weight as a clean trend with room to continue.
    if breakout_state == "testing_resistance" and long_score > short_score:
        long_score *= 0.78
        reasons.append("Uptrend is being compressed under resistance")
    elif breakout_state == "testing_support" and short_score > long_score:
        short_score *= 0.78
        reasons.append("Downtrend is leaning into support")

    if location_state == "near_resistance" and long_score > short_score and breakout_state != "bullish_breakout":
        long_score *= 0.88
    elif location_state == "near_support" and short_score > long_score and breakout_state != "bearish_breakdown":
        short_score *= 0.88

    if regime == "trend":
        long_score *= 1.1 if long_score > short_score else 1.0
        short_score *= 1.1 if short_score > long_score else 1.0
    elif regime == "compression":
        long_score *= 0.85
        short_score *= 0.85

    if strength < 0.4:
        long_score *= 0.92
        short_score *= 0.92

    return {
        "trend_long_score": round(long_score, 4),
        "trend_short_score": round(short_score, 4),
        "trend_reasons": reasons[:4],
    }


def _compute_short_horizon_reaction_signal(
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
) -> Dict[str, object]:
    """
    Detect next-1-bar style rebound / pullback opportunities.

    Financial note:
    Ultra-short-horizon forecasts often reward local support bounces and
    resistance rejections before a broader trend has fully changed.
    """

    location_state = str(trend_features.get("location_state", "mid_range"))
    breakout_state = str(trend_features.get("breakout_state", "inside_range"))
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    pattern_name = str(pattern_features.get("pattern", "none"))
    trend_direction = str(trend_features.get("trend_direction", "sideways"))
    trend_alignment = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    rsi_state = str(indicator_features.get("rsi_state", "neutral"))
    stoch_state = str(indicator_features.get("stoch_state", "neutral"))
    rsi_divergence = str(indicator_features.get("rsi_divergence", "none"))
    macd_cross = str(indicator_features.get("macd_cross", "none"))

    bullish_score = 0.0
    bearish_score = 0.0
    reasons: List[str] = []

    if location_state == "near_support":
        bullish_score += 0.55
        reasons.append("price is pressing into support, where short-horizon rebounds often start")
    elif location_state == "near_resistance":
        bearish_score += 0.55
        reasons.append("price is pressing into resistance, where short-horizon pullbacks often start")

    if rsi_state == "oversold" or stoch_state == "oversold":
        bullish_score += 0.5
        reasons.append("oscillators are stretched to oversold levels")
    if rsi_state == "overbought" or stoch_state == "overbought":
        bearish_score += 0.5
        reasons.append("oscillators are stretched to overbought levels")

    if rsi_divergence == "bullish_divergence":
        bullish_score += 0.45
        reasons.append("bullish divergence supports a local rebound")
    elif rsi_divergence == "bearish_divergence":
        bearish_score += 0.45
        reasons.append("bearish divergence supports a local pullback")

    if macd_cross == "bullish_cross":
        bullish_score += 0.35
    elif macd_cross == "bearish_cross":
        bearish_score += 0.35

    if pattern_bias == "bullish" and pattern_name in {"support_bounce", "double_bottom", "v_shaped_reversal"}:
        bullish_score += 0.45
        reasons.append("local price structure resembles a bullish short-horizon reversal")
    elif pattern_bias == "bearish" and pattern_name in {"resistance_rejection", "double_top", "inverted_v_reversal"}:
        bearish_score += 0.45
        reasons.append("local price structure resembles a bearish short-horizon rejection")

    if trend_alignment == "bullish_short_vs_bearish_long":
        bullish_score += 0.35
    elif trend_alignment == "bearish_short_vs_bullish_long":
        bearish_score += 0.35

    if breakout_state == "testing_support":
        bullish_score += 0.25
    elif breakout_state == "testing_resistance":
        bearish_score += 0.25

    if trend_direction == "downtrend" and bullish_score > bearish_score:
        bullish_score += 0.08
    elif trend_direction == "uptrend" and bearish_score > bullish_score:
        bearish_score += 0.08

    signal_bias = "none"
    signal_score = 0.0
    if bullish_score >= bearish_score and bullish_score >= 1.25:
        signal_bias = "bullish_rebound_candidate"
        signal_score = bullish_score
    elif bearish_score > bullish_score and bearish_score >= 1.25:
        signal_bias = "bearish_pullback_candidate"
        signal_score = bearish_score

    return {
        "short_horizon_bias": signal_bias,
        "short_horizon_score": round(signal_score, 4),
        "short_horizon_reasons": reasons[:4],
    }


def _compute_continuation_signal(
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
    risk_features: Dict[str, object],
) -> Dict[str, object]:
    """
    Detect whether price still looks like a healthy continuation rather than
    an exhausted move near the end of trend.

    Why this matters:
    QuantAgent often wins on samples where classical oscillators look stretched
    or where price is near resistance, but the full structure still resembles
    a continuation breakout rather than a true reversal. This helper makes that
    distinction explicit for KuantAgent.
    """

    trend_direction = str(trend_features.get("trend_direction", "sideways"))
    market_regime = str(trend_features.get("market_regime", "range"))
    breakout_state = str(trend_features.get("breakout_state", "inside_range"))
    location_state = str(trend_features.get("location_state", "mid_range"))
    trend_alignment = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    high_low_structure = str(trend_features.get("high_low_structure", "mixed"))
    trend_continuation_quality = str(trend_features.get("trend_continuation_quality", "low"))
    trend_exhaustion_risk = str(trend_features.get("trend_exhaustion_risk", "low"))
    trend_strength = float(trend_features.get("trend_strength_score", 0.0) or 0.0)

    pattern_name = str(pattern_features.get("pattern", "none"))
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    pattern_family = str(pattern_features.get("pattern_family", "none"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    breakout_authenticity_score = float(pattern_features.get("breakout_authenticity_score", 0.0) or 0.0)

    macd_cross = str(indicator_features.get("macd_cross", "none"))
    macd_hist = float(indicator_features.get("macd_hist", 0.0) or 0.0)
    roc = float(indicator_features.get("roc", 0.0) or 0.0)
    rsi_state = str(indicator_features.get("rsi_state", "neutral"))
    stoch_state = str(indicator_features.get("stoch_state", "neutral"))
    price_vs_ma_fast_pct = float(indicator_features.get("price_vs_ma_fast_pct", 0.0) or 0.0)
    price_vs_ma_slow_pct = float(indicator_features.get("price_vs_ma_slow_pct", 0.0) or 0.0)
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))

    bullish_score = 0.0
    bearish_score = 0.0
    bullish_reasons: List[str] = []
    bearish_reasons: List[str] = []

    if trend_direction == "uptrend":
        bullish_score += 0.55
        bullish_reasons.append("broader structure is still an uptrend")
    elif trend_direction == "downtrend":
        bearish_score += 0.55
        bearish_reasons.append("broader structure is still a downtrend")

    if high_low_structure == "higher_high_higher_low":
        bullish_score += 0.45
        bullish_reasons.append("higher-high higher-low structure supports continuation")
    elif high_low_structure == "lower_high_lower_low":
        bearish_score += 0.45
        bearish_reasons.append("lower-high lower-low structure supports continuation")

    if trend_alignment == "aligned_bullish":
        bullish_score += 0.35
    elif trend_alignment == "aligned_bearish":
        bearish_score += 0.35

    if breakout_state == "bullish_breakout":
        bullish_score += 0.75
        bullish_reasons.append("price is extending a bullish breakout")
    elif breakout_state == "bearish_breakdown":
        bearish_score += 0.75
        bearish_reasons.append("price is extending a bearish breakdown")
    elif breakout_state == "testing_resistance" and trend_direction == "uptrend":
        bullish_score += 0.22
        bullish_reasons.append("uptrend is pressing resistance with continuation pressure")
    elif breakout_state == "testing_support" and trend_direction == "downtrend":
        bearish_score += 0.22
        bearish_reasons.append("downtrend is pressing support with continuation pressure")

    if breakout_confirmed and pattern_bias == "bullish":
        bullish_score += 0.45
        bullish_reasons.append("bullish pattern has breakout confirmation")
    elif breakout_confirmed and pattern_bias == "bearish":
        bearish_score += 0.45
        bearish_reasons.append("bearish pattern has breakdown confirmation")

    if pattern_name in {"v_shaped_reversal", "support_bounce"} and pattern_bias == "bullish":
        bullish_score += 0.25
        bullish_reasons.append("local structure can act as launchpad for upside continuation")
    elif pattern_name in {"inverted_v_reversal", "resistance_rejection"} and pattern_bias == "bearish":
        bearish_score += 0.25
        bearish_reasons.append("local structure can act as launchpad for downside continuation")
    elif pattern_family == "continuation" and pattern_bias == "bullish":
        bullish_score += 0.34 if breakout_confirmed else 0.24
        bullish_reasons.append("continuation pattern supports upside extension")
    elif pattern_family == "continuation" and pattern_bias == "bearish":
        bearish_score += 0.34 if breakout_confirmed else 0.24
        bearish_reasons.append("continuation pattern supports downside extension")
    elif pattern_family == "classic" and pattern_bias == "bullish":
        bullish_score += 0.16
    elif pattern_family == "classic" and pattern_bias == "bearish":
        bearish_score += 0.16

    if macd_cross == "bullish_cross" or macd_hist > 0:
        bullish_score += 0.28
    elif macd_cross == "bearish_cross" or macd_hist < 0:
        bearish_score += 0.28

    if roc > 0:
        bullish_score += min(0.28, abs(roc) / 8.0)
    elif roc < 0:
        bearish_score += min(0.28, abs(roc) / 8.0)

    if price_vs_ma_fast_pct > 0 and price_vs_ma_slow_pct > 0:
        bullish_score += 0.22
    elif price_vs_ma_fast_pct < 0 and price_vs_ma_slow_pct < 0:
        bearish_score += 0.22

    if trend_continuation_quality == "high":
        bullish_score += 0.16 if trend_direction == "uptrend" else 0.0
        bearish_score += 0.16 if trend_direction == "downtrend" else 0.0
    elif trend_continuation_quality == "low":
        bullish_score -= 0.18 if trend_direction == "uptrend" else 0.0
        bearish_score -= 0.18 if trend_direction == "downtrend" else 0.0

    if trend_exhaustion_risk == "high":
        bullish_score -= 0.2 if trend_direction == "uptrend" else 0.0
        bearish_score -= 0.2 if trend_direction == "downtrend" else 0.0
    elif trend_exhaustion_risk == "medium":
        bullish_score -= 0.08 if trend_direction == "uptrend" else 0.0
        bearish_score -= 0.08 if trend_direction == "downtrend" else 0.0

    # Overbought/oversold is not enough to kill continuation by itself.
    if rsi_state == "overbought" or stoch_state == "overbought":
        bullish_score -= 0.05
        bearish_score += 0.08
    elif rsi_state == "oversold" or stoch_state == "oversold":
        bearish_score -= 0.05
        bullish_score += 0.08

    if false_breakout_risk == "high":
        bullish_score -= 0.15
        bearish_score -= 0.15

    if market_regime in {"compression", "range"} and not breakout_confirmed:
        bullish_score -= 0.12
        bearish_score -= 0.12

    if (
        trend_direction == "uptrend"
        and breakout_authenticity_score < 0.46
        and breakout_state != "bullish_breakout"
        and (trend_exhaustion_risk == "high" or location_state == "near_resistance")
    ):
        bullish_score -= 0.18
        bearish_score += 0.08
    if (
        trend_direction == "downtrend"
        and breakout_authenticity_score < 0.46
        and breakout_state != "bearish_breakdown"
        and (trend_exhaustion_risk == "high" or location_state == "near_support")
    ):
        bearish_score -= 0.18
        bullish_score += 0.08

    if (
        trend_alignment == "bullish_short_vs_bearish_long"
        and breakout_state != "bearish_breakdown"
        and (macd_cross == "bullish_cross" or rsi_state in {"bullish", "oversold"})
    ):
        bearish_score -= 0.16
        bullish_score += 0.06
    elif (
        trend_alignment == "bearish_short_vs_bullish_long"
        and breakout_state != "bullish_breakout"
        and (macd_cross == "bearish_cross" or rsi_state in {"bearish", "overbought"})
    ):
        bullish_score -= 0.16
        bearish_score += 0.06

    if trend_strength >= 0.7:
        if trend_direction == "uptrend":
            bullish_score += 0.12
        elif trend_direction == "downtrend":
            bearish_score += 0.12

    continuation_bias = "none"
    continuation_score = 0.0
    continuation_reasons: List[str] = []
    exhaustion_risk = "low"

    bullish_score = round(max(0.0, bullish_score), 4)
    bearish_score = round(max(0.0, bearish_score), 4)

    if bullish_score >= bearish_score and bullish_score >= 1.65:
        continuation_bias = "bullish_continuation_candidate"
        continuation_score = bullish_score
        continuation_reasons = bullish_reasons
        if (
            trend_direction == "uptrend"
            and breakout_state != "bullish_breakout"
            and location_state == "near_resistance"
            and trend_exhaustion_risk == "high"
        ):
            exhaustion_risk = "medium"
    elif bearish_score > bullish_score and bearish_score >= 1.65:
        continuation_bias = "bearish_continuation_candidate"
        continuation_score = bearish_score
        continuation_reasons = bearish_reasons
        if (
            trend_direction == "downtrend"
            and breakout_state != "bearish_breakdown"
            and location_state == "near_support"
            and trend_exhaustion_risk == "high"
        ):
            exhaustion_risk = "medium"

    return {
        "continuation_bias": continuation_bias,
        "continuation_score": round(continuation_score, 4),
        "continuation_reasons": continuation_reasons[:4],
        "continuation_exhaustion_risk": exhaustion_risk,
        "bullish_continuation_score": bullish_score,
        "bearish_continuation_score": bearish_score,
    }


def _compute_three_bar_path_signal(
    indicator_features: Dict[str, object],
    trend_features: Dict[str, object],
    short_horizon_signal: Dict[str, object],
    continuation_signal: Dict[str, object],
    reversal_signal: Dict[str, object],
) -> Dict[str, object]:
    """
    Estimate the likely path across the next three bars.

    QuantAgent's original protocol scores the next three candles one by one,
    so a useful predictor should separate immediate reaction from later
    follow-through instead of collapsing everything into one direction.
    """

    trend_direction = str(trend_features.get("trend_direction", "sideways"))
    trend_alignment = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    trend_quality = str(trend_features.get("trend_continuation_quality", "low"))
    trend_strength = float(trend_features.get("trend_strength_score", 0.0) or 0.0)
    breakout_state = str(trend_features.get("breakout_state", "inside_range"))
    location_state = str(trend_features.get("location_state", "mid_range"))
    market_regime = str(trend_features.get("market_regime", "range"))

    rsi_state = str(indicator_features.get("rsi_state", "neutral"))
    stoch_state = str(indicator_features.get("stoch_state", "neutral"))
    macd_hist = float(indicator_features.get("macd_hist", 0.0) or 0.0)
    roc = float(indicator_features.get("roc", 0.0) or 0.0)

    short_horizon_bias = str(short_horizon_signal.get("short_horizon_bias", "none"))
    short_horizon_score = float(short_horizon_signal.get("short_horizon_score", 0.0) or 0.0)
    continuation_bias = str(continuation_signal.get("continuation_bias", "none"))
    continuation_score = float(continuation_signal.get("continuation_score", 0.0) or 0.0)
    reversal_bias = str(reversal_signal.get("reversal_bias", "none"))
    reversal_score = float(reversal_signal.get("reversal_score", 0.0) or 0.0)
    reversal_confirmed = bool(reversal_signal.get("reversal_confirmed", False))

    def trend_side() -> str:
        if trend_strength < 0.45 and breakout_state not in {"bullish_breakout", "bearish_breakdown"}:
            if trend_alignment == "bullish_short_vs_bearish_long":
                return "LONG"
            if trend_alignment == "bearish_short_vs_bullish_long":
                return "SHORT"
            return "NEUTRAL"
        if trend_alignment in {"aligned_bullish", "bullish_short_vs_bearish_long"}:
            return "LONG"
        if trend_alignment in {"aligned_bearish", "bearish_short_vs_bullish_long"}:
            return "SHORT"
        if trend_direction == "uptrend":
            return "LONG"
        if trend_direction == "downtrend":
            return "SHORT"
        return "NEUTRAL"

    def momentum_side() -> str:
        bullish = 0.0
        bearish = 0.0
        if macd_hist > 0:
            bullish += 0.35
        elif macd_hist < 0:
            bearish += 0.35
        if roc > 0:
            bullish += min(0.45, abs(roc) / 6.0)
        elif roc < 0:
            bearish += min(0.45, abs(roc) / 6.0)
        if rsi_state in {"bullish", "oversold"}:
            bullish += 0.18
        elif rsi_state in {"bearish", "overbought"}:
            bearish += 0.18
        if stoch_state in {"bullish", "oversold"}:
            bullish += 0.14
        elif stoch_state in {"bearish", "overbought"}:
            bearish += 0.14
        if bullish >= bearish + 0.22:
            return "LONG"
        if bearish >= bullish + 0.22:
            return "SHORT"
        return "NEUTRAL"

    bar1 = "NEUTRAL"
    bar2 = "NEUTRAL"
    bar3 = "NEUTRAL"
    reasons: List[str] = []

    if short_horizon_bias == "bullish_rebound_candidate" and short_horizon_score >= 1.35:
        bar1 = "LONG"
        reasons.append("bar1 favors support-bounce reaction")
    elif short_horizon_bias == "bearish_pullback_candidate" and short_horizon_score >= 1.35:
        bar1 = "SHORT"
        reasons.append("bar1 favors resistance-rejection reaction")
    else:
        bar1 = momentum_side()
        if bar1 != "NEUTRAL":
            reasons.append("bar1 follows immediate momentum")

    if reversal_confirmed and reversal_bias == "bullish_reversal":
        bar2 = "LONG"
        bar3 = "LONG"
        reasons.append("confirmed bullish transition can persist into bars 2-3")
    elif reversal_confirmed and reversal_bias == "bearish_reversal":
        bar2 = "SHORT"
        bar3 = "SHORT"
        reasons.append("confirmed bearish transition can persist into bars 2-3")
    elif continuation_bias == "bullish_continuation_candidate" and continuation_score >= 1.8:
        bar2 = "LONG"
        bar3 = "LONG" if trend_quality in {"medium", "high"} else trend_side()
        reasons.append("bars 2-3 favor bullish continuation pressure")
    elif continuation_bias == "bearish_continuation_candidate" and continuation_score >= 1.8:
        bar2 = "SHORT"
        bar3 = "SHORT" if trend_quality in {"medium", "high"} else trend_side()
        reasons.append("bars 2-3 favor bearish continuation pressure")
    else:
        bar2 = momentum_side()
        bar3 = trend_side()
        reasons.append("later bars fall back to momentum and trend alignment")

    if trend_strength < 0.45 and breakout_state not in {"bullish_breakout", "bearish_breakdown"}:
        if bar2 == trend_side() and bar2 != "NEUTRAL":
            bar2 = momentum_side()
            reasons.append("weak trend prevents the second bar from blindly inheriting stale trend direction")
        if bar3 != "NEUTRAL" and market_regime in {"range", "compression", "trend"}:
            bar3 = "NEUTRAL" if bar2 == "NEUTRAL" else bar2
            reasons.append("weak trend lowers confidence in third-bar follow-through without a real breakout")

    if bar2 == "NEUTRAL":
        bar2 = trend_side()
    if bar3 == "NEUTRAL":
        bar3 = trend_side()

    if market_regime in {"range", "compression"} and breakout_state == "inside_range":
        if location_state == "near_resistance" and bar3 == "LONG":
            bar3 = "NEUTRAL"
            reasons.append("range resistance weakens third-bar bullish follow-through")
        elif location_state == "near_support" and bar3 == "SHORT":
            bar3 = "NEUTRAL"
            reasons.append("range support weakens third-bar bearish follow-through")

    votes = [bar1, bar2, bar3]
    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")
    neutral_votes = votes.count("NEUTRAL")
    if long_votes > short_votes:
        majority_bias = "LONG"
    elif short_votes > long_votes:
        majority_bias = "SHORT"
    else:
        majority_bias = "MIXED"

    consistency = max(long_votes, short_votes, neutral_votes) / 3.0
    path_score = 0.0
    if majority_bias != "MIXED":
        path_score += consistency
        if continuation_score >= 1.8:
            path_score += min(0.45, continuation_score / 6.0)
        if reversal_confirmed:
            path_score += min(0.45, reversal_score / 6.0)
        if trend_quality == "high":
            path_score += 0.16
        elif trend_quality == "low":
            path_score -= 0.08

    return {
        "three_bar_path": votes,
        "three_bar_majority_bias": majority_bias,
        "three_bar_path_consistency": round(consistency, 4),
        "three_bar_path_score": round(max(0.0, path_score), 4),
        "three_bar_path_reasons": reasons[:5],
    }


def _score_risk_context(risk_features: Dict[str, object]) -> Dict[str, object]:
    """Convert volatility and environment descriptors into risk penalties."""

    volatility_regime = risk_features.get("volatility_regime", "unknown")
    trend_environment = risk_features.get("trend_environment", "unknown")
    breakout_quality = risk_features.get("breakout_quality", "unknown")
    false_breakout_risk = risk_features.get("false_breakout_risk", "unknown")

    penalty = 0.0
    reasons: List[str] = []

    if volatility_regime == "high":
        penalty += 0.3
        reasons.append("Volatility regime is high")
    elif volatility_regime == "medium":
        penalty += 0.15

    if trend_environment == "weak":
        penalty += 0.2
        reasons.append("Trend strength is weak")
    if breakout_quality == "fragile":
        penalty += 0.16
        reasons.append("Breakout quality is fragile")
    if false_breakout_risk == "high":
        penalty += 0.2
        reasons.append("False breakout risk is high")
    elif false_breakout_risk == "medium":
        penalty += 0.1

    risk_level = "low"
    if penalty >= 0.4:
        risk_level = "high"
    elif penalty >= 0.2:
        risk_level = "medium"

    return {
        "risk_penalty": round(penalty, 4),
        "risk_level": risk_level,
        "risk_reasons": reasons[:3],
    }


def _collect_direction_votes(
    indicator_bias: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
) -> List[str]:
    """Collect directional votes from the three main evidence families."""

    votes: List[str] = []

    indicator_state = indicator_bias.get("indicator_bias", "neutral")
    if indicator_state == "bullish":
        votes.append("LONG")
    elif indicator_state == "bearish":
        votes.append("SHORT")

    pattern_bias = pattern_features.get("pattern_bias", "neutral")
    if pattern_bias == "bullish":
        votes.append("LONG")
    elif pattern_bias == "bearish":
        votes.append("SHORT")

    trend_direction = trend_features.get("trend_direction")
    if trend_direction == "uptrend":
        votes.append("LONG")
    elif trend_direction == "downtrend":
        votes.append("SHORT")

    return votes


def _derive_structure_semantics(
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
) -> Dict[str, object]:
    pattern_name = str(pattern_features.get("pattern", "none"))
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    pattern_geometry_score = float(pattern_features.get("pattern_geometry_score", 0.0) or 0.0)
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    breakout_authenticity_score = float(pattern_features.get("breakout_authenticity_score", 0.0) or 0.0)
    breakout_body_ratio = float(pattern_features.get("breakout_body_ratio", 0.0) or 0.0)
    breakout_retest_quality = str(pattern_features.get("breakout_retest_quality", "unknown"))
    market_regime = str(trend_features.get("market_regime", "range"))
    trend_direction = str(trend_features.get("trend_direction", "sideways"))
    trend_alignment_state = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    breakout_state = str(trend_features.get("breakout_state", "inside_range"))
    channel_direction = str(trend_features.get("channel_direction", "mixed"))
    swing_bias = str(trend_features.get("swing_bias", "mixed"))
    swing_quality_score = float(trend_features.get("swing_quality_score", 0.0) or 0.0)
    swing_compression_score = float(trend_features.get("swing_compression_score", 0.0) or 0.0)
    structure_break_state = str(trend_features.get("structure_break_state", "none"))
    structure_break_distance_pct = float(trend_features.get("structure_break_distance_pct", 0.0) or 0.0)
    recent_breakout_followthrough_score = float(trend_features.get("recent_breakout_followthrough_score", 0.0) or 0.0)
    recent_breakout_failure_score = float(trend_features.get("recent_breakout_failure_score", 0.0) or 0.0)
    level_reclaim_state = str(trend_features.get("level_reclaim_state", "none"))
    rejection_wick_bias = str(trend_features.get("rejection_wick_bias", "neutral"))
    location_state = str(trend_features.get("location_state", "mid_range"))
    latest_close_near_high_ratio = float(trend_features.get("latest_close_near_high_ratio", 0.0) or 0.0)
    latest_close_near_low_ratio = float(trend_features.get("latest_close_near_low_ratio", 0.0) or 0.0)
    macd_hist = float(indicator_features.get("macd_hist", 0.0) or 0.0)
    macd_cross = str(indicator_features.get("macd_cross", "none"))
    rsi_state = str(indicator_features.get("rsi_state", "neutral"))
    stoch_state = str(indicator_features.get("stoch_state", "neutral"))
    rsi_divergence = str(indicator_features.get("rsi_divergence", "none"))
    recent_consecutive_up_closes = int(indicator_features.get("recent_consecutive_up_closes", 0) or 0)
    recent_consecutive_down_closes = int(indicator_features.get("recent_consecutive_down_closes", 0) or 0)
    last_candle_body_ratio = float(indicator_features.get("last_candle_body_ratio", 0.0) or 0.0)
    last_candle_lower_wick_ratio = float(indicator_features.get("last_candle_lower_wick_ratio", 0.0) or 0.0)
    last_candle_upper_wick_ratio = float(indicator_features.get("last_candle_upper_wick_ratio", 0.0) or 0.0)
    candidate_confidence_map = dict(pattern_features.get("candidate_confidence_map", {}) or {})
    candidate_breakout_map = dict(pattern_features.get("candidate_breakout_map", {}) or {})
    candidate_detail_map = dict(pattern_features.get("candidate_detail_map", {}) or {})

    def _candidate_score(pattern_key: str) -> float:
        return float(candidate_confidence_map.get(pattern_key, 0.0) or 0.0)

    def _candidate_confirmed(pattern_key: str) -> bool:
        return bool(candidate_breakout_map.get(pattern_key, False))

    def _candidate_metric(pattern_key: str, metric_key: str) -> float:
        details = candidate_detail_map.get(pattern_key, {}) or {}
        return float(details.get(metric_key, 0.0) or 0.0)

    double_bottom_score = _candidate_score("double_bottom")
    double_top_score = _candidate_score("double_top")
    hidden_base_score = _candidate_score("hidden_base_breakout")
    hidden_distribution_score = _candidate_score("hidden_distribution_breakdown")
    symmetrical_triangle_score = _candidate_score("symmetrical_triangle")
    bullish_flag_score = _candidate_score("bullish_flag")
    bearish_flag_score = _candidate_score("bearish_flag")
    v_reversal_score = _candidate_score("v_shaped_reversal")
    inverted_v_score = _candidate_score("inverted_v_reversal")
    support_bounce_score = _candidate_score("support_bounce")
    resistance_rejection_score = _candidate_score("resistance_rejection")
    double_bottom_breakout_margin = _candidate_metric("double_bottom", "breakout_margin_pct")
    double_top_breakout_margin = _candidate_metric("double_top", "breakout_margin_pct")
    bullish_flag_retrace_pct = _candidate_metric("bullish_flag", "retrace_pct")
    bearish_flag_retrace_pct = _candidate_metric("bearish_flag", "retrace_pct")
    support_bounce_return_pct = _candidate_metric("support_bounce", "reaction_return_pct")
    resistance_rejection_return_pct = _candidate_metric("resistance_rejection", "reaction_return_pct")
    bullish_break_ready = (
        structure_break_state == "bullish_break"
        and swing_quality_score >= 0.48
        and structure_break_distance_pct >= 0.12
        and latest_close_near_high_ratio >= 0.56
        and rejection_wick_bias != "bearish"
        and (
            breakout_state == "bullish_breakout"
            or recent_breakout_followthrough_score >= 0.2
            or level_reclaim_state == "bullish_reclaim"
            or pattern_bias == "bullish"
        )
    )
    bearish_break_ready = (
        structure_break_state == "bearish_break"
        and swing_quality_score >= 0.48
        and structure_break_distance_pct >= 0.12
        and latest_close_near_low_ratio >= 0.56
        and rejection_wick_bias != "bullish"
        and (
            breakout_state == "bearish_breakdown"
            or recent_breakout_followthrough_score >= 0.2
            or level_reclaim_state == "bearish_reclaim"
            or pattern_bias == "bearish"
        )
    )

    semantic_label = "neutral_structure"
    semantic_bias = "NONE"
    semantic_score = 0.0
    reasons: List[str] = []

    if (
        _candidate_confirmed("hidden_base_breakout")
        and hidden_base_score >= 0.62
        and (
            breakout_authenticity_score >= 0.3
            or latest_close_near_high_ratio >= 0.62
            or recent_consecutive_up_closes >= 2
        )
    ):
        semantic_label = "hidden_base_release"
        semantic_bias = "LONG"
        semantic_score = 0.66 + 0.16 * max(hidden_base_score, breakout_authenticity_score)
        reasons.append("A candidate hidden base breakout is already confirmed and should be treated as a real bullish release")
    elif (
        _candidate_confirmed("hidden_distribution_breakdown")
        and hidden_distribution_score >= 0.62
        and (
            breakout_authenticity_score >= 0.3
            or latest_close_near_low_ratio >= 0.62
            or recent_consecutive_down_closes >= 2
        )
    ):
        semantic_label = "hidden_distribution_release"
        semantic_bias = "SHORT"
        semantic_score = 0.66 + 0.16 * max(hidden_distribution_score, breakout_authenticity_score)
        reasons.append("A candidate hidden distribution breakdown is already confirmed and should be treated as a real bearish release")
    elif (
        symmetrical_triangle_score >= 0.7
        and trend_alignment_state == "bullish_short_vs_bearish_long"
        and breakout_authenticity_score >= 0.16
        and breakout_body_ratio >= 0.3
        and latest_close_near_high_ratio >= 0.56
        and rejection_wick_bias != "bearish"
        and recent_consecutive_up_closes >= 1
        and (
            structure_break_state == "bullish_break"
            or breakout_state == "bullish_breakout"
            or recent_breakout_followthrough_score >= 0.12
        )
        and (
            macd_cross == "bullish_cross"
            or macd_hist >= -0.02
            or rsi_divergence == "bullish_divergence"
        )
    ):
        semantic_label = "bullish_structure_break"
        semantic_bias = "LONG"
        semantic_score = (
            0.58
            + 0.1 * min(1.0, symmetrical_triangle_score)
            + 0.08 * min(1.0, breakout_authenticity_score)
            + 0.06 * min(1.0, latest_close_near_high_ratio)
        )
        reasons.append("Compressed triangle structure is breaking upward against a stale bearish backdrop")
        reasons.append("Bullish candle quality and short-vs-long structure shift both support upside regime handoff")
    elif (
        trend_features.get("trend_alignment_state") == "bullish_short_vs_bearish_long"
        and structure_break_state == "bullish_break"
        and latest_close_near_high_ratio >= 0.56
        and recent_consecutive_up_closes >= 2
        and (
            macd_cross == "bullish_cross"
            or rsi_divergence == "bullish_divergence"
            or macd_hist >= 0
        )
    ):
        semantic_label = "bullish_structure_break"
        semantic_bias = "LONG"
        semantic_score = 0.6 + 0.12 * swing_quality_score + 0.08 * min(1.0, latest_close_near_high_ratio)
        reasons.append("Short-term upside structure has already broken higher against the stale bearish backdrop")
    elif (
        trend_features.get("trend_alignment_state") == "bearish_short_vs_bullish_long"
        and structure_break_state == "bearish_break"
        and latest_close_near_low_ratio >= 0.56
        and recent_consecutive_down_closes >= 2
        and (
            macd_cross == "bearish_cross"
            or rsi_divergence == "bearish_divergence"
            or macd_hist <= 0
        )
    ):
        semantic_label = "bearish_structure_break"
        semantic_bias = "SHORT"
        semantic_score = 0.6 + 0.12 * swing_quality_score + 0.08 * min(1.0, latest_close_near_low_ratio)
        reasons.append("Short-term downside structure has already broken lower against the stale bullish backdrop")
    elif (
        resistance_rejection_score >= 0.66
        and location_state == "near_resistance"
        and latest_close_near_low_ratio >= 0.54
        and breakout_body_ratio >= 0.32
        and rejection_wick_bias == "bearish"
        and (
            recent_consecutive_down_closes >= 1
            or breakout_state == "testing_resistance"
            or recent_breakout_failure_score >= 0.16
        )
        and (
            macd_cross == "bearish_cross"
            or rsi_divergence == "bearish_divergence"
            or rsi_state in {"overbought", "bearish"}
            or stoch_state in {"overbought", "bearish"}
        )
        and trend_alignment_state in {"aligned_bullish", "bearish_short_vs_bullish_long", "mixed_transition"}
    ):
        semantic_label = "resistance_failure_rotation"
        semantic_bias = "SHORT"
        semantic_score = (
            0.56
            + 0.08 * min(1.0, resistance_rejection_score)
            + 0.06 * min(1.0, latest_close_near_low_ratio)
            + 0.05 * min(1.0, breakout_body_ratio)
        )
        reasons.append("Price was rejected at resistance with a bearish close near the low of the bar")
        reasons.append("This looks more like failed upside continuation and short-term downside rotation than a clean bullish breakout")
    elif (
        double_bottom_score >= 0.74
        and breakout_authenticity_score >= 0.3
        and breakout_body_ratio >= 0.6
        and latest_close_near_high_ratio >= 0.58
        and rejection_wick_bias != "bearish"
        and (
            recent_consecutive_up_closes >= 1
            or breakout_state == "bullish_breakout"
            or recent_breakout_followthrough_score >= 0.16
        )
        and location_state != "near_resistance"
        and (
            level_reclaim_state == "bullish_reclaim"
            or structure_break_state == "bullish_break"
            or trend_alignment_state in {"aligned_bullish", "bullish_short_vs_bearish_long"}
            or macd_cross == "bullish_cross"
            or macd_hist >= 0
        )
        and not (
            double_top_score >= double_bottom_score - 0.02
            and (
                rsi_divergence == "bearish_divergence"
                or macd_cross == "bearish_cross"
                or rsi_state in {"overbought", "bearish"}
                or stoch_state in {"overbought", "bearish"}
            )
        )
    ):
        semantic_label = "support_reclaim_rotation"
        semantic_bias = "LONG"
        semantic_score = (
            0.58
            + 0.1 * min(1.0, double_bottom_score)
            + 0.08 * min(1.0, breakout_authenticity_score)
            + 0.05 * min(1.0, breakout_body_ratio)
        )
        reasons.append("Bottoming base is not just forming; it is already reclaiming upward with a breakout-style candle")
        reasons.append("Support reclaim structure plus candle quality favors bullish rotation over stale downside inertia")
    elif (
        double_bottom_score >= 0.84
        and breakout_authenticity_score >= 0.32
        and latest_close_near_high_ratio >= 0.62
        and recent_consecutive_up_closes >= 2
        and location_state != "near_resistance"
        and (
            macd_cross == "bullish_cross"
            or rsi_divergence == "bullish_divergence"
            or macd_hist >= 0
        )
    ):
        semantic_label = "support_reclaim_rotation"
        semantic_bias = "LONG"
        semantic_score = 0.56 + 0.12 * min(1.0, double_bottom_score) + 0.08 * min(1.0, breakout_authenticity_score)
        reasons.append("A high-quality bottoming base is rotating upward with breakout-style candle support")
    elif (
        double_top_score >= 0.84
        and breakout_authenticity_score >= 0.32
        and latest_close_near_low_ratio >= 0.62
        and recent_consecutive_down_closes >= 2
        and location_state != "near_support"
        and (
            macd_cross == "bearish_cross"
            or rsi_divergence == "bearish_divergence"
            or macd_hist <= 0
        )
    ):
        semantic_label = "resistance_failure_rotation"
        semantic_bias = "SHORT"
        semantic_score = 0.56 + 0.12 * min(1.0, double_top_score) + 0.08 * min(1.0, breakout_authenticity_score)
        reasons.append("A high-quality topping base is rotating downward with breakdown-style candle support")
    elif (
        pattern_name == "hidden_base_breakout"
        and breakout_confirmed
        and breakout_authenticity_score >= 0.4
    ):
        semantic_label = "hidden_base_release"
        semantic_bias = "LONG"
        semantic_score = 0.7 + 0.2 * pattern_geometry_score
        reasons.append("Hidden base compression has already released upward")
    elif (
        pattern_name == "hidden_distribution_breakdown"
        and breakout_confirmed
        and breakout_authenticity_score >= 0.4
    ):
        semantic_label = "hidden_distribution_release"
        semantic_bias = "SHORT"
        semantic_score = 0.7 + 0.2 * pattern_geometry_score
        reasons.append("Hidden distribution has already released downward")
    elif (
        breakout_confirmed
        and breakout_retest_quality == "healthy"
        and breakout_authenticity_score >= 0.38
        and pattern_bias == "bullish"
    ):
        semantic_label = "confirmed_breakout_with_retest"
        semantic_bias = "LONG"
        semantic_score = 0.66 + 0.18 * breakout_authenticity_score
        reasons.append("Bullish breakout survived an early retest")
    elif (
        breakout_confirmed
        and breakout_retest_quality == "healthy"
        and breakout_authenticity_score >= 0.38
        and pattern_bias == "bearish"
    ):
        semantic_label = "confirmed_breakdown_with_retest"
        semantic_bias = "SHORT"
        semantic_score = 0.66 + 0.18 * breakout_authenticity_score
        reasons.append("Bearish breakdown survived an early retest")
    elif (
        market_regime == "compression"
        and swing_compression_score >= 0.32
        and breakout_state == "bullish_breakout"
        and recent_breakout_followthrough_score >= 0.35
    ):
        semantic_label = "compression_release_breakout"
        semantic_bias = "LONG"
        semantic_score = 0.62 + 0.16 * swing_compression_score
        reasons.append("Compressed structure is releasing upward with follow-through")
    elif (
        market_regime == "compression"
        and swing_compression_score >= 0.32
        and breakout_state == "bearish_breakdown"
        and recent_breakout_followthrough_score >= 0.35
    ):
        semantic_label = "compression_release_breakdown"
        semantic_bias = "SHORT"
        semantic_score = 0.62 + 0.16 * swing_compression_score
        reasons.append("Compressed structure is releasing downward with follow-through")
    elif bullish_break_ready:
        semantic_label = "bullish_structure_break"
        semantic_bias = "LONG"
        semantic_score = (
            0.52
            + 0.14 * swing_quality_score
            + min(0.08, structure_break_distance_pct / 4.0)
            + min(0.08, recent_breakout_followthrough_score * 0.2)
        )
        reasons.append("Price broke the latest swing high and changed local structure")
    elif bearish_break_ready:
        semantic_label = "bearish_structure_break"
        semantic_bias = "SHORT"
        semantic_score = (
            0.52
            + 0.14 * swing_quality_score
            + min(0.08, structure_break_distance_pct / 4.0)
            + min(0.08, recent_breakout_followthrough_score * 0.2)
        )
        reasons.append("Price broke the latest swing low and changed local structure")
    elif (
        not _candidate_confirmed("double_bottom")
        and double_bottom_score >= 0.58
        and double_bottom_breakout_margin >= -0.45
        and not (
            double_top_score >= double_bottom_score - 0.03
            and (
                rsi_divergence == "bearish_divergence"
                or macd_cross == "bearish_cross"
                or rsi_state in {"overbought", "bearish"}
                or stoch_state in {"overbought", "bearish"}
            )
        )
        and (
            rsi_divergence == "bullish_divergence"
            or macd_cross == "bullish_cross"
            or macd_hist >= -0.03
            or recent_consecutive_up_closes >= 2
        )
        and location_state in {"near_support", "mid_range"}
    ):
        semantic_label = "developing_double_bottom_pressure"
        semantic_bias = "LONG"
        semantic_score = (
            0.5
            + 0.14 * min(1.0, double_bottom_score)
            + 0.08 * min(1.0, max(0.0, double_bottom_breakout_margin + 0.45) / 0.45)
        )
        reasons.append("A double-bottom style base is pressing toward neckline confirmation")
        reasons.append("Momentum and closing behavior suggest the bearish leg is weakening")
    elif (
        not _candidate_confirmed("double_top")
        and double_top_score >= 0.58
        and double_top_breakout_margin >= -0.45
        and not (
            double_bottom_score >= double_top_score - 0.03
            and (
                rsi_divergence == "bullish_divergence"
                or macd_cross == "bullish_cross"
                or rsi_state in {"oversold", "bullish"}
                or stoch_state in {"oversold", "bullish"}
            )
        )
        and (
            rsi_divergence == "bearish_divergence"
            or macd_cross == "bearish_cross"
            or macd_hist <= 0.03
            or recent_consecutive_down_closes >= 2
        )
        and location_state in {"near_resistance", "mid_range"}
    ):
        semantic_label = "developing_double_top_pressure"
        semantic_bias = "SHORT"
        semantic_score = (
            0.5
            + 0.14 * min(1.0, double_top_score)
            + 0.08 * min(1.0, max(0.0, double_top_breakout_margin + 0.45) / 0.45)
        )
        reasons.append("A double-top style ceiling is pressing toward neckline failure")
        reasons.append("Momentum and closing behavior suggest the bullish leg is weakening")
    elif (
        bullish_flag_score >= 0.58
        and bullish_flag_retrace_pct <= 2.0
        and trend_direction != "downtrend"
        and (
            breakout_state == "bullish_breakout"
            or recent_breakout_followthrough_score >= 0.14
            or recent_consecutive_up_closes >= 2
        )
    ):
        semantic_label = "developing_bullish_flag_pressure"
        semantic_bias = "LONG"
        semantic_score = 0.5 + 0.15 * min(1.0, bullish_flag_score) + 0.06 * max(0.0, 1.0 - bullish_flag_retrace_pct / 2.0)
        reasons.append("The pullback remains shallow enough to look like a bullish flag rather than exhaustion")
        reasons.append("Short-horizon closes suggest the compression is leaning upward")
    elif (
        bearish_flag_score >= 0.58
        and bearish_flag_retrace_pct <= 2.0
        and trend_direction != "uptrend"
        and (
            breakout_state == "bearish_breakdown"
            or recent_breakout_followthrough_score >= 0.14
            or recent_consecutive_down_closes >= 2
        )
    ):
        semantic_label = "developing_bearish_flag_pressure"
        semantic_bias = "SHORT"
        semantic_score = 0.5 + 0.15 * min(1.0, bearish_flag_score) + 0.06 * max(0.0, 1.0 - bearish_flag_retrace_pct / 2.0)
        reasons.append("The rebound remains shallow enough to look like a bearish flag rather than reversal")
        reasons.append("Short-horizon closes suggest the compression is leaning downward")
    elif (
        trend_features.get("trend_alignment_state") == "bullish_short_vs_bearish_long"
        and recent_consecutive_up_closes >= 2
        and latest_close_near_high_ratio >= 0.58
        and rejection_wick_bias != "bearish"
        and (
            macd_cross == "bullish_cross"
            or rsi_divergence == "bullish_divergence"
            or macd_hist >= 0
            or rsi_state in {"bullish", "oversold"}
        )
        and double_top_score < 0.84
    ):
        semantic_label = "emerging_upside_rotation"
        semantic_bias = "LONG"
        semantic_score = (
            0.5
            + 0.08 * min(1.0, latest_close_near_high_ratio)
            + 0.08 * min(1.0, max(0.0, macd_hist + 0.2))
            + 0.08 * min(1.0, recent_consecutive_up_closes / 3.0)
        )
        reasons.append("Short-term structure is rotating upward even though the older bearish backdrop has not fully flipped yet")
        reasons.append("Strong closes near the high suggest breakout pressure rather than a simple bearish pullback")
    elif (
        trend_features.get("trend_alignment_state") == "bearish_short_vs_bullish_long"
        and recent_consecutive_down_closes >= 2
        and latest_close_near_low_ratio >= 0.58
        and rejection_wick_bias != "bullish"
        and (
            macd_cross == "bearish_cross"
            or rsi_divergence == "bearish_divergence"
            or macd_hist <= 0
            or rsi_state in {"bearish", "overbought"}
        )
        and double_bottom_score < 0.84
    ):
        semantic_label = "emerging_downside_rotation"
        semantic_bias = "SHORT"
        semantic_score = (
            0.5
            + 0.08 * min(1.0, latest_close_near_low_ratio)
            + 0.08 * min(1.0, max(0.0, -macd_hist + 0.2))
            + 0.08 * min(1.0, recent_consecutive_down_closes / 3.0)
        )
        reasons.append("Short-term structure is rotating downward even though the older bullish backdrop has not fully flipped yet")
        reasons.append("Strong closes near the low suggest breakdown pressure rather than a simple bullish rebound")
    elif (
        (support_bounce_score >= 0.56 or v_reversal_score >= 0.6)
        and location_state == "near_support"
        and (
            last_candle_lower_wick_ratio >= 0.34
            or latest_close_near_high_ratio >= 0.6
            or support_bounce_return_pct >= 0.35
        )
        and (
            rsi_state in {"oversold", "bullish"}
            or stoch_state in {"oversold", "bullish"}
            or macd_hist >= -0.03
        )
    ):
        semantic_label = "support_reclaim_rotation"
        semantic_bias = "LONG"
        semantic_score = (
            0.49
            + 0.12 * max(support_bounce_score, v_reversal_score)
            + 0.06 * min(1.0, latest_close_near_high_ratio)
            + 0.04 * min(1.0, last_candle_lower_wick_ratio + last_candle_body_ratio)
        )
        reasons.append("Support is being reclaimed with a strong closing response instead of passive oversold drift")
        reasons.append("This often precedes the kind of short-horizon bullish turn that image-driven agents catch")
    elif (
        (resistance_rejection_score >= 0.56 or inverted_v_score >= 0.6)
        and location_state == "near_resistance"
        and (
            last_candle_upper_wick_ratio >= 0.34
            or latest_close_near_low_ratio >= 0.6
            or resistance_rejection_return_pct >= 0.35
        )
        and (
            rsi_state in {"overbought", "bearish"}
            or stoch_state in {"overbought", "bearish"}
            or macd_hist <= 0.03
        )
    ):
        semantic_label = "resistance_failure_rotation"
        semantic_bias = "SHORT"
        semantic_score = (
            0.49
            + 0.12 * max(resistance_rejection_score, inverted_v_score)
            + 0.06 * min(1.0, latest_close_near_low_ratio)
            + 0.04 * min(1.0, last_candle_upper_wick_ratio + last_candle_body_ratio)
        )
        reasons.append("Resistance is rejecting price with a strong closing failure instead of healthy continuation")
        reasons.append("This often precedes the kind of short-horizon bearish turn that image-driven agents catch")
    elif (
        level_reclaim_state == "failed_bullish_breakout_reentry"
        or (rejection_wick_bias == "bearish" and latest_close_near_low_ratio >= 0.62)
    ):
        semantic_label = "false_breakout_reentry"
        semantic_bias = "SHORT"
        semantic_score = 0.6 + 0.14 * recent_breakout_failure_score
        reasons.append("Price rejected the upside breakout and slipped back inside the prior structure")
    elif (
        level_reclaim_state == "failed_bearish_breakdown_reentry"
        or (rejection_wick_bias == "bullish" and latest_close_near_high_ratio >= 0.62)
    ):
        semantic_label = "false_breakdown_reentry"
        semantic_bias = "LONG"
        semantic_score = 0.6 + 0.14 * recent_breakout_failure_score
        reasons.append("Price rejected the downside breakdown and reclaimed the prior structure")
    elif (
        pattern_name == "double_bottom"
        and breakout_confirmed
        and (
            breakout_authenticity_score >= 0.14
            or recent_breakout_followthrough_score >= 0.16
            or latest_close_near_high_ratio >= 0.66
        )
        and (
            latest_close_near_high_ratio >= 0.56
            or last_candle_body_ratio >= 0.28
            or level_reclaim_state in {"bullish_reclaim", "failed_bearish_breakdown_reentry"}
        )
    ):
        semantic_label = "hidden_base_release"
        semantic_bias = "LONG"
        semantic_score = (
            0.56
            + 0.12 * min(1.0, breakout_authenticity_score * 2.0)
            + 0.08 * min(1.0, recent_breakout_followthrough_score * 2.4)
            + 0.05 * min(1.0, latest_close_near_high_ratio)
        )
        reasons.append("Confirmed double-bottom breakout is treated as a real bullish base release rather than a neutral setup")
        reasons.append("Strong close behavior and follow-through make this closer to the image-driven bullish release cases")
    elif (
        pattern_name == "double_top"
        and breakout_confirmed
        and (
            breakout_authenticity_score >= 0.14
            or recent_breakout_followthrough_score >= 0.16
            or latest_close_near_low_ratio >= 0.66
        )
        and (
            latest_close_near_low_ratio >= 0.56
            or last_candle_body_ratio >= 0.28
            or level_reclaim_state in {"bearish_reclaim", "failed_bullish_breakout_reentry"}
        )
    ):
        semantic_label = "hidden_distribution_release"
        semantic_bias = "SHORT"
        semantic_score = (
            0.56
            + 0.12 * min(1.0, breakout_authenticity_score * 2.0)
            + 0.08 * min(1.0, recent_breakout_followthrough_score * 2.4)
            + 0.05 * min(1.0, latest_close_near_low_ratio)
        )
        reasons.append("Confirmed double-top breakdown is treated as a real bearish distribution release rather than a neutral setup")
        reasons.append("Strong close failure and follow-through make this closer to the image-driven bearish release cases")
    elif (
        location_state == "near_support"
        and pattern_name == "support_bounce"
        and recent_breakout_followthrough_score < 0.22
    ):
        semantic_label = "support_bounce_without_followthrough"
        semantic_bias = "LONG"
        semantic_score = 0.46 + 0.12 * pattern_geometry_score
        reasons.append("Support bounce exists but follow-through is still weak")
    elif (
        location_state == "near_resistance"
        and pattern_name == "resistance_rejection"
        and recent_breakout_followthrough_score < 0.22
    ):
        semantic_label = "resistance_rejection_without_followthrough"
        semantic_bias = "SHORT"
        semantic_score = 0.46 + 0.12 * pattern_geometry_score
        reasons.append("Resistance rejection exists but downside follow-through is still weak")
    elif (
        channel_direction in {"ascending", "descending"}
        and swing_bias in {"bullish", "bearish"}
        and swing_bias == ("bullish" if trend_direction == "uptrend" else "bearish")
        and macd_hist != 0
    ):
        semantic_label = "trend_channel_continuation"
        semantic_bias = "LONG" if trend_direction == "uptrend" else "SHORT" if trend_direction == "downtrend" else "NONE"
        semantic_score = 0.48 + 0.1 * swing_quality_score
        reasons.append("Swing structure still respects the dominant trend channel")

    semantic_score = round(min(1.0, max(0.0, semantic_score)), 4)
    return {
        "structure_semantic_label": semantic_label,
        "structure_semantic_bias": semantic_bias,
        "structure_semantic_score": semantic_score,
        "structure_semantic_reasons": reasons[:3],
    }


def _classify_trend_continuation_quality(
    direction: str,
    breakout_state: str,
    location_state: str,
    trend_fit_r2: float,
    adx_value: float | None,
) -> str:
    """Estimate whether the current trend still has healthy continuation quality."""

    if direction not in {"uptrend", "downtrend"}:
        return "low"

    if direction == "uptrend" and location_state == "near_resistance" and breakout_state != "bullish_breakout":
        return "low"
    if direction == "downtrend" and location_state == "near_support" and breakout_state != "bearish_breakdown":
        return "low"

    adx_score = adx_value or 0.0
    if trend_fit_r2 >= 0.3 and adx_score >= 25:
        return "high"
    if trend_fit_r2 >= 0.15 and adx_score >= 20:
        return "medium"
    return "low"


def _classify_trend_exhaustion_risk(
    direction: str,
    breakout_state: str,
    location_state: str,
) -> str:
    """Estimate whether a trend is at risk of exhausting near a key level."""

    if direction == "uptrend" and location_state == "near_resistance" and breakout_state != "bullish_breakout":
        return "high"
    if direction == "downtrend" and location_state == "near_support" and breakout_state != "bearish_breakdown":
        return "high"
    if direction in {"uptrend", "downtrend"} and breakout_state in {"testing_resistance", "testing_support"}:
        return "medium"
    return "low"


def _classify_quality_band(score: float, high_threshold: float, medium_threshold: float) -> str:
    """Map a 0-1 quality score into stable semantic bands for downstream arbitration."""

    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _compute_reversal_signal(
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
    risk_features: Dict[str, object],
) -> Dict[str, object]:
    """
    Detect whether the current sample resembles a structure-transition setup.

    Why this matters:
    QuantAgent tends to win when price structure has already started rotating
    but traditional directional features still lag. This helper makes that
    regime explicit so KuantAgent can grant AI more authority only on those
    hard cases.
    """

    trend_direction = str(trend_features.get("trend_direction", "sideways"))
    breakout_state = str(trend_features.get("breakout_state", "inside_range"))
    location_state = str(trend_features.get("location_state", "mid_range"))
    trend_exhaustion_risk = str(trend_features.get("trend_exhaustion_risk", "low"))
    trend_alignment = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    rsi_state = str(indicator_features.get("rsi_state", "neutral"))
    stoch_state = str(indicator_features.get("stoch_state", "neutral"))
    rsi_divergence = str(indicator_features.get("rsi_divergence", "none"))
    macd_cross = str(indicator_features.get("macd_cross", "none"))
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))

    bullish_score = 0.0
    bearish_score = 0.0
    bullish_reasons: List[str] = []
    bearish_reasons: List[str] = []

    if trend_direction in {"downtrend", "compression", "sideways"} and pattern_bias == "bullish":
        bullish_score += 0.8
        bullish_reasons.append("bullish pattern is trying to reverse a weak/old downside structure")
    if trend_direction in {"uptrend", "compression", "sideways"} and pattern_bias == "bearish":
        bearish_score += 0.8
        bearish_reasons.append("bearish pattern is trying to reverse a weak/old upside structure")

    if breakout_state == "bullish_breakout" or (breakout_confirmed and pattern_bias == "bullish"):
        bullish_score += 1.0
        bullish_reasons.append("bullish breakout confirmation is present")
    elif breakout_state == "testing_support" and pattern_bias == "bullish":
        bullish_score += 0.35
        bullish_reasons.append("price is still holding around support during a bullish structure")

    if breakout_state == "bearish_breakdown" or (breakout_confirmed and pattern_bias == "bearish"):
        bearish_score += 1.0
        bearish_reasons.append("bearish breakdown confirmation is present")
    elif breakout_state == "testing_resistance" and pattern_bias == "bearish":
        bearish_score += 0.35
        bearish_reasons.append("price is still failing around resistance during a bearish structure")

    if location_state == "near_support":
        bullish_score += 0.45
        bullish_reasons.append("location is favorable for a support-led bullish turn")
    elif location_state == "near_resistance":
        bearish_score += 0.45
        bearish_reasons.append("location is favorable for a resistance-led bearish turn")

    if rsi_divergence == "bullish_divergence":
        bullish_score += 0.75
        bullish_reasons.append("bullish divergence suggests downside momentum is decaying")
    elif rsi_divergence == "bearish_divergence":
        bearish_score += 0.75
        bearish_reasons.append("bearish divergence suggests upside momentum is decaying")

    if macd_cross == "bullish_cross":
        bullish_score += 0.55
        bullish_reasons.append("MACD has already turned bullish")
    elif macd_cross == "bearish_cross":
        bearish_score += 0.55
        bearish_reasons.append("MACD has already turned bearish")

    if rsi_state == "oversold" or stoch_state == "oversold":
        bullish_score += 0.4
        bullish_reasons.append("oscillators are in an oversold rebound zone")
    if rsi_state == "overbought" or stoch_state == "overbought":
        bearish_score += 0.4
        bearish_reasons.append("oscillators are in an overbought pullback zone")

    pattern_name = str(pattern_features.get("pattern", "none"))

    if trend_exhaustion_risk == "high" and trend_direction == "downtrend":
        bullish_score += 0.45
        bullish_reasons.append("the previous downtrend looks exhausted near a key level")
    elif trend_exhaustion_risk == "high" and trend_direction == "uptrend":
        bearish_score += 0.45
        bearish_reasons.append("the previous uptrend looks exhausted near a key level")

    if trend_alignment == "bullish_short_vs_bearish_long":
        bullish_score += 0.7
        bullish_reasons.append("short-term slope has flipped up against an older bearish trend")
    elif trend_alignment == "bearish_short_vs_bullish_long":
        bearish_score += 0.7
        bearish_reasons.append("short-term slope has flipped down against an older bullish trend")

    if pattern_name == "v_shaped_reversal":
        bullish_score += 0.35
        bullish_reasons.append("V-shaped rebound often matters on short horizons")
    elif pattern_name == "inverted_v_reversal":
        bearish_score += 0.35
        bearish_reasons.append("inverted-V rejection often matters on short horizons")

    # Guardrail: if local structure and short-term slope still point upward,
    # do not over-confirm a bearish transition from resistance alone.
    if (
        trend_features.get("high_low_structure") == "higher_high_higher_low"
        and trend_alignment == "bullish_short_vs_bearish_long"
        and breakout_state != "bearish_breakdown"
        and pattern_name not in {"resistance_rejection", "inverted_v_reversal"}
    ):
        bearish_score = max(0.0, bearish_score - 0.55)
        bearish_reasons.append("bearish reversal is downgraded because short-term structure is still climbing")
    if (
        trend_features.get("high_low_structure") == "lower_high_lower_low"
        and trend_alignment == "bearish_short_vs_bullish_long"
        and breakout_state != "bullish_breakout"
        and pattern_name not in {"support_bounce", "v_shaped_reversal"}
    ):
        bullish_score = max(0.0, bullish_score - 0.55)
        bullish_reasons.append("bullish reversal is downgraded because short-term structure is still falling")

    if false_breakout_risk == "high":
        bullish_score -= 0.15
        bearish_score -= 0.15

    bullish_score = round(max(0.0, bullish_score), 4)
    bearish_score = round(max(0.0, bearish_score), 4)
    reversal_bias = "none"
    reversal_score = 0.0
    reversal_confirmed = False
    reversal_reasons: List[str] = []

    if bullish_score >= bearish_score and bullish_score >= 1.9:
        reversal_bias = "bullish_reversal"
        reversal_score = bullish_score
        reversal_confirmed = bullish_score >= 2.1 and (
            breakout_state == "bullish_breakout"
            or breakout_confirmed
            or trend_alignment == "bullish_short_vs_bearish_long"
            or (
                location_state == "near_support"
                and rsi_divergence == "bullish_divergence"
            )
        )
        reversal_reasons = bullish_reasons
    elif bearish_score > bullish_score and bearish_score >= 1.9:
        reversal_bias = "bearish_reversal"
        reversal_score = bearish_score
        reversal_confirmed = bearish_score >= 2.1 and (
            breakout_state == "bearish_breakdown"
            or breakout_confirmed
            or trend_alignment == "bearish_short_vs_bullish_long"
            or (
                location_state == "near_resistance"
                and rsi_divergence == "bearish_divergence"
            )
        )
        reversal_reasons = bearish_reasons

    return {
        "reversal_bias": reversal_bias,
        "reversal_score": round(reversal_score, 4),
        "reversal_confirmed": reversal_confirmed,
        "bullish_reversal_score": bullish_score,
        "bearish_reversal_score": bearish_score,
        "reversal_reasons": reversal_reasons[:4],
    }


def extract_decision_features(
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
    risk_features: Dict[str, object],
    forecast_horizon_bars: int = 1,
) -> Dict[str, object]:
    """
    Build a compact decision context from lower-level features.

    Why this matters:
    The algorithm layer should not stop at 'many signals'. It should organize
    those signals into a cleaner decision surface so the LLM can spend its
    reasoning budget on conflict resolution and explanation rather than on
    low-level bookkeeping.
    """

    indicator_bias = _score_indicator_bias(indicator_features)
    pattern_bias = _score_pattern_bias(pattern_features)
    trend_bias = _score_trend_bias(trend_features)
    risk_context = _score_risk_context(risk_features)
    short_horizon_signal = _compute_short_horizon_reaction_signal(
        indicator_features=indicator_features,
        pattern_features=pattern_features,
        trend_features=trend_features,
    )
    continuation_signal = _compute_continuation_signal(
        indicator_features=indicator_features,
        pattern_features=pattern_features,
        trend_features=trend_features,
        risk_features=risk_features,
    )

    long_score = (
        indicator_bias["indicator_long_score"]
        + pattern_bias["pattern_long_score"]
        + trend_bias["trend_long_score"]
    )
    short_score = (
        indicator_bias["indicator_short_score"]
        + pattern_bias["pattern_short_score"]
        + trend_bias["trend_short_score"]
    )

    pattern_completed = bool(pattern_features.get("pattern_completed", False))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    pattern_bias_name = pattern_features.get("pattern_bias", "neutral")
    pattern_name = str(pattern_features.get("pattern", "none"))
    pattern_geometry_score = float(pattern_features.get("pattern_geometry_score", 0.0) or 0.0)
    breakout_authenticity_score = float(pattern_features.get("breakout_authenticity_score", 0.0) or 0.0)
    breakout_body_ratio = float(pattern_features.get("breakout_body_ratio", 0.0) or 0.0)
    breakout_retest_quality = str(pattern_features.get("breakout_retest_quality", "unknown"))
    candidate_patterns = list(pattern_features.get("candidate_patterns", []) or [])
    candidate_pattern_summaries = list(pattern_features.get("candidate_pattern_summaries", []) or [])
    location_state = trend_features.get("location_state")
    breakout_state = trend_features.get("breakout_state")
    market_regime = trend_features.get("market_regime")
    trend_direction = str(trend_features.get("trend_direction", "unknown"))
    trend_alignment_state = str(
        trend_features.get("trend_alignment_state", "mixed_transition")
    )
    channel_direction = str(trend_features.get("channel_direction", "mixed"))
    structure_break_state = str(trend_features.get("structure_break_state", "none"))
    structure_break_distance_pct = float(trend_features.get("structure_break_distance_pct", 0.0) or 0.0)
    recent_breakout_followthrough_score = float(trend_features.get("recent_breakout_followthrough_score", 0.0) or 0.0)
    recent_breakout_failure_score = float(trend_features.get("recent_breakout_failure_score", 0.0) or 0.0)
    level_reclaim_state = str(trend_features.get("level_reclaim_state", "none"))
    breakout_margin_pct = float(trend_features.get("breakout_margin_pct", 0.0) or 0.0)
    trend_strength = float(trend_features.get("trend_strength_score", 0.0) or 0.0)
    trend_continuation_quality = trend_features.get("trend_continuation_quality", "medium")
    trend_exhaustion_risk = trend_features.get("trend_exhaustion_risk", "low")
    pattern_span = int(pattern_features.get("pattern_span", 0) or 0)
    macd_hist = float(indicator_features.get("macd_hist", 0.0) or 0.0)
    rsi_state = indicator_features.get("rsi_state", "neutral")
    stoch_state = indicator_features.get("stoch_state", "neutral")
    reversal_signal = _compute_reversal_signal(
        indicator_features=indicator_features,
        pattern_features=pattern_features,
        trend_features=trend_features,
        risk_features=risk_features,
    )
    reversal_bias = str(reversal_signal.get("reversal_bias", "none"))
    reversal_score = float(reversal_signal.get("reversal_score", 0.0) or 0.0)
    reversal_confirmed = bool(reversal_signal.get("reversal_confirmed", False))
    three_bar_path_signal = _compute_three_bar_path_signal(
        indicator_features=indicator_features,
        trend_features=trend_features,
        short_horizon_signal=short_horizon_signal,
        continuation_signal=continuation_signal,
        reversal_signal=reversal_signal,
    )
    three_bar_majority_bias = str(three_bar_path_signal.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(three_bar_path_signal.get("three_bar_path_score", 0.0) or 0.0)
    three_bar_path_consistency = float(three_bar_path_signal.get("three_bar_path_consistency", 0.0) or 0.0)
    short_horizon_bias = str(short_horizon_signal.get("short_horizon_bias", "none"))
    short_horizon_score = float(short_horizon_signal.get("short_horizon_score", 0.0) or 0.0)
    continuation_bias = str(continuation_signal.get("continuation_bias", "none"))
    continuation_score = float(continuation_signal.get("continuation_score", 0.0) or 0.0)
    continuation_exhaustion_risk = str(
        continuation_signal.get("continuation_exhaustion_risk", "low")
    )
    structure_semantics = _derive_structure_semantics(
        indicator_features=indicator_features,
        pattern_features=pattern_features,
        trend_features=trend_features,
    )
    structure_semantic_label = str(
        structure_semantics.get("structure_semantic_label", "neutral_structure")
    )
    structure_semantic_bias = str(
        structure_semantics.get("structure_semantic_bias", "NONE")
    ).upper()
    structure_semantic_score = float(
        structure_semantics.get("structure_semantic_score", 0.0) or 0.0
    )
    structure_semantic_reasons = list(
        structure_semantics.get("structure_semantic_reasons", [])
    )
    confirmed_bullish_candidates = [
        item
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_bottom", "hidden_base_breakout", "v_shaped_reversal"}
        and bool(item.get("breakout_confirmed", False))
        and float(item.get("confidence", 0.0) or 0.0) >= 0.72
    ]
    confirmed_bearish_candidates = [
        item
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_top", "hidden_distribution_breakdown", "inverted_v_reversal"}
        and bool(item.get("breakout_confirmed", False))
        and float(item.get("confidence", 0.0) or 0.0) >= 0.72
    ]
    bullish_candidate_hints = [
        item
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_bottom", "hidden_base_breakout", "v_shaped_reversal"}
        and float(item.get("confidence", 0.0) or 0.0) >= 0.78
    ]
    bearish_candidate_hints = [
        item
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_top", "hidden_distribution_breakdown", "inverted_v_reversal"}
        and float(item.get("confidence", 0.0) or 0.0) >= 0.78
    ]
    broad_bullish_candidate_count = sum(
        1
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_bottom", "hidden_base_breakout", "v_shaped_reversal"}
        and float(item.get("confidence", 0.0) or 0.0) >= 0.62
    )
    broad_bearish_candidate_count = sum(
        1
        for item in candidate_pattern_summaries
        if isinstance(item, dict)
        and item.get("pattern") in {"double_top", "hidden_distribution_breakdown", "inverted_v_reversal"}
        and float(item.get("confidence", 0.0) or 0.0) >= 0.62
    )
    bullish_candidate_hint_top_conf = max(
        (
            float(item.get("confidence", 0.0) or 0.0)
            for item in candidate_pattern_summaries
            if isinstance(item, dict)
            and item.get("pattern") in {"double_bottom", "hidden_base_breakout", "v_shaped_reversal"}
        ),
        default=0.0,
    )
    bearish_candidate_hint_top_conf = max(
        (
            float(item.get("confidence", 0.0) or 0.0)
            for item in candidate_pattern_summaries
            if isinstance(item, dict)
            and item.get("pattern") in {"double_top", "hidden_distribution_breakdown", "inverted_v_reversal"}
        ),
        default=0.0,
    )
    ambiguous_high_conf_candidate_conflict = (
        bullish_candidate_hint_top_conf >= 0.7
        and bearish_candidate_hint_top_conf >= 0.7
        and abs(bullish_candidate_hint_top_conf - bearish_candidate_hint_top_conf) <= 0.12
    )
    if (
        structure_semantic_label == "neutral_structure"
        and three_bar_majority_bias == "LONG"
        and (
            confirmed_bullish_candidates
            or (breakout_confirmed and pattern_name == "double_bottom" and breakout_authenticity_score >= 0.14)
        )
        and breakout_body_ratio >= 0.28
    ):
        structure_semantic_label = "hidden_base_release"
        structure_semantic_bias = "LONG"
        structure_semantic_score = max(
            structure_semantic_score,
            0.62 + min(0.16, breakout_authenticity_score * 0.22 + three_bar_path_score * 0.04),
        )
        structure_semantic_reasons = [
            "Confirmed bullish candidate structure is upgraded from neutral to bullish base release",
            "Three-bar path and candle quality support the same upside handoff",
        ]
    elif (
        structure_semantic_label == "neutral_structure"
        and three_bar_majority_bias == "SHORT"
        and (
            confirmed_bearish_candidates
            or (breakout_confirmed and pattern_name == "double_top" and breakout_authenticity_score >= 0.14)
        )
        and breakout_body_ratio >= 0.28
    ):
        structure_semantic_label = "hidden_distribution_release"
        structure_semantic_bias = "SHORT"
        structure_semantic_score = max(
            structure_semantic_score,
            0.62 + min(0.16, breakout_authenticity_score * 0.22 + three_bar_path_score * 0.04),
        )
        structure_semantic_reasons = [
            "Confirmed bearish candidate structure is upgraded from neutral to bearish distribution release",
            "Three-bar path and candle quality support the same downside handoff",
        ]
    if (
        structure_semantic_label == "developing_double_bottom_pressure"
        and three_bar_majority_bias == "LONG"
        and three_bar_path_score >= 0.8
        and (
            confirmed_bullish_candidates
            or breakout_authenticity_score >= 0.2
            or level_reclaim_state in {"bullish_reclaim", "failed_bearish_breakdown_reentry"}
        )
    ):
        structure_semantic_label = "support_reclaim_rotation"
        structure_semantic_bias = "LONG"
        structure_semantic_score = max(
            structure_semantic_score,
            0.64 + min(0.12, three_bar_path_score * 0.04 + breakout_authenticity_score * 0.18),
        )
        structure_semantic_reasons = [
            "Developing double-bottom is upgraded because reclaim-style behavior and path alignment already support a real bullish handoff",
            "This is treated as stronger than a generic early bullish pattern because confirmed candidate structure is already present",
        ]
    elif (
        structure_semantic_label == "developing_double_top_pressure"
        and three_bar_majority_bias == "SHORT"
        and three_bar_path_score >= 0.8
        and (
            confirmed_bearish_candidates
            or breakout_authenticity_score >= 0.2
            or level_reclaim_state in {"bearish_reclaim", "failed_bullish_breakout_reentry"}
        )
    ):
        structure_semantic_label = "resistance_failure_rotation"
        structure_semantic_bias = "SHORT"
        structure_semantic_score = max(
            structure_semantic_score,
            0.64 + min(0.12, three_bar_path_score * 0.04 + breakout_authenticity_score * 0.18),
        )
        structure_semantic_reasons = [
            "Developing double-top is upgraded because resistance-failure behavior and path alignment already support a real bearish handoff",
            "This is treated as stronger than a generic early bearish pattern because confirmed candidate structure is already present",
        ]
    votes = _collect_direction_votes(indicator_bias, pattern_features, trend_features)
    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")
    stable_three_bar_path = three_bar_path_consistency >= 0.99
    stale_bearish_regime = trend_direction == "downtrend" or trend_alignment_state in {
        "aligned_bearish",
        "bullish_short_vs_bearish_long",
    }
    stale_bullish_regime = trend_direction == "uptrend" or trend_alignment_state in {
        "aligned_bullish",
        "bearish_short_vs_bullish_long",
    }
    structure_confirmation_tier = "none"
    if (
        (
            breakout_confirmed
            and breakout_authenticity_score >= 0.46
            and pattern_bias_name in {"bullish", "bearish"}
        )
        or (
            structure_semantic_label in {
                "confirmed_breakout_with_retest",
                "confirmed_breakdown_with_retest",
                "compression_release_breakout",
                "compression_release_breakdown",
                "hidden_base_release",
                "hidden_distribution_release",
                "false_breakout_reentry",
                "false_breakdown_reentry",
                "support_reclaim_rotation",
                "resistance_failure_rotation",
            }
            and structure_semantic_score >= 0.62
        )
        or (
            structure_semantic_label in {"bullish_structure_break", "bearish_structure_break"}
            and structure_semantic_score >= 0.64
            and structure_break_state in {"bullish_break", "bearish_break"}
        )
        or (
            structure_semantic_label in {"support_reclaim_rotation", "resistance_failure_rotation"}
            and structure_semantic_score >= 0.56
            and (
                level_reclaim_state in {"bullish_reclaim", "bearish_reclaim"}
                or recent_breakout_failure_score >= 0.24
                or (
                    structure_semantic_bias in {"LONG", "SHORT"}
                    and three_bar_majority_bias == structure_semantic_bias
                    and three_bar_path_score >= 0.52
                )
            )
        )
        or (
            structure_semantic_label in {"bullish_structure_break", "bearish_structure_break"}
            and structure_semantic_score >= 0.58
            and (
                breakout_authenticity_score >= 0.28
                or (
                    structure_semantic_bias in {"LONG", "SHORT"}
                    and three_bar_majority_bias == structure_semantic_bias
                    and three_bar_path_score >= 0.6
                )
            )
        )
    ):
        structure_confirmation_tier = "confirmed"
    elif (
        structure_semantic_label != "neutral_structure"
        and structure_semantic_score >= 0.54
    ) or reversal_confirmed:
        structure_confirmation_tier = "developing"
    elif pattern_completed or reversal_bias != "none":
        structure_confirmation_tier = "candidate"

    trend_failure_score = 0.0
    trend_failure_state = "intact"
    trend_failure_reasons: List[str] = []
    if structure_break_state == "bullish_break" and stale_bearish_regime:
        trend_failure_score += 0.22
        trend_failure_reasons.append("Bearish regime swing structure has been broken upward")
    elif structure_break_state == "bearish_break" and stale_bullish_regime:
        trend_failure_score += 0.22
        trend_failure_reasons.append("Bullish regime swing structure has been broken downward")

    if level_reclaim_state == "bullish_reclaim" and stale_bearish_regime:
        trend_failure_score += 0.16
        trend_failure_reasons.append("Price reclaimed a failed support breakdown, weakening the bearish regime")
    elif level_reclaim_state == "bearish_reclaim" and stale_bullish_regime:
        trend_failure_score += 0.16
        trend_failure_reasons.append("Price reclaimed a failed resistance breakout, weakening the bullish regime")
    elif level_reclaim_state == "failed_bullish_breakout_reentry":
        trend_failure_score += 0.18
        trend_failure_reasons.append("Failed bullish breakout suggests the prior upside continuation thesis is stale")
    elif level_reclaim_state == "failed_bearish_breakdown_reentry":
        trend_failure_score += 0.18
        trend_failure_reasons.append("Failed bearish breakdown suggests the prior downside continuation thesis is stale")

    if (
        structure_semantic_label == "support_reclaim_rotation"
        and stale_bearish_regime
        and structure_semantic_score >= 0.56
        and (
            level_reclaim_state == "bullish_reclaim"
            or three_bar_majority_bias == "LONG"
            or recent_breakout_failure_score >= 0.24
        )
    ):
        trend_failure_score += 0.1
        trend_failure_reasons.append("Bullish support reclaim structure suggests the older bearish regime is already losing control")
    elif (
        structure_semantic_label == "resistance_failure_rotation"
        and stale_bullish_regime
        and structure_semantic_score >= 0.56
        and (
            level_reclaim_state == "bearish_reclaim"
            or three_bar_majority_bias == "SHORT"
            or recent_breakout_failure_score >= 0.24
        )
    ):
        trend_failure_score += 0.1
        trend_failure_reasons.append("Bearish resistance-failure structure suggests the older bullish regime is already losing control")

    if (
        structure_semantic_label == "hidden_base_release"
        and stale_bearish_regime
        and structure_semantic_score >= 0.66
    ):
        trend_failure_score += 0.1
        trend_failure_reasons.append("Confirmed hidden-base release implies the prior bearish structure is already stale")
    elif (
        structure_semantic_label == "hidden_distribution_release"
        and stale_bullish_regime
        and structure_semantic_score >= 0.66
    ):
        trend_failure_score += 0.1
        trend_failure_reasons.append("Confirmed hidden-distribution release implies the prior bullish structure is already stale")

    if breakout_confirmed and breakout_authenticity_score >= 0.46:
        if pattern_bias_name == "bullish" and stale_bearish_regime:
            trend_failure_score += 0.22
            trend_failure_reasons.append("Confirmed bullish breakout is large enough to challenge the older bearish regime")
        elif pattern_bias_name == "bearish" and stale_bullish_regime:
            trend_failure_score += 0.22
            trend_failure_reasons.append("Confirmed bearish breakdown is large enough to challenge the older bullish regime")

    if reversal_confirmed:
        if reversal_bias == "bullish_reversal" and stale_bearish_regime:
            trend_failure_score += 0.14
            trend_failure_reasons.append("Confirmed bullish reversal says the prior bearish trend may have failed")
        elif reversal_bias == "bearish_reversal" and stale_bullish_regime:
            trend_failure_score += 0.14
            trend_failure_reasons.append("Confirmed bearish reversal says the prior bullish trend may have failed")

    if (
        structure_semantic_label == "bullish_structure_break"
        and structure_semantic_score >= 0.58
        and stale_bearish_regime
    ):
        trend_failure_score += 0.16
        trend_failure_reasons.append("Bullish structure break directly challenges the stale bearish regime")
    elif (
        structure_semantic_label == "bearish_structure_break"
        and structure_semantic_score >= 0.58
        and stale_bullish_regime
    ):
        trend_failure_score += 0.16
        trend_failure_reasons.append("Bearish structure break directly challenges the stale bullish regime")

    if (
        structure_semantic_label == "support_reclaim_rotation"
        and structure_semantic_score >= 0.56
        and stale_bearish_regime
    ):
        trend_failure_score += 0.15
        trend_failure_reasons.append("Support reclaim rotation says the prior bearish continuation thesis may already be stale")
    elif (
        structure_semantic_label == "resistance_failure_rotation"
        and structure_semantic_score >= 0.56
        and stale_bullish_regime
    ):
        trend_failure_score += 0.15
        trend_failure_reasons.append("Resistance failure rotation says the prior bullish continuation thesis may already be stale")

    if structure_semantic_label in {"false_breakout_reentry", "false_breakdown_reentry"} and structure_semantic_score >= 0.62:
        trend_failure_score += 0.12
        trend_failure_reasons.append("False-break reentry is a direct sign that the previous continuation regime may have failed")

    trend_failure_score = round(min(1.0, max(0.0, trend_failure_score)), 4)
    if trend_failure_score >= 0.5:
        trend_failure_state = "confirmed_failure"
    elif trend_failure_score >= 0.28:
        trend_failure_state = "probable_failure"
    elif trend_failure_score >= 0.14:
        trend_failure_state = "early_failure"

    structure_followthrough_score = 0.0
    if structure_semantic_label != "neutral_structure":
        structure_followthrough_score += 0.12
    if breakout_confirmed:
        structure_followthrough_score += 0.1
    structure_followthrough_score += min(0.22, breakout_authenticity_score * 0.22)
    structure_followthrough_score += min(0.14, recent_breakout_followthrough_score * 0.2)
    if breakout_retest_quality == "healthy":
        structure_followthrough_score += 0.12
    elif breakout_retest_quality != "none":
        structure_followthrough_score += 0.05
    if (
        structure_semantic_bias in {"LONG", "SHORT"}
        and three_bar_majority_bias == structure_semantic_bias
        and three_bar_path_score >= 0.58
    ):
        structure_followthrough_score += 0.08
    if (
        structure_semantic_label in {
            "support_reclaim_rotation",
            "resistance_failure_rotation",
            "false_breakout_reentry",
            "false_breakdown_reentry",
        }
        and level_reclaim_state != "none"
    ):
        structure_followthrough_score += 0.06
    if location_state == "near_resistance" and structure_semantic_bias == "LONG" and breakout_state != "bullish_breakout":
        structure_followthrough_score -= 0.12
    elif location_state == "near_support" and structure_semantic_bias == "SHORT" and breakout_state != "bearish_breakdown":
        structure_followthrough_score -= 0.12
    if structure_semantic_bias == "LONG" and short_horizon_bias == "bearish_pullback_candidate":
        structure_followthrough_score -= 0.07
    elif structure_semantic_bias == "SHORT" and short_horizon_bias == "bullish_rebound_candidate":
        structure_followthrough_score -= 0.07
    structure_followthrough_score = round(min(1.0, max(0.0, structure_followthrough_score)), 4)
    structure_followthrough_state = _classify_quality_band(
        structure_followthrough_score,
        high_threshold=0.52,
        medium_threshold=0.34,
    )

    countertrend_pressure_score = 0.0
    countertrend_pressure_bias = "none"
    if structure_semantic_bias == "SHORT" or continuation_bias == "bearish_continuation_candidate":
        countertrend_pressure_bias = "LONG"
        if short_horizon_bias == "bullish_rebound_candidate":
            countertrend_pressure_score += 0.24 + min(0.18, short_horizon_score * 0.1)
        if location_state == "near_support" and breakout_state != "bearish_breakdown":
            countertrend_pressure_score += 0.14
        if bullish_candidate_hint_top_conf >= 0.7:
            countertrend_pressure_score += 0.1
        if breakout_authenticity_score < 0.3:
            countertrend_pressure_score += 0.08
        if trend_exhaustion_risk == "high":
            countertrend_pressure_score += 0.1
        elif continuation_exhaustion_risk in {"medium", "high"}:
            countertrend_pressure_score += 0.07
        if recent_breakout_followthrough_score < 0.1 and breakout_retest_quality == "none":
            countertrend_pressure_score += 0.06
    elif structure_semantic_bias == "LONG" or continuation_bias == "bullish_continuation_candidate":
        countertrend_pressure_bias = "SHORT"
        if short_horizon_bias == "bearish_pullback_candidate":
            countertrend_pressure_score += 0.24 + min(0.18, short_horizon_score * 0.1)
        if location_state == "near_resistance" and breakout_state != "bullish_breakout":
            countertrend_pressure_score += 0.14
        if bearish_candidate_hint_top_conf >= 0.7:
            countertrend_pressure_score += 0.1
        if breakout_authenticity_score < 0.3:
            countertrend_pressure_score += 0.08
        if trend_exhaustion_risk == "high":
            countertrend_pressure_score += 0.1
        elif continuation_exhaustion_risk in {"medium", "high"}:
            countertrend_pressure_score += 0.07
        if recent_breakout_followthrough_score < 0.1 and breakout_retest_quality == "none":
            countertrend_pressure_score += 0.06
    countertrend_pressure_score = round(min(1.0, max(0.0, countertrend_pressure_score)), 4)
    countertrend_pressure_state = _classify_quality_band(
        countertrend_pressure_score,
        high_threshold=0.48,
        medium_threshold=0.28,
    )

    continuation_integrity_score = 0.0
    if continuation_bias != "none":
        continuation_integrity_score += 0.22 + min(0.18, continuation_score * 0.08)
        if trend_continuation_quality == "high":
            continuation_integrity_score += 0.16
        elif trend_continuation_quality == "medium":
            continuation_integrity_score += 0.08
        continuation_integrity_score += min(0.12, float(trend_features.get("channel_stability_score", 0.0) or 0.0) * 0.12)
        continuation_integrity_score += min(0.08, recent_breakout_followthrough_score * 0.12)
        continuation_integrity_score += min(0.08, breakout_authenticity_score * 0.08)
        if continuation_bias == "bullish_continuation_candidate":
            if location_state == "near_resistance" and breakout_state != "bullish_breakout":
                continuation_integrity_score -= 0.1
            if structure_break_state == "bearish_break":
                continuation_integrity_score -= 0.14
            if bearish_candidate_hint_top_conf >= 0.82:
                continuation_integrity_score -= 0.08
        elif continuation_bias == "bearish_continuation_candidate":
            if location_state == "near_support" and breakout_state != "bearish_breakdown":
                continuation_integrity_score -= 0.1
            if structure_break_state == "bullish_break":
                continuation_integrity_score -= 0.14
            if bullish_candidate_hint_top_conf >= 0.82:
                continuation_integrity_score -= 0.08
        if trend_failure_state == "early_failure":
            continuation_integrity_score -= 0.1
        elif trend_failure_state == "probable_failure":
            continuation_integrity_score -= 0.16
        elif trend_failure_state == "confirmed_failure":
            continuation_integrity_score -= 0.22
    continuation_integrity_score = round(min(1.0, max(0.0, continuation_integrity_score)), 4)
    continuation_integrity_state = _classify_quality_band(
        continuation_integrity_score,
        high_threshold=0.54,
        medium_threshold=0.34,
    )
    fragile_structure_handoff = (
        structure_confirmation_tier == "confirmed"
        and structure_semantic_label in {
            "hidden_base_release",
            "hidden_distribution_release",
            "false_breakout_reentry",
            "false_breakdown_reentry",
            "support_reclaim_rotation",
            "resistance_failure_rotation",
            "bullish_structure_break",
            "bearish_structure_break",
        }
        and structure_followthrough_state == "low"
        and countertrend_pressure_state == "high"
    )
    path_semantic_role = "support_only"
    if forecast_horizon_bars >= 3 and stable_three_bar_path and three_bar_path_score >= 0.82:
        path_semantic_role = "sequence_confirmed"
    elif forecast_horizon_bars >= 3 and three_bar_majority_bias in {"LONG", "SHORT"}:
        path_semantic_role = "sequence_context"

    confirmed_structure_bias = "none"
    confirmed_structure_score = 0.0
    confirmed_structure_strength = "none"
    confirmed_structure_reasons: List[str] = []
    if breakout_confirmed and pattern_bias_name == "bullish":
        if breakout_authenticity_score >= 0.46 or breakout_retest_quality == "healthy":
            confirmed_structure_bias = "LONG"
            confirmed_structure_score += 0.22
            confirmed_structure_reasons.append("Confirmed bullish structure should outrank stale bearish pressure")
            if pattern_name in {"v_shaped_reversal", "bullish_flag", "double_bottom"}:
                confirmed_structure_score += 0.08
                confirmed_structure_reasons.append("Bullish reversal/breakout pattern is one of the cleaner QuantAgent-like wins")
    elif breakout_confirmed and pattern_bias_name == "bearish":
        if breakout_authenticity_score >= 0.46 or breakout_retest_quality == "healthy":
            confirmed_structure_bias = "SHORT"
            confirmed_structure_score += 0.22
            confirmed_structure_reasons.append("Confirmed bearish structure should outrank stale bullish pressure")
            if pattern_name in {"inverted_v_reversal", "bearish_flag", "double_top"}:
                confirmed_structure_score += 0.08
                confirmed_structure_reasons.append("Bearish reversal/breakdown pattern is one of the cleaner QuantAgent-like wins")
    if reversal_confirmed and reversal_bias == "bullish_reversal":
        confirmed_structure_bias = "LONG"
        confirmed_structure_score += 0.1
        confirmed_structure_reasons.append("Confirmed bullish reversal adds structure-transition credibility")
    elif reversal_confirmed and reversal_bias == "bearish_reversal":
        confirmed_structure_bias = "SHORT"
        confirmed_structure_score += 0.1
        confirmed_structure_reasons.append("Confirmed bearish reversal adds structure-transition credibility")
    if structure_semantic_bias in {"LONG", "SHORT"} and structure_semantic_score >= 0.62:
        confirmed_structure_bias = structure_semantic_bias
        confirmed_structure_score += min(0.14, 0.06 + structure_semantic_score * 0.1)
        confirmed_structure_reasons.extend(structure_semantic_reasons[:2])
    elif (
        structure_semantic_label in {"support_reclaim_rotation", "resistance_failure_rotation"}
        and structure_semantic_bias in {"LONG", "SHORT"}
        and structure_semantic_score >= 0.56
    ):
        confirmed_structure_bias = structure_semantic_bias
        confirmed_structure_score += min(0.13, 0.055 + structure_semantic_score * 0.095)
        confirmed_structure_reasons.extend(structure_semantic_reasons[:2])
        confirmed_structure_reasons.append("Support/resistance reclaim rotation is strong enough to challenge a weak algorithmic baseline")
        if (
            three_bar_majority_bias == structure_semantic_bias
            and three_bar_path_score >= 0.52
        ) or level_reclaim_state in {
            "bullish_reclaim",
            "bearish_reclaim",
            "failed_bullish_breakout_reentry",
            "failed_bearish_breakdown_reentry",
        }:
            confirmed_structure_score += 0.05
            confirmed_structure_reasons.append("Path alignment or reclaim evidence upgrades the rotation from hint to usable structure")
    if (
        structure_semantic_label in {"false_breakout_reentry", "false_breakdown_reentry"}
        and structure_semantic_bias in {"LONG", "SHORT"}
        and structure_semantic_score >= 0.6
    ):
        confirmed_structure_bias = structure_semantic_bias
        confirmed_structure_score += min(0.15, 0.06 + structure_semantic_score * 0.1)
        confirmed_structure_reasons.extend(structure_semantic_reasons[:2])
        confirmed_structure_reasons.append("False-break reentry is one of the cleaner structure shifts that should challenge stale continuation")
    if (
        structure_semantic_label == "neutral_structure"
        and breakout_confirmed
        and pattern_name in {"double_bottom", "double_top"}
        and pattern_geometry_score >= 0.52
        and breakout_authenticity_score >= 0.14
        and three_bar_majority_bias in {"LONG", "SHORT"}
    ):
        confirmed_structure_bias = "LONG" if pattern_name == "double_bottom" else "SHORT"
        confirmed_structure_score += 0.11
        confirmed_structure_reasons.append("Confirmed double-top/double-bottom geometry is strong enough to challenge a stale base case even before richer semantics appear")
        if three_bar_majority_bias == confirmed_structure_bias and three_bar_path_score >= 0.62:
            confirmed_structure_score += 0.04
            confirmed_structure_reasons.append("Three-bar path aligns with the confirmed classic structure")
    if confirmed_structure_bias == "LONG" and trend_failure_state in {"probable_failure", "confirmed_failure"}:
        confirmed_structure_score += 0.08 if trend_failure_state == "confirmed_failure" else 0.05
        confirmed_structure_reasons.append("Confirmed long structure is reinforced because the stale bearish regime is already failing")
    elif confirmed_structure_bias == "SHORT" and trend_failure_state in {"probable_failure", "confirmed_failure"}:
        confirmed_structure_score += 0.08 if trend_failure_state == "confirmed_failure" else 0.05
        confirmed_structure_reasons.append("Confirmed short structure is reinforced because the stale bullish regime is already failing")
    confirmed_structure_score = round(min(0.42, confirmed_structure_score), 4)
    confirmed_structure_priority_labels = {
        "confirmed_breakout_with_retest",
        "confirmed_breakdown_with_retest",
        "hidden_base_release",
        "hidden_distribution_release",
        "false_breakout_reentry",
        "false_breakdown_reentry",
        "compression_release_breakout",
        "compression_release_breakdown",
        "support_reclaim_rotation",
        "resistance_failure_rotation",
        "bullish_structure_break",
        "bearish_structure_break",
    }
    if confirmed_structure_bias in {"LONG", "SHORT"}:
        if (
            structure_confirmation_tier == "confirmed"
            and not fragile_structure_handoff
            and (
                structure_semantic_label in confirmed_structure_priority_labels
                or breakout_retest_quality == "healthy"
                or breakout_authenticity_score >= 0.5
                or confirmed_structure_score >= 0.26
            )
        ):
            confirmed_structure_strength = "strong"
        elif (
            structure_confirmation_tier == "confirmed"
            or confirmed_structure_score >= 0.18
            or reversal_confirmed
        ):
            confirmed_structure_strength = "weak"

    channel_dominance_bias = "none"
    channel_dominance_score = 0.0
    channel_dominance_reasons: List[str] = []
    if (
        market_regime == "trend"
        and trend_strength >= 0.5
        and trend_alignment_state == "aligned_bullish"
        and macd_hist > 0
    ):
        channel_dominance_bias = "LONG"
        channel_dominance_score = 0.28
        channel_dominance_reasons.append("Long and short trend slopes remain aligned bullish")
        channel_dominance_reasons.append("Positive MACD momentum supports bullish channel dominance")
    elif (
        market_regime == "trend"
        and trend_strength >= 0.5
        and trend_alignment_state == "aligned_bearish"
        and macd_hist < 0
    ):
        channel_dominance_bias = "SHORT"
        channel_dominance_score = 0.28
        channel_dominance_reasons.append("Long and short trend slopes remain aligned bearish")
        channel_dominance_reasons.append("Negative MACD momentum supports bearish channel dominance")

    structure_conflict = False
    structure_conflict_reasons: List[str] = []
    signal_conflict_count = 0
    if (
        reversal_bias == "bullish_reversal"
        and continuation_bias == "bearish_continuation_candidate"
        and reversal_score >= 1.7
        and continuation_score >= 1.8
    ) or (
        reversal_bias == "bearish_reversal"
        and continuation_bias == "bullish_continuation_candidate"
        and reversal_score >= 1.7
        and continuation_score >= 1.8
    ):
        structure_conflict = True
        signal_conflict_count += 1
        structure_conflict_reasons.append("Reversal and continuation signals are both strong but point in opposite directions")
    if (
        reversal_bias == "bullish_reversal"
        and short_horizon_bias == "bearish_pullback_candidate"
        and reversal_score >= 1.7
        and short_horizon_score >= 1.35
    ) or (
        reversal_bias == "bearish_reversal"
        and short_horizon_bias == "bullish_rebound_candidate"
        and reversal_score >= 1.7
        and short_horizon_score >= 1.35
    ):
        structure_conflict = True
        signal_conflict_count += 1
        structure_conflict_reasons.append("Reversal read conflicts with immediate short-horizon reaction")
    if (
        continuation_bias == "bullish_continuation_candidate"
        and short_horizon_bias == "bearish_pullback_candidate"
        and continuation_score >= 1.85
        and short_horizon_score >= 1.35
    ) or (
        continuation_bias == "bearish_continuation_candidate"
        and short_horizon_bias == "bullish_rebound_candidate"
        and continuation_score >= 1.85
        and short_horizon_score >= 1.35
    ):
        structure_conflict = True
        signal_conflict_count += 1
        structure_conflict_reasons.append("Continuation read conflicts with the immediate reaction signal")
    if (
        forecast_horizon_bars >= 3
        and stable_three_bar_path
        and three_bar_majority_bias in {"LONG", "SHORT"}
    ):
        structure_path_conflict = (
            structure_confirmation_tier == "confirmed"
            and structure_semantic_bias in {"LONG", "SHORT"}
            and structure_semantic_bias != three_bar_majority_bias
        )
        if (
            continuation_bias == "bullish_continuation_candidate"
            and three_bar_majority_bias == "SHORT"
            and continuation_score >= 1.9
            and not structure_path_conflict
        ) or (
            continuation_bias == "bearish_continuation_candidate"
            and three_bar_majority_bias == "LONG"
            and continuation_score >= 1.9
            and not structure_path_conflict
        ):
            structure_conflict = True
            signal_conflict_count += 1
            structure_conflict_reasons.append("Stable three-bar path conflicts with the continuation read")

    structural_penalty = 0.0
    structural_reasons: List[str] = []

    if pattern_completed and not breakout_confirmed and pattern_bias_name in {"bullish", "bearish"}:
        structural_penalty += 0.1
        structural_reasons.append("Unconfirmed pattern setup lowers confidence")
    if pattern_geometry_score >= 0.45 and pattern_bias_name in {"bullish", "bearish"}:
        structural_reasons.append("Pattern geometry is relatively clean and visually structured")
    if structure_semantic_label != "neutral_structure" and structure_semantic_score >= 0.45:
        structural_reasons.extend(structure_semantic_reasons[:2])

    if location_state == "near_resistance" and long_score > short_score:
        structural_penalty += 0.08
        structural_reasons.append("Long setup is too close to resistance")
    elif location_state == "near_support" and short_score > long_score:
        structural_penalty += 0.08
        structural_reasons.append("Short setup is too close to support")

    if market_regime in {"compression", "range"} and breakout_state not in {"bullish_breakout", "bearish_breakdown"}:
        structural_penalty += 0.06
        structural_reasons.append("Range/compression environment lowers directional conviction")

    if trend_strength < 0.4:
        structural_penalty += 0.06
        structural_reasons.append("Weak trend structure lowers confidence")

    if trend_continuation_quality == "low":
        structural_penalty += 0.08
        structural_reasons.append("Trend continuation quality is low")
    elif trend_continuation_quality == "medium":
        structural_penalty += 0.03

    if trend_exhaustion_risk == "high":
        structural_penalty += 0.08
        structural_reasons.append("Trend may be exhausting near a key level")
    elif trend_exhaustion_risk == "medium":
        structural_penalty += 0.03
    if continuation_bias != "none" and continuation_score >= 1.85:
        structural_penalty = max(0.0, structural_penalty - 0.04)
        structural_reasons.append("Continuation structure offsets part of the exhaustion penalty")
    if trend_failure_state in {"probable_failure", "confirmed_failure"} and continuation_bias != "none":
        structural_penalty += 0.05
        structural_reasons.append("Continuation evidence is discounted because the older trend regime is already failing")

    if long_votes > 0 and short_votes > 0:
        structural_penalty += 0.16
        structural_reasons.append("Indicator, pattern, and trend evidence are conflicting")
    if structure_conflict:
        structural_penalty += min(0.18, 0.08 + 0.04 * signal_conflict_count)
        structural_reasons.extend(structure_conflict_reasons[:2])
        structural_reasons.append("Mixed directional structures are treated as a low-trust setup")

    if (
        pattern_bias_name == "bullish"
        and trend_direction == "downtrend"
        and not breakout_confirmed
    ):
        structural_penalty += 0.12
        structural_reasons.append("Bullish pattern is fighting the prevailing downtrend")
    elif (
        pattern_bias_name == "bearish"
        and trend_direction == "uptrend"
        and not breakout_confirmed
    ):
        structural_penalty += 0.12
        structural_reasons.append("Bearish pattern is fighting the prevailing uptrend")

    if (
        breakout_state == "testing_resistance"
        and long_score > short_score
        and not breakout_confirmed
    ):
        if not (
            continuation_bias == "bullish_continuation_candidate"
            and continuation_score >= 1.85
            and trend_direction == "uptrend"
        ):
            structural_penalty += 0.1
            structural_reasons.append("Long setup is not yet clear while price is still capped by resistance")
        else:
            structural_reasons.append("Resistance pressure is treated as continuation pressure, not a full rejection")
    elif (
        breakout_state == "testing_support"
        and short_score > long_score
        and not breakout_confirmed
    ):
        if not (
            continuation_bias == "bearish_continuation_candidate"
            and continuation_score >= 1.85
            and trend_direction == "downtrend"
        ):
            structural_penalty += 0.1
            structural_reasons.append("Short setup is not yet clear while price is still sitting on support")
        else:
            structural_reasons.append("Support pressure is treated as continuation pressure, not a full rejection")

    if pattern_completed and not breakout_confirmed and pattern_span and pattern_span < 8:
        structural_penalty += 0.08
        structural_reasons.append("Pattern formed too quickly and is treated as a candidate only")

    # A pattern that is still inside the recent range has not actually escaped
    # market consensus yet. In practice this is often where false continuation
    # reads appear, so we cap the score edge before the breakout happens.
    if pattern_completed and not breakout_confirmed and breakout_state == "inside_range":
        if pattern_bias_name == "bullish" and long_score > short_score:
            long_score = min(long_score, short_score + (0.55 if pattern_span and pattern_span < 10 else 0.75))
            structural_reasons.append("Unconfirmed bullish pattern remains range-bound and cannot dominate yet")
        elif pattern_bias_name == "bearish" and short_score > long_score:
            short_score = min(short_score, long_score + (0.55 if pattern_span and pattern_span < 10 else 0.75))
            structural_reasons.append("Unconfirmed bearish pattern remains range-bound and cannot dominate yet")

    # Hard guardrails: near a key level without confirmation, a directional
    # setup should not keep an oversized edge.
    if long_score > short_score and location_state == "near_resistance" and breakout_state != "bullish_breakout":
        if not (
            continuation_bias == "bullish_continuation_candidate"
            and continuation_score >= 1.9
            and trend_features.get("trend_direction") == "uptrend"
            and trend_continuation_quality in {"medium", "high"}
        ):
            long_score = min(long_score, short_score + 0.65)
            structural_reasons.append("Long score is capped below resistance without a clean breakout")
        else:
            long_score = min(long_score, short_score + 0.95)
            structural_reasons.append("Long score cap is relaxed because continuation evidence is stronger than a normal resistance fade")
    elif short_score > long_score and location_state == "near_support" and breakout_state != "bearish_breakdown":
        if not (
            continuation_bias == "bearish_continuation_candidate"
            and continuation_score >= 1.9
            and trend_features.get("trend_direction") == "downtrend"
            and trend_continuation_quality in {"medium", "high"}
        ):
            short_score = min(short_score, long_score + 0.65)
            structural_reasons.append("Short score is capped above support without a clean breakdown")
        else:
            short_score = min(short_score, long_score + 0.95)
            structural_reasons.append("Short score cap is relaxed because continuation evidence is stronger than a normal support bounce")

    if (
        pattern_bias_name == "bullish"
        and not breakout_confirmed
        and trend_direction == "downtrend"
    ):
        long_score = max(0.0, long_score - 0.2)
        structural_reasons.append("Unconfirmed bullish pattern is downgraded inside a downtrend")
    elif (
        pattern_bias_name == "bearish"
        and not breakout_confirmed
        and trend_direction == "uptrend"
    ):
        short_score = max(0.0, short_score - 0.2)
        structural_reasons.append("Unconfirmed bearish pattern is downgraded inside an uptrend")

    # A confirmed breakout is still not enough if the surrounding structure is
    # poor. This targets cases where a nominal breakout appears, but the trend
    # environment and location make follow-through unreliable.
    if breakout_confirmed and pattern_bias_name == "bullish":
        if breakout_margin_pct >= 0.18:
            long_score += 0.08
            structural_reasons.append("Bullish breakout has measurable geometric extension beyond resistance")
        if breakout_authenticity_score >= 0.42:
            long_score += 0.1
            structural_reasons.append("Bullish breakout authenticity is strong")
        if market_regime != "trend" or trend_continuation_quality == "low":
            long_score = max(0.0, long_score - 0.22)
            structural_reasons.append("Bullish breakout is downgraded because follow-through quality is weak")
        if trend_direction == "downtrend":
            long_score = max(0.0, long_score - 0.18)
            structural_reasons.append("Bullish breakout still conflicts with a broader downtrend structure")
        if location_state == "near_resistance" and breakout_state != "bullish_breakout":
            long_score = max(0.0, long_score - 0.18)
            structural_reasons.append("Bullish breakout is downgraded because price is still boxed by resistance")
        if breakout_state != "bullish_breakout":
            long_score = max(0.0, long_score - 0.22)
            structural_reasons.append("Bullish breakout is downgraded because the current bar is no longer extending the breakout")
        if indicator_bias["indicator_short_score"] >= indicator_bias["indicator_long_score"] + 0.2:
            long_score = max(0.0, long_score - 0.2)
            structural_reasons.append("Bullish breakout is downgraded because indicator momentum leans against it")
        if confirmed_structure_bias == "LONG" and confirmed_structure_score >= 0.24:
            long_score += 0.14
            short_score = max(0.0, short_score - 0.08)
            structural_reasons.extend(confirmed_structure_reasons[:2])
        if trend_failure_state in {"probable_failure", "confirmed_failure"} and trend_direction == "downtrend":
            long_score += 0.08
            short_score = max(0.0, short_score - 0.06)
            structural_reasons.extend(trend_failure_reasons[:2])
    elif breakout_confirmed and pattern_bias_name == "bearish":
        if breakout_margin_pct >= 0.18:
            short_score += 0.08
            structural_reasons.append("Bearish breakdown has measurable geometric extension beyond support")
        if breakout_authenticity_score >= 0.42:
            short_score += 0.1
            structural_reasons.append("Bearish breakdown authenticity is strong")
        if market_regime != "trend" or trend_continuation_quality == "low":
            short_score = max(0.0, short_score - 0.22)
            structural_reasons.append("Bearish breakdown is downgraded because follow-through quality is weak")
        if trend_direction == "uptrend":
            short_score = max(0.0, short_score - 0.18)
            structural_reasons.append("Bearish breakdown still conflicts with a broader uptrend structure")
        if location_state == "near_support" and breakout_state != "bearish_breakdown":
            short_score = max(0.0, short_score - 0.18)
            structural_reasons.append("Bearish breakdown is downgraded because price is still sitting on support")
        if breakout_state != "bearish_breakdown":
            short_score = max(0.0, short_score - 0.22)
            structural_reasons.append("Bearish breakdown is downgraded because the current bar is no longer extending the breakdown")
        if indicator_bias["indicator_long_score"] >= indicator_bias["indicator_short_score"] + 0.2:
            short_score = max(0.0, short_score - 0.2)
            structural_reasons.append("Bearish breakdown is downgraded because indicator momentum leans against it")
        if confirmed_structure_bias == "SHORT" and confirmed_structure_score >= 0.24:
            short_score += 0.14
            long_score = max(0.0, long_score - 0.08)
            structural_reasons.extend(confirmed_structure_reasons[:2])
        if trend_failure_state in {"probable_failure", "confirmed_failure"} and trend_direction == "uptrend":
            short_score += 0.08
            long_score = max(0.0, long_score - 0.06)
            structural_reasons.extend(trend_failure_reasons[:2])

    if reversal_bias == "bullish_reversal":
        reversal_bonus = 0.22 if reversal_confirmed else 0.12
        long_score += reversal_bonus
        structural_reasons.extend(reversal_signal["reversal_reasons"][:2])
        if trend_direction == "downtrend" and short_score > long_score:
            short_score = max(0.0, short_score - (0.18 if reversal_confirmed else 0.1))
            structural_reasons.append("Bearish continuation is discounted because reversal evidence is stacking up")
    elif reversal_bias == "bearish_reversal":
        reversal_bonus = 0.22 if reversal_confirmed else 0.12
        short_score += reversal_bonus
        structural_reasons.extend(reversal_signal["reversal_reasons"][:2])
        if trend_direction == "uptrend" and long_score > short_score:
            long_score = max(0.0, long_score - (0.18 if reversal_confirmed else 0.1))
            structural_reasons.append("Bullish continuation is discounted because reversal evidence is stacking up")

    if continuation_bias == "bullish_continuation_candidate":
        fragile_bullish_continuation = (
            trend_strength < 0.5
            or (
                breakout_authenticity_score < 0.46
                and trend_exhaustion_risk == "high"
                and location_state == "near_resistance"
                and breakout_state != "bullish_breakout"
            )
        )
        continuation_bonus = 0.1 if fragile_bullish_continuation else (0.24 if continuation_score >= 2.0 else 0.16)
        long_score += continuation_bonus
        structural_reasons.extend(continuation_signal["continuation_reasons"][:2])
        if (
            reversal_bias == "bearish_reversal"
            and not reversal_confirmed
            and continuation_score >= reversal_score + 0.15
        ):
            short_score = max(0.0, short_score - (0.06 if fragile_bullish_continuation else 0.14))
            structural_reasons.append("Bearish reversal read is discounted because bullish continuation structure is stronger")
    elif continuation_bias == "bearish_continuation_candidate":
        fragile_bearish_continuation = (
            trend_strength < 0.5
            or (
                breakout_authenticity_score < 0.46
                and trend_exhaustion_risk == "high"
                and location_state == "near_support"
                and breakout_state != "bearish_breakdown"
            )
        )
        continuation_bonus = 0.1 if fragile_bearish_continuation else (0.24 if continuation_score >= 2.0 else 0.16)
        short_score += continuation_bonus
        structural_reasons.extend(continuation_signal["continuation_reasons"][:2])
        if (
            reversal_bias == "bullish_reversal"
            and not reversal_confirmed
            and continuation_score >= reversal_score + 0.15
        ):
            long_score = max(0.0, long_score - (0.06 if fragile_bearish_continuation else 0.14))
            structural_reasons.append("Bullish reversal read is discounted because bearish continuation structure is stronger")

    if short_horizon_bias == "bullish_rebound_candidate":
        long_score += 0.16
        structural_reasons.extend(short_horizon_signal["short_horizon_reasons"][:2])
        if location_state == "near_support" and breakout_state != "bearish_breakdown":
            short_score = max(0.0, short_score - 0.08)
            structural_reasons.append("Immediate short edge is discounted because support-bounce conditions are present")
    elif short_horizon_bias == "bearish_pullback_candidate":
        short_score += 0.16
        structural_reasons.extend(short_horizon_signal["short_horizon_reasons"][:2])
        if location_state == "near_resistance" and breakout_state != "bullish_breakout":
            long_score = max(0.0, long_score - 0.08)
            structural_reasons.append("Immediate long edge is discounted because resistance-rejection conditions are present")

    path_is_blocked_by_confirmed_structure = (
        structure_confirmation_tier == "confirmed"
        and structure_semantic_bias in {"LONG", "SHORT"}
        and three_bar_majority_bias in {"LONG", "SHORT"}
        and three_bar_majority_bias != structure_semantic_bias
        and (
            trend_failure_state in {"probable_failure", "confirmed_failure"}
            or confirmed_structure_score >= 0.24
        )
    )
    if (
        forecast_horizon_bars >= 3
        and stable_three_bar_path
        and three_bar_majority_bias == "LONG"
        and not path_is_blocked_by_confirmed_structure
    ):
        path_bonus = min(0.18, 0.06 + three_bar_path_score * 0.05)
        long_score += path_bonus
        if short_score > long_score:
            short_score = max(0.0, short_score - 0.06)
        structural_reasons.extend(three_bar_path_signal["three_bar_path_reasons"][:2])
        structural_reasons.append("Three-bar path majority leans LONG")
    elif (
        forecast_horizon_bars >= 3
        and stable_three_bar_path
        and three_bar_majority_bias == "SHORT"
        and not path_is_blocked_by_confirmed_structure
    ):
        path_bonus = min(0.18, 0.06 + three_bar_path_score * 0.05)
        short_score += path_bonus
        if long_score > short_score:
            long_score = max(0.0, long_score - 0.06)
        structural_reasons.extend(three_bar_path_signal["three_bar_path_reasons"][:2])
        structural_reasons.append("Three-bar path majority leans SHORT")
    elif forecast_horizon_bars >= 3 and path_is_blocked_by_confirmed_structure:
        structural_reasons.append(
            "Confirmed structure outranks the conflicting three-bar path, so path evidence is treated as execution context only"
        )
    elif forecast_horizon_bars >= 3:
        structural_penalty += 0.03
        structural_reasons.append("Three-bar path is not fully consistent, so it is used as context rather than a directional driver")
    if channel_direction == "converging" and not breakout_confirmed:
        structural_penalty += 0.04
        structural_reasons.append("Converging channel suggests compression, so breakouts need stronger confirmation")

    if (
        channel_dominance_bias == "SHORT"
        and trend_direction == "downtrend"
        and not reversal_confirmed
        and not breakout_confirmed
        and (
            pattern_name in {"v_shaped_reversal", "support_bounce"}
            or short_horizon_bias == "bullish_rebound_candidate"
        )
    ):
        long_score = max(0.0, long_score - 0.14)
        structural_reasons.extend(channel_dominance_reasons[:2])
        structural_reasons.append("Local bullish rebound is treated as a counter-trend bounce inside a dominant bearish channel")
    elif (
        channel_dominance_bias == "LONG"
        and trend_direction == "uptrend"
        and not reversal_confirmed
        and not breakout_confirmed
        and (
            pattern_name in {"inverted_v_reversal", "resistance_rejection"}
            or short_horizon_bias == "bearish_pullback_candidate"
        )
    ):
        short_score = max(0.0, short_score - 0.14)
        structural_reasons.extend(channel_dominance_reasons[:2])
        structural_reasons.append("Local bearish pullback is treated as a counter-trend pause inside a dominant bullish channel")
    if (
        structure_semantic_label == "trend_channel_continuation"
        and structure_semantic_bias in {"LONG", "SHORT"}
        and three_bar_majority_bias in {"LONG", "SHORT"}
        and three_bar_majority_bias != structure_semantic_bias
        and three_bar_path_score >= 0.62
    ):
        if structure_semantic_bias == "LONG":
            long_score = max(0.0, long_score - 0.12)
        else:
            short_score = max(0.0, short_score - 0.12)
        structural_reasons.append("Trend-channel continuation is downgraded because the three-bar path is already leaning the other way")
    if (
        structure_semantic_label == "trend_channel_continuation"
        and structure_semantic_bias == "LONG"
        and confirmed_bearish_candidates
        and three_bar_majority_bias == "SHORT"
    ):
        long_score = max(0.0, long_score - 0.14)
        structural_reasons.append("Bullish trend-channel continuation is downgraded because confirmed bearish candidate structure already exists against it")
    elif (
        structure_semantic_label == "trend_channel_continuation"
        and structure_semantic_bias == "SHORT"
        and confirmed_bullish_candidates
        and three_bar_majority_bias == "LONG"
    ):
        short_score = max(0.0, short_score - 0.14)
        structural_reasons.append("Bearish trend-channel continuation is downgraded because confirmed bullish candidate structure already exists against it")

    if (
        structure_semantic_label == "trend_channel_continuation"
        and structure_semantic_bias == "SHORT"
        and broad_bullish_candidate_count >= 1
        and continuation_bias == "none"
        and not breakout_confirmed
        and breakout_authenticity_score < 0.08
        and three_bar_majority_bias == "SHORT"
        and three_bar_path_consistency >= 0.99
        and three_bar_path_score >= 0.9
    ):
        short_score = max(0.0, short_score - 0.18)
        long_score += 0.05
        structural_reasons.append(
            "Bearish trend-channel continuation is downgraded because path-only continuation without breakout confirmation is too fragile when opposite bullish candidate structure still survives"
        )

    if (
        structure_semantic_bias == "SHORT"
        and structure_semantic_label in {
            "developing_double_top_pressure",
            "bearish_structure_break",
            "hidden_distribution_release",
            "resistance_failure_rotation",
            "false_breakout_reentry",
        }
        and not breakout_confirmed
        and trend_failure_state == "intact"
        and three_bar_path_score <= 0.62
        and breakout_authenticity_score < 0.22
    ):
        short_score = max(0.0, short_score - 0.18)
        structural_reasons.append(
            "Bearish structure is downgraded because authenticity and follow-through remain too weak to justify a decisive short read"
        )
        if len(bullish_candidate_hints) >= len(bearish_candidate_hints) and bullish_candidate_hints:
            short_score = max(0.0, short_score - 0.08)
            long_score += 0.05
            structural_reasons.append(
                "Opposite bullish candidate structure is still alive, so the weak bearish read is treated as a fragile ceiling rather than a decisive breakdown"
            )

    if (
        structure_semantic_bias == "SHORT"
        and structure_semantic_label in {
            "developing_double_top_pressure",
            "bearish_structure_break",
            "hidden_distribution_release",
            "resistance_failure_rotation",
            "false_breakout_reentry",
        }
        and ambiguous_high_conf_candidate_conflict
        and broad_bearish_candidate_count <= broad_bullish_candidate_count
        and breakout_state != "bearish_breakdown"
        and not breakout_confirmed
        and breakout_authenticity_score < 0.3
    ):
        short_score = max(0.0, short_score - 0.2)
        long_score += 0.06
        structural_reasons.append(
            "Bearish structure is heavily downgraded because bullish and bearish candidate structures are both strong while downside breakout confirmation is still missing"
        )
        if short_horizon_bias == "bullish_rebound_candidate":
            short_score = max(0.0, short_score - 0.06)
            long_score += 0.04
            structural_reasons.append(
                "Bullish rebound pressure is still present, so the ambiguous bearish setup is treated as a likely fake handoff"
            )

    # Late-trend reversals often start with momentum weakening before the price
    # structure fully breaks. If momentum is already disagreeing with a
    # direction that still lacks breakout confirmation, reduce that side.
    long_momentum_headwind = (
        macd_hist < 0
        or stoch_state in {"bearish", "overbought"}
        or rsi_state == "overbought"
    )
    short_momentum_headwind = (
        macd_hist > 0
        or stoch_state in {"bullish", "oversold"}
        or rsi_state == "oversold"
    )

    if (
        market_regime == "trend"
        and trend_strength >= 0.45
        and long_score > short_score
        and trend_features.get("trend_direction") == "uptrend"
        and not breakout_confirmed
        and long_momentum_headwind
    ):
        downgrade = 0.06 if continuation_bias == "bullish_continuation_candidate" and continuation_score >= 1.85 else 0.12
        long_score = max(0.0, long_score - downgrade)
        structural_reasons.append("Long direction is downgraded because momentum is no longer confirming it")
    elif (
        market_regime == "trend"
        and trend_strength >= 0.45
        and short_score > long_score
        and trend_features.get("trend_direction") == "downtrend"
        and not breakout_confirmed
        and short_momentum_headwind
    ):
        downgrade = 0.06 if continuation_bias == "bearish_continuation_candidate" and continuation_score >= 1.85 else 0.12
        short_score = max(0.0, short_score - downgrade)
        structural_reasons.append("Short direction is downgraded because momentum is no longer confirming it")

    # A mature trend that is not breaking out should not keep full directional
    # weight when continuation quality is mediocre or poor.
    if (
        market_regime == "trend"
        and not breakout_confirmed
        and breakout_state in {"inside_range", "testing_resistance", "testing_support"}
    ):
        if long_score > short_score and trend_features.get("trend_direction") == "uptrend":
            if trend_continuation_quality == "low":
                long_score = max(0.0, long_score - 0.12)
                structural_reasons.append("Uptrend direction is reduced because continuation quality is low")
            elif trend_continuation_quality == "medium":
                long_score = max(0.0, long_score - 0.06)
                structural_reasons.append("Uptrend direction is slightly reduced until continuation improves")
        elif short_score > long_score and trend_features.get("trend_direction") == "downtrend":
            if trend_continuation_quality == "low":
                short_score = max(0.0, short_score - 0.12)
                structural_reasons.append("Downtrend direction is reduced because continuation quality is low")
            elif trend_continuation_quality == "medium":
                short_score = max(0.0, short_score - 0.06)
                structural_reasons.append("Downtrend direction is slightly reduced until continuation improves")

    early_structure_labels = {
        "developing_double_bottom_pressure",
        "developing_double_top_pressure",
        "developing_bullish_flag_pressure",
        "developing_bearish_flag_pressure",
        "support_reclaim_rotation",
        "resistance_failure_rotation",
        "emerging_upside_rotation",
        "emerging_downside_rotation",
    }

    if structure_semantic_bias == "LONG" and structure_semantic_score >= 0.5:
        semantic_bonus = (
            min(0.08, 0.02 + max(0.0, structure_semantic_score - 0.5) * 0.12)
            if structure_semantic_label in early_structure_labels
            else min(0.26, 0.06 + structure_semantic_score * 0.22)
        )
        long_score += semantic_bonus
        if structure_semantic_label in {
            "false_breakdown_reentry",
            "confirmed_breakout_with_retest",
            "compression_release_breakout",
            "hidden_base_release",
            "bullish_structure_break",
        }:
            short_score = max(0.0, short_score - min(0.12, semantic_bonus * 0.45))
        structural_reasons.append(f"Structure semantic layer favors LONG via {structure_semantic_label}")
    elif structure_semantic_bias == "SHORT" and structure_semantic_score >= 0.5:
        semantic_bonus = (
            min(0.08, 0.02 + max(0.0, structure_semantic_score - 0.5) * 0.12)
            if structure_semantic_label in early_structure_labels
            else min(0.26, 0.06 + structure_semantic_score * 0.22)
        )
        short_score += semantic_bonus
        if structure_semantic_label in {
            "false_breakout_reentry",
            "confirmed_breakdown_with_retest",
            "compression_release_breakdown",
            "hidden_distribution_release",
            "bearish_structure_break",
        }:
            long_score = max(0.0, long_score - min(0.12, semantic_bonus * 0.45))
        structural_reasons.append(f"Structure semantic layer favors SHORT via {structure_semantic_label}")

    if structure_semantic_label in early_structure_labels:
        structural_reasons.append("Early structure is treated as a confidence-supporting clue, not a decisive directional signal")
    if trend_failure_state in {"probable_failure", "confirmed_failure"}:
        structural_reasons.extend(trend_failure_reasons[:2])
        structural_reasons.append("Old trend failure is treated as stronger evidence than stale continuation inertia")

    score_gap = abs(long_score - short_score)
    total_score = long_score + short_score
    dominance_ratio = 0.0 if total_score <= 1e-8 else score_gap / total_score
    consensus_level = "weak"
    if dominance_ratio >= 0.4 and score_gap >= 1.1:
        consensus_level = "strong"
    elif dominance_ratio >= 0.2 and score_gap >= 0.55:
        consensus_level = "moderate"

    weak_environment = (
        market_regime in {"range", "compression"}
        and not breakout_confirmed
        and breakout_state in {"inside_range", "testing_support", "testing_resistance"}
    )
    evidence_conflict = long_votes > 0 and short_votes > 0
    gate_reasons: List[str] = []
    should_abstain = False
    if total_score < 0.45:
        should_abstain = True
        gate_reasons.append("Total evidence is too small")
    elif weak_environment and evidence_conflict and score_gap < 0.12:
        should_abstain = True
        gate_reasons.append("Weak range/compression setup has conflicting evidence")
    elif weak_environment and score_gap < 0.06 and total_score < 1.0:
        should_abstain = True
        gate_reasons.append("Weak range/compression setup does not have enough directional edge")
    elif evidence_conflict and dominance_ratio < 0.06 and total_score < 0.9:
        should_abstain = True
        gate_reasons.append("Conflicting evidence leaves no reliable directional advantage")

    raw_dominant_side = "NEUTRAL"
    dominant_side = "NEUTRAL"
    if long_score > short_score:
        raw_dominant_side = "LONG"
        dominant_side = "LONG"
    elif short_score > long_score:
        raw_dominant_side = "SHORT"
        dominant_side = "SHORT"

    if should_abstain:
        dominant_side = "NEUTRAL"
        structural_reasons.extend(gate_reasons)
        structural_reasons.append("Signal quality is too weak or too conflicted, so the algorithm abstains")

    evidence_scale = min(1.0, total_score / 3.8)
    confirmation_multiplier = 1.0 if breakout_confirmed else 0.84 if pattern_completed else 0.9
    risk_penalty = min(0.95, risk_context["risk_penalty"] + structural_penalty)
    adjusted_confidence = max(
        0.0,
        min(
            1.0,
            dominance_ratio
            * evidence_scale
            * confirmation_multiplier
            * (1.0 - risk_penalty),
        ),
    )
    if should_abstain:
        adjusted_confidence = min(adjusted_confidence, 0.08)
        consensus_level = "watchlist"

    location_is_poor = (
        raw_dominant_side == "LONG"
        and location_state == "near_resistance"
        and breakout_state != "bullish_breakout"
    ) or (
        raw_dominant_side == "SHORT"
        and location_state == "near_support"
        and breakout_state != "bearish_breakdown"
    )
    low_confirmation = pattern_completed and not breakout_confirmed
    fragile_breakout = risk_features.get("breakout_quality") == "fragile"
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))
    high_false_breakout_risk = false_breakout_risk == "high"
    low_trend_quality = trend_continuation_quality == "low"

    hard_case_score = 0.0
    intervention_reasons: List[str] = []
    if should_abstain:
        hard_case_score += 0.32
        intervention_reasons.append("algorithm abstained because the directional edge is too weak")
    if evidence_conflict:
        hard_case_score += 0.18
        intervention_reasons.append("indicator, pattern, and trend evidence are conflicting")
    if weak_environment:
        hard_case_score += 0.16
        intervention_reasons.append("market is still in range/compression without a clear breakout")
    if location_is_poor:
        hard_case_score += 0.12
        intervention_reasons.append("current location is poor for the dominant side")
    if low_confirmation:
        hard_case_score += 0.1
        intervention_reasons.append("pattern exists but breakout confirmation is still missing")
    if fragile_breakout or high_false_breakout_risk:
        hard_case_score += 0.14
        intervention_reasons.append("breakout follow-through risk is fragile")
    if low_trend_quality:
        hard_case_score += 0.08
        intervention_reasons.append("trend continuation quality is weak")
    if reversal_bias != "none":
        hard_case_score += 0.1
        intervention_reasons.append("price structure may be rotating into a reversal regime")
    if reversal_confirmed:
        hard_case_score += 0.08
        intervention_reasons.append("reversal evidence is strong enough that AI may need more authority")
    if short_horizon_bias != "none" and short_horizon_score >= 1.45:
        hard_case_score += 0.08
        intervention_reasons.append("short-horizon rebound/pullback evidence conflicts with the slower structural read")
    if forecast_horizon_bars >= 3 and three_bar_path_consistency < 0.99:
        hard_case_score += 0.08
        intervention_reasons.append("three-bar path is not fully consistent, so AI must reason about sequence rather than a single direction")
    if (
        forecast_horizon_bars >= 3
        and stable_three_bar_path
        and three_bar_majority_bias in {"LONG", "SHORT"}
        and raw_dominant_side in {"LONG", "SHORT"}
        and three_bar_majority_bias != raw_dominant_side
    ):
        hard_case_score += 0.1
        intervention_reasons.append("three-bar path majority conflicts with the raw algorithmic side")
    if continuation_bias != "none" and continuation_score >= 1.95 and location_is_poor:
        hard_case_score += 0.08
        intervention_reasons.append("strong continuation evidence challenges the usual key-level fade logic")
    if structure_conflict:
        hard_case_score += min(0.12, 0.04 * signal_conflict_count)
        intervention_reasons.append("mixed structure conflict means AI should prefer calibration over aggressive flipping")
    if structure_semantic_label in {
        "compression_release_breakout",
        "compression_release_breakdown",
        "hidden_base_release",
        "hidden_distribution_release",
        "bullish_structure_break",
        "bearish_structure_break",
    }:
        hard_case_score += 0.06
        intervention_reasons.append("structural geometry suggests a real regime handoff rather than a normal calibration case")
    elif structure_semantic_label in early_structure_labels:
        hard_case_score += 0.03
        intervention_reasons.append("an early structure exists, but it should mainly calibrate confidence until confirmation improves")
    if consensus_level == "strong" and adjusted_confidence >= 0.16 and not evidence_conflict and not weak_environment:
        hard_case_score -= 0.12

    hard_case_score = round(min(1.0, max(0.0, hard_case_score)), 4)
    hard_case = hard_case_score >= 0.3

    decision_authority_regime = "balanced_calibration"
    authority_owner = "shared"
    authority_reasons: List[str] = []
    if (
        (
            breakout_confirmed
            and pattern_bias_name in {"bullish", "bearish"}
            and pattern_geometry_score >= 0.5
            and breakout_authenticity_score >= 0.46
            and trend_failure_state in {"probable_failure", "confirmed_failure"}
        )
        or (
            structure_semantic_bias in {"LONG", "SHORT"}
            and structure_semantic_score >= 0.62
            and structure_semantic_label
            in {
                "confirmed_breakout_with_retest",
                "confirmed_breakdown_with_retest",
                "compression_release_breakout",
                "compression_release_breakdown",
                "hidden_base_release",
                "hidden_distribution_release",
                "bullish_structure_break",
                "bearish_structure_break",
                "false_breakout_reentry",
                "false_breakdown_reentry",
            }
            and trend_failure_state in {"probable_failure", "confirmed_failure"}
        )
    ):
        decision_authority_regime = "confirmed_structure_priority"
        authority_owner = "ai_structure"
        authority_reasons.append("Confirmed geometric structure should outrank stale inertia because the older regime is already failing")
    elif (
        confirmed_structure_strength == "strong"
        and confirmed_structure_bias in {"LONG", "SHORT"}
        and structure_confirmation_tier == "confirmed"
        and structure_semantic_label
        in {
            "confirmed_breakout_with_retest",
            "confirmed_breakdown_with_retest",
            "compression_release_breakout",
            "compression_release_breakdown",
            "hidden_base_release",
            "hidden_distribution_release",
            "false_breakout_reentry",
            "false_breakdown_reentry",
            "support_reclaim_rotation",
            "resistance_failure_rotation",
            "bullish_structure_break",
            "bearish_structure_break",
        }
        and trend_failure_state in {"early_failure", "intact"}
    ):
        decision_authority_regime = "confirmed_structure_but_trend_failure_incomplete"
        authority_owner = "ai_structure"
        authority_reasons.append("Confirmed structure is already strong enough to contest the stale base case even though old-trend failure is not fully mature yet")
    elif (
        channel_dominance_bias in {"LONG", "SHORT"}
        and channel_dominance_score >= 0.26
        and not breakout_confirmed
        and not reversal_confirmed
        and continuation_integrity_state != "low"
    ):
        decision_authority_regime = "trend_inertia_priority"
        authority_owner = "algorithm"
        authority_reasons.append("Dominant channel structure should outweigh premature counter-trend flips")
    elif (
        structure_semantic_label == "trend_channel_continuation"
        and continuation_integrity_state == "low"
        and trend_failure_state in {"early_failure", "probable_failure", "confirmed_failure"}
    ):
        decision_authority_regime = "structure_present_execution_uncertain"
        authority_owner = "shared"
        authority_reasons.append("Trend continuation remains visible, but its integrity is degrading, so stale channel inertia should not own direction outright")
    elif forecast_horizon_bars <= 2 and short_horizon_bias != "none" and short_horizon_score >= 1.45:
        decision_authority_regime = "micro_reaction_priority"
        authority_owner = "ai_calibrator"
        authority_reasons.append("Ultra-short horizon gives extra value to immediate price reaction evidence")
    elif (
        should_abstain
        and structure_confirmation_tier == "none"
        and structure_semantic_label == "neutral_structure"
    ):
        decision_authority_regime = "hard_skip_uncertainty"
        authority_owner = "risk_control"
        authority_reasons.append("No trustworthy structure is present, so selective abstention should dominate this sample")
    elif (
        structure_conflict
        or (weak_environment and not breakout_confirmed)
        or (
            should_abstain
            and structure_confirmation_tier in {"candidate", "developing"}
        )
    ):
        decision_authority_regime = "structure_present_execution_uncertain"
        authority_owner = "shared"
        authority_reasons.append("Some structure exists, but execution quality is uncertain, so caution should dominate over hard abstention")
    else:
        authority_reasons.append("No single evidence family is dominant enough to own the decision outright")

    if structure_semantic_label in early_structure_labels:
        authority_reasons.append("Early structure is recorded for calibration, but it is not allowed to own direction before confirmation")

    simple_case = (
        not hard_case
        and dominant_side in {"LONG", "SHORT"}
        and consensus_level == "strong"
        and adjusted_confidence >= 0.16
        and dominance_ratio >= 0.2
        and not weak_environment
        and not evidence_conflict
        and not location_is_poor
        and not high_false_breakout_risk
        and (
            channel_dominance_score >= 0.26
            or structure_confirmation_tier == "confirmed"
        )
        and trend_failure_state == "intact"
    )

    if should_abstain or hard_case_score >= 0.55:
        execution_grade = "D"
    elif hard_case_score >= 0.4 or adjusted_confidence < 0.08:
        execution_grade = "C"
    elif hard_case_score >= 0.22 or adjusted_confidence < 0.14:
        execution_grade = "B"
    else:
        execution_grade = "A"

    if simple_case:
        ai_intervention_reason = "simple_case_strong_algorithmic_edge"
        ai_route_hint = "algorithm_only"
    elif reversal_confirmed and hard_case_score >= 0.3:
        ai_intervention_reason = "confirmed_structure_transition_needs_stronger_ai_review"
        ai_route_hint = "ai_override_candidate"
    elif should_abstain:
        ai_intervention_reason = "algorithm_abstained_needs_tie_break_or_skip"
        ai_route_hint = "ai_review_required"
    elif hard_case_score >= 0.45:
        ai_intervention_reason = "hard_case_requires_ai_risk_calibration"
        ai_route_hint = "ai_review_required"
    elif hard_case:
        ai_intervention_reason = "borderline_case_prefers_ai_calibration"
        ai_route_hint = "ai_review_preferred"
    else:
        ai_intervention_reason = "algorithm_edge_is_usable_but_can_still_be_calibrated"
        ai_route_hint = "algorithm_first_ai_optional"

    return {
        "dominant_side": dominant_side,
        "raw_dominant_side": raw_dominant_side,
        "long_score": round(long_score, 4),
        "short_score": round(short_score, 4),
        "score_gap": round(score_gap, 4),
        "dominance_ratio": round(dominance_ratio, 4),
        "long_votes": long_votes,
        "short_votes": short_votes,
        "consensus_level": consensus_level,
        "signal_gate": "abstain" if should_abstain else "directional",
        "gate_reasons": gate_reasons,
        "indicator_long_score": indicator_bias["indicator_long_score"],
        "indicator_short_score": indicator_bias["indicator_short_score"],
        "pattern_long_score": pattern_bias["pattern_long_score"],
        "pattern_short_score": pattern_bias["pattern_short_score"],
        "pattern_family": pattern_bias["pattern_family"],
        "pattern_geometry_score": round(pattern_geometry_score, 4),
        "breakout_authenticity_score": round(breakout_authenticity_score, 4),
        "breakout_body_ratio": round(breakout_body_ratio, 4),
        "breakout_retest_quality": breakout_retest_quality,
        "structure_break_state": structure_break_state,
        "candidate_patterns": candidate_patterns[:5],
        "candidate_pattern_summaries": candidate_pattern_summaries[:5],
        "structure_semantic_label": structure_semantic_label,
        "structure_semantic_bias": structure_semantic_bias,
        "structure_semantic_score": round(structure_semantic_score, 4),
        "structure_semantic_reasons": structure_semantic_reasons[:3],
        "structure_confirmation_tier": structure_confirmation_tier,
        "path_semantic_role": path_semantic_role,
        "trend_long_score": trend_bias["trend_long_score"],
        "trend_short_score": trend_bias["trend_short_score"],
        "risk_level": risk_context["risk_level"],
        "risk_penalty": round(risk_penalty, 4),
        "algorithmic_confidence": round(adjusted_confidence, 4),
        "hard_case_score": hard_case_score,
        "hard_case": hard_case,
        "simple_case": simple_case,
        "ai_intervention_reason": ai_intervention_reason,
        "ai_route_hint": ai_route_hint,
        "execution_grade": execution_grade,
        "location_is_poor": location_is_poor,
        "evidence_conflict": evidence_conflict,
        "weak_environment": weak_environment,
        "reversal_bias": reversal_bias,
        "reversal_score": round(reversal_score, 4),
        "reversal_confirmed": reversal_confirmed,
        "continuation_bias": continuation_bias,
        "continuation_score": round(continuation_score, 4),
        "continuation_exhaustion_risk": continuation_exhaustion_risk,
        "continuation_integrity_score": continuation_integrity_score,
        "continuation_integrity_state": continuation_integrity_state,
        "short_horizon_bias": short_horizon_bias,
        "short_horizon_score": round(short_horizon_score, 4),
        "structure_followthrough_score": structure_followthrough_score,
        "structure_followthrough_state": structure_followthrough_state,
        "countertrend_pressure_bias": countertrend_pressure_bias,
        "countertrend_pressure_score": countertrend_pressure_score,
        "countertrend_pressure_state": countertrend_pressure_state,
        "confirmed_structure_bias": confirmed_structure_bias.upper(),
        "confirmed_structure_score": confirmed_structure_score,
        "confirmed_structure_strength": confirmed_structure_strength,
        "trend_failure_state": trend_failure_state,
        "trend_failure_score": trend_failure_score,
        "trend_failure_reasons": trend_failure_reasons[:3],
        "channel_dominance_bias": channel_dominance_bias.upper(),
        "channel_dominance_score": round(channel_dominance_score, 4),
        "structure_conflict": structure_conflict,
        "signal_conflict_count": signal_conflict_count,
        "decision_authority_regime": decision_authority_regime,
        "authority_owner": authority_owner,
        "authority_reasons": authority_reasons[:3],
        "three_bar_path": three_bar_path_signal["three_bar_path"],
        "three_bar_majority_bias": three_bar_majority_bias,
        "three_bar_path_consistency": round(three_bar_path_consistency, 4),
        "three_bar_path_score": round(three_bar_path_score, 4),
        "three_bar_path_reasons": three_bar_path_signal["three_bar_path_reasons"],
        "trend_alignment_state": str(trend_features.get("trend_alignment_state", "mixed_transition")),
        "channel_direction": channel_direction,
        "channel_stability_score": float(trend_features.get("channel_stability_score", 0.0) or 0.0),
        "trend_continuation_quality": trend_continuation_quality,
        "trend_exhaustion_risk": trend_exhaustion_risk,
        "breakout_margin_pct": round(breakout_margin_pct, 4),
        "forecast_horizon_bars": forecast_horizon_bars,
        "supporting_reasons": (
            indicator_bias["indicator_reasons"]
            + pattern_bias["pattern_reasons"]
            + trend_bias["trend_reasons"]
            + risk_context["risk_reasons"]
            + reversal_signal["reversal_reasons"]
            + intervention_reasons
            + structural_reasons
        )[:8],
    }


def extract_indicator_features(
    ohlc_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Dict[str, object]:
    """Build structured indicator features for the decision layer."""

    config = config or FeatureConfig()
    close = ohlc_df["Close"]
    high = ohlc_df["High"]
    low = ohlc_df["Low"]

    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=config.macd_fast,
        slowperiod=config.macd_slow,
        signalperiod=config.macd_signal,
    )
    rsi = talib.RSI(close, timeperiod=config.rsi_period)
    roc = talib.ROC(close, timeperiod=config.roc_period)
    stoch_k, stoch_d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=config.stoch_period,
        slowk_period=3,
        slowd_period=3,
    )
    willr = talib.WILLR(high, low, close, timeperiod=config.willr_period)
    atr = talib.ATR(high, low, close, timeperiod=config.atr_period)
    adx = talib.ADX(high, low, close, timeperiod=config.adx_period)
    upper_band, middle_band, lower_band = talib.BBANDS(
        close, timeperiod=config.bollinger_period
    )
    ma_fast = talib.SMA(close, timeperiod=config.ma_fast_period)
    ma_slow = talib.SMA(close, timeperiod=config.ma_slow_period)

    current_macd = _safe_last(macd)
    previous_macd = _safe_prev(macd)
    current_signal = _safe_last(macd_signal)
    previous_signal = _safe_prev(macd_signal)
    current_rsi = _safe_last(rsi)
    current_roc = _safe_last(roc)
    current_stoch_k = _safe_last(stoch_k)
    current_stoch_d = _safe_last(stoch_d)
    current_willr = _safe_last(willr)
    current_atr = _safe_last(atr)
    current_adx = _safe_last(adx)
    current_close = float(close.iloc[-1])
    current_upper_band = _safe_last(pd.Series(upper_band))
    current_middle_band = _safe_last(pd.Series(middle_band))
    current_lower_band = _safe_last(pd.Series(lower_band))
    current_ma_fast = _safe_last(pd.Series(ma_fast))
    current_ma_slow = _safe_last(pd.Series(ma_slow))
    previous_ma_fast = _safe_prev(pd.Series(ma_fast))
    previous_ma_slow = _safe_prev(pd.Series(ma_slow))
    rsi_divergence = _detect_simple_divergence(close, pd.Series(rsi))
    last_candle = _compute_candle_geometry(
        open_price=float(ohlc_df["Open"].iloc[-1]),
        high_price=float(high.iloc[-1]),
        low_price=float(low.iloc[-1]),
        close_price=current_close,
    )
    prev_candle = _compute_candle_geometry(
        open_price=float(ohlc_df["Open"].iloc[-2]),
        high_price=float(high.iloc[-2]),
        low_price=float(low.iloc[-2]),
        close_price=float(close.iloc[-2]),
    ) if len(ohlc_df) >= 2 else {"body_ratio": 0.0, "candle_direction": "neutral"}

    atr_pct = None if current_atr is None else float(current_atr / current_close * 100.0)
    band_width_pct = None
    if None not in {current_upper_band, current_lower_band, current_middle_band} and current_middle_band:
        band_width_pct = float((current_upper_band - current_lower_band) / current_middle_band * 100.0)
    price_vs_ma_fast_pct = None if current_ma_fast is None else float((current_close - current_ma_fast) / current_close * 100.0)
    price_vs_ma_slow_pct = None if current_ma_slow is None else float((current_close - current_ma_slow) / current_close * 100.0)

    return {
        "current_close": current_close,
        "price_change_pct": float(((close.iloc[-1] / close.iloc[0]) - 1.0) * 100.0),
        "rsi": current_rsi,
        "rsi_state": _classify_rsi(current_rsi),
        "rsi_divergence": rsi_divergence,
        "macd": current_macd,
        "macd_signal": current_signal,
        "macd_hist": _safe_last(macd_hist),
        "macd_cross": _detect_cross(previous_macd, current_macd, previous_signal, current_signal),
        "roc": current_roc,
        "stoch_k": current_stoch_k,
        "stoch_d": current_stoch_d,
        "stoch_state": _classify_stoch(current_stoch_k, current_stoch_d),
        "stoch_cross": _detect_cross(_safe_prev(stoch_k), current_stoch_k, _safe_prev(stoch_d), current_stoch_d),
        "willr": current_willr,
        "willr_state": _classify_willr(current_willr),
        "atr": current_atr,
        "atr_pct": atr_pct,
        "adx": current_adx,
        "adx_strength": _classify_adx(current_adx),
        "bollinger_upper": current_upper_band,
        "bollinger_middle": current_middle_band,
        "bollinger_lower": current_lower_band,
        "bollinger_bandwidth_pct": band_width_pct,
        "ma_fast": current_ma_fast,
        "ma_slow": current_ma_slow,
        "ma_cross": _detect_cross(previous_ma_fast, current_ma_fast, previous_ma_slow, current_ma_slow),
        "price_vs_ma_fast_pct": price_vs_ma_fast_pct,
        "price_vs_ma_slow_pct": price_vs_ma_slow_pct,
        "volatility_regime": _classify_volatility(atr_pct),
        "last_candle_direction": last_candle["candle_direction"],
        "last_candle_body_ratio": last_candle["body_ratio"],
        "last_candle_upper_wick_ratio": last_candle["upper_wick_ratio"],
        "last_candle_lower_wick_ratio": last_candle["lower_wick_ratio"],
        "last_candle_range_pct": round(float((float(high.iloc[-1]) - float(low.iloc[-1])) / max(current_close, 1e-8) * 100.0), 4),
        "prev_candle_body_ratio": prev_candle.get("body_ratio", 0.0),
        "recent_consecutive_up_closes": _count_consecutive_direction(close, "up"),
        "recent_consecutive_down_closes": _count_consecutive_direction(close, "down"),
    }


def extract_pattern_features(
    ohlc_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Dict[str, object]:
    """
    Detect price structures from pivots.

    Financial note:
    A double bottom is a classic bullish reversal structure. It means price
    tested a low twice and failed to break lower, which often suggests that
    selling pressure is weakening. A double top is the bearish mirror image.
    """

    config = config or FeatureConfig()
    close = ohlc_df["Close"].reset_index(drop=True)
    high = ohlc_df["High"].reset_index(drop=True)
    low = ohlc_df["Low"].reset_index(drop=True)
    pivots = _pivot_points(close, order=config.pivot_order)
    pivot_highs = pivots["pivot_highs"]
    pivot_lows = pivots["pivot_lows"]

    candidates = [
        _detect_double_bottom(
            close,
            pivot_lows,
            pivot_highs,
            config.pattern_tolerance,
            config.neckline_breakout_buffer,
        ),
        _detect_double_top(
            close,
            pivot_highs,
            pivot_lows,
            config.pattern_tolerance,
            config.neckline_breakout_buffer,
        ),
        _detect_triangle(pivot_highs, pivot_lows),
        _detect_v_reversal(close),
        _detect_level_reaction(close, high, low),
        _detect_flag_continuation(close, high, low),
        _detect_hidden_base(close, high, low),
    ]
    candidates = [candidate for candidate in candidates if candidate]
    sorted_candidates = sorted(
        candidates,
        key=lambda item: float(item.get("pattern_confidence", 0.0) or 0.0),
        reverse=True,
    )

    best_candidate = max(
        candidates,
        key=lambda item: item["pattern_confidence"],
        default={
            "pattern": "none",
            "pattern_bias": "neutral",
            "pattern_family": "none",
            "pattern_confidence": 0.0,
            "pattern_completed": False,
            "breakout_confirmed": False,
        },
    )
    geometry_score = 0.0
    if best_candidate.get("pattern") in {"double_bottom", "double_top"}:
        geometry_score += 0.35 * float(best_candidate.get("symmetry_score", 0.0) or 0.0)
        geometry_score += 0.25 if best_candidate.get("breakout_confirmed", False) else 0.0
    if best_candidate.get("pattern") in {"v_shaped_reversal", "inverted_v_reversal"}:
        geometry_score += 0.25 * min(1.0, float(best_candidate.get("recovery_ratio", 0.0) or 0.0))
        geometry_score += 0.2 * float(best_candidate.get("leg_symmetry_score", 0.0) or 0.0)
    if best_candidate.get("pattern") in {"bullish_flag", "bearish_flag"}:
        retrace_pct = float(best_candidate.get("retrace_pct", 0.0) or 0.0)
        geometry_score += 0.2 if retrace_pct <= 1.8 else 0.08
        geometry_score += 0.15 if best_candidate.get("breakout_confirmed", False) else 0.0
    if best_candidate.get("pattern") in {"support_bounce", "resistance_rejection"}:
        geometry_score += 0.18 if best_candidate.get("breakout_confirmed", False) else 0.08
    if best_candidate.get("pattern") in {"hidden_base_breakout", "hidden_distribution_breakdown"}:
        geometry_score += 0.24
        geometry_score += 0.22 * float(best_candidate.get("base_tightness_score", 0.0) or 0.0)
        geometry_score += 0.16 if best_candidate.get("breakout_confirmed", False) else 0.0
    geometry_score = round(min(1.0, geometry_score), 4)
    breakout_auth = _compute_breakout_authenticity(ohlc_df, best_candidate)
    candidate_pattern_summaries: List[Dict[str, object]] = []
    candidate_confidence_map: Dict[str, float] = {}
    candidate_breakout_map: Dict[str, bool] = {}
    candidate_detail_map: Dict[str, Dict[str, object]] = {}
    for candidate in sorted_candidates[:5]:
        pattern_name = str(candidate.get("pattern", "none"))
        candidate_confidence_map[pattern_name] = round(float(candidate.get("pattern_confidence", 0.0) or 0.0), 4)
        candidate_breakout_map[pattern_name] = bool(candidate.get("breakout_confirmed", False))
        candidate_detail_map[pattern_name] = {
            "breakout_margin_pct": round(float(candidate.get("breakout_margin_pct", 0.0) or 0.0), 4),
            "neckline_distance_pct": round(float(candidate.get("neckline_distance_pct", 0.0) or 0.0), 4),
            "reaction_return_pct": round(float(candidate.get("reaction_return_pct", 0.0) or 0.0), 4),
            "retrace_pct": round(float(candidate.get("retrace_pct", 0.0) or 0.0), 4),
            "recovery_ratio": round(float(candidate.get("recovery_ratio", 0.0) or 0.0), 4),
            "base_tightness_score": round(float(candidate.get("base_tightness_score", 0.0) or 0.0), 4),
        }
        candidate_pattern_summaries.append(
            {
                "pattern": pattern_name,
                "bias": str(candidate.get("pattern_bias", "neutral")),
                "confidence": round(float(candidate.get("pattern_confidence", 0.0) or 0.0), 4),
                "breakout_confirmed": bool(candidate.get("breakout_confirmed", False)),
            }
        )

    return {
        **best_candidate,
        "pattern_geometry_score": geometry_score,
        **breakout_auth,
        "candidate_patterns": [item["pattern"] for item in candidate_pattern_summaries],
        "candidate_pattern_summaries": candidate_pattern_summaries,
        "candidate_confidence_map": candidate_confidence_map,
        "candidate_breakout_map": candidate_breakout_map,
        "candidate_detail_map": candidate_detail_map,
        "pivot_high_count": len(pivot_highs),
        "pivot_low_count": len(pivot_lows),
        "recent_pivot_highs": pivot_highs[-4:],
        "recent_pivot_lows": pivot_lows[-4:],
    }


def extract_trend_features(
    ohlc_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Dict[str, object]:
    """
    Summarize trend direction and market regime.

    Financial note:
    In trend analysis, "higher highs + higher lows" usually means buyers remain
    in control. "Lower highs + lower lows" means sellers remain in control.
    This is often more robust than staring only at one indicator.
    """

    config = config or FeatureConfig()
    close = ohlc_df["Close"].reset_index(drop=True)
    high = ohlc_df["High"].reset_index(drop=True)
    low = ohlc_df["Low"].reset_index(drop=True)
    latest_close = float(close.iloc[-1])

    regression = _compute_regression_stats(close)
    medium_regression = _compute_regression_stats(close.tail(min(30, len(close))).reset_index(drop=True))
    short_regression = _compute_regression_stats(close.tail(min(15, len(close))).reset_index(drop=True))
    pivots = _pivot_points(close, order=config.pivot_order)
    pivot_highs = pivots["pivot_highs"]
    pivot_lows = pivots["pivot_lows"]
    structure = _infer_high_low_structure(pivot_highs, pivot_lows)
    swing_geometry = _compute_swing_geometry(pivot_highs, pivot_lows, latest_close)

    support_level = float(low.tail(10).min())
    resistance_level = float(high.tail(10).max())
    channel_width_pct = float((resistance_level - support_level) / latest_close * 100.0)
    range_span = resistance_level - support_level
    location_ratio = None
    if range_span > 0:
        location_ratio = float((latest_close - support_level) / range_span)
        location_ratio = min(1.5, max(-0.5, location_ratio))

    adx = talib.ADX(high, low, close, timeperiod=config.adx_period)
    adx_value = _safe_last(adx)
    support_touch_count = _count_level_touches(low.tail(12), support_level)
    resistance_touch_count = _count_level_touches(high.tail(12), resistance_level)
    upper_line_stats = _compute_line_stats(pivot_highs[-4:])
    lower_line_stats = _compute_line_stats(pivot_lows[-4:])
    support_slope = float(lower_line_stats.get("slope", 0.0) or 0.0)
    resistance_slope = float(upper_line_stats.get("slope", 0.0) or 0.0)
    if support_slope > 0 and resistance_slope > 0:
        channel_direction = "ascending"
    elif support_slope < 0 and resistance_slope < 0:
        channel_direction = "descending"
    elif support_slope > 0 and resistance_slope < 0:
        channel_direction = "converging"
    elif support_slope < 0 and resistance_slope > 0:
        channel_direction = "expanding"
    else:
        channel_direction = "mixed"
    first_half_high = float(high.tail(20).head(min(10, len(high.tail(20)))).max()) if len(high) >= 10 else resistance_level
    first_half_low = float(low.tail(20).head(min(10, len(low.tail(20)))).min()) if len(low) >= 10 else support_level
    recent_half_high = float(high.tail(min(10, len(high))).max())
    recent_half_low = float(low.tail(min(10, len(low))).min())
    older_width = max(first_half_high - first_half_low, 1e-8)
    recent_width = max(recent_half_high - recent_half_low, 1e-8)
    channel_compression_ratio = round(float(recent_width / older_width), 4)
    channel_width_change_pct = round((recent_width / older_width - 1.0) * 100.0, 4)
    
    # Define breakout_state early so it can be used below
    breakout_state = _classify_breakout_state(latest_close, support_level, resistance_level)
    
    breakout_margin_pct = 0.0
    if breakout_state == "bullish_breakout":
        breakout_margin_pct = round(_safe_pct_change(resistance_level, latest_close), 4)
    elif breakout_state == "bearish_breakdown":
        breakout_margin_pct = round(_safe_pct_change(latest_close, support_level), 4)
    price_position_in_channel = round(min(1.5, max(-0.5, location_ratio if location_ratio is not None else 0.5)), 4)
    breakout_microstructure = _compute_breakout_microstructure(
        ohlc_df=ohlc_df.reset_index(drop=True),
        support_level=support_level,
        resistance_level=resistance_level,
    )
    channel_stability_score = round(
        min(
            1.0,
            (
                float(lower_line_stats.get("fit_r2", 0.0) or 0.0)
                + float(upper_line_stats.get("fit_r2", 0.0) or 0.0)
            )
            / 2.0,
        ),
        4,
    )

    if structure == "higher_high_higher_low" and regression["slope"] > 0:
        direction = "uptrend"
    elif structure == "lower_high_lower_low" and regression["slope"] < 0:
        direction = "downtrend"
    elif structure == "compression":
        direction = "compression"
    else:
        direction = "sideways"

    if direction in {"uptrend", "downtrend"} and (adx_value or 0) >= 20:
        market_regime = "trend"
    elif direction == "compression":
        market_regime = "compression"
    else:
        market_regime = "range"

    trend_alignment_state = "mixed_transition"
    if regression["slope"] >= 0 and short_regression["slope"] >= 0:
        trend_alignment_state = "aligned_bullish"
    elif regression["slope"] <= 0 and short_regression["slope"] <= 0:
        trend_alignment_state = "aligned_bearish"
    elif regression["slope"] < 0 < short_regression["slope"]:
        trend_alignment_state = "bullish_short_vs_bearish_long"
    elif regression["slope"] > 0 > short_regression["slope"]:
        trend_alignment_state = "bearish_short_vs_bullish_long"

    location_state = _classify_location_ratio(location_ratio)

    return {
        "trend_direction": direction,
        "market_regime": market_regime,
        "trend_slope": regression["slope"],
        "trend_slope_medium": medium_regression["slope"],
        "trend_slope_short": short_regression["slope"],
        "trend_alignment_state": trend_alignment_state,
        "trend_fit_r2": regression["r_squared"],
        "trend_strength_score": round(
            min(
                1.0,
                (
                    abs(regression["slope"]) / max(latest_close, 1e-8) * len(close) * 140.0
                )
                + regression["r_squared"] * 0.55
                + ((adx_value or 0.0) / 100.0) * 0.22,
            ),
            4,
        ),
        "support_level": support_level,
        "resistance_level": resistance_level,
        "distance_to_support_pct": float((latest_close - support_level) / latest_close * 100.0),
        "distance_to_resistance_pct": float((resistance_level - latest_close) / latest_close * 100.0),
        "channel_width_pct": channel_width_pct,
        "channel_volatility": regression["residual_std"],
        "location_ratio": location_ratio,
        "location_state": location_state,
        "breakout_state": breakout_state,
        "trend_continuation_quality": _classify_trend_continuation_quality(
            direction=direction,
            breakout_state=breakout_state,
            location_state=location_state,
            trend_fit_r2=regression["r_squared"],
            adx_value=adx_value,
        ),
        "trend_exhaustion_risk": _classify_trend_exhaustion_risk(
            direction=direction,
            breakout_state=breakout_state,
            location_state=location_state,
        ),
        "support_touch_count": support_touch_count,
        "resistance_touch_count": resistance_touch_count,
        "support_slope": round(support_slope, 6),
        "resistance_slope": round(resistance_slope, 6),
        "support_line_fit_r2": lower_line_stats.get("fit_r2", 0.0),
        "resistance_line_fit_r2": upper_line_stats.get("fit_r2", 0.0),
        "channel_direction": channel_direction,
        "channel_stability_score": channel_stability_score,
        "channel_compression_ratio": channel_compression_ratio,
        "channel_width_change_pct": channel_width_change_pct,
        "price_position_in_channel": price_position_in_channel,
        "price_to_support_pct": round(float((latest_close - support_level) / max(latest_close, 1e-8) * 100.0), 4),
        "price_to_resistance_pct": round(float((resistance_level - latest_close) / max(latest_close, 1e-8) * 100.0), 4),
        "breakout_margin_pct": breakout_margin_pct,
        "high_low_structure": structure,
        "adx": adx_value,
        "recent_pivot_highs": pivot_highs[-3:],
        "recent_pivot_lows": pivot_lows[-3:],
        **swing_geometry,
        **breakout_microstructure,
    }


def extract_risk_features(
    ohlc_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Dict[str, object]:
    """
    Estimate risk descriptors for the final decision layer.

    Financial note:
    ATR measures absolute price movement range. It is not direction, it is
    "how violently price is moving". High ATR usually means higher uncertainty
    and wider stop-loss / take-profit ranges are needed.
    """

    config = config or FeatureConfig()
    high = ohlc_df["High"]
    low = ohlc_df["Low"]
    close = ohlc_df["Close"]

    atr = talib.ATR(high, low, close, timeperiod=config.atr_period)
    adx = talib.ADX(high, low, close, timeperiod=config.adx_period)
    returns = close.pct_change().dropna()
    latest_close = float(close.iloc[-1])
    atr_value = _safe_last(atr)
    adx_value = _safe_last(adx)
    atr_pct = None if atr_value is None else float(atr_value / latest_close * 100.0)

    upside_range = float(ohlc_df["High"].tail(10).max() - latest_close)
    downside_range = float(latest_close - ohlc_df["Low"].tail(10).min())
    breakout_quality = "healthy"
    false_breakout_risk = "low"
    volatility_regime = _classify_volatility(atr_pct)
    trend_environment = _classify_adx(adx_value)

    if volatility_regime == "high" and trend_environment == "weak":
        breakout_quality = "fragile"
        false_breakout_risk = "high"
    elif volatility_regime == "medium" and trend_environment == "weak":
        false_breakout_risk = "medium"

    return {
        "atr": atr_value,
        "atr_pct": atr_pct,
        "return_volatility": None if returns.empty else float(returns.std()),
        "adx": adx_value,
        "trend_environment": trend_environment,
        "volatility_regime": volatility_regime,
        "breakout_quality": breakout_quality,
        "false_breakout_risk": false_breakout_risk,
        "upside_range": upside_range,
        "downside_range": downside_range,
    }


def extract_market_features(
    ohlc_df: pd.DataFrame,
    config: FeatureConfig | None = None,
    forecast_horizon_bars: int = 1,
) -> Dict[str, object]:
    """Single entry point for the whole algorithm analysis layer."""

    config = config or FeatureConfig()
    indicator_features = extract_indicator_features(ohlc_df, config=config)
    pattern_features = extract_pattern_features(ohlc_df, config=config)
    trend_features = extract_trend_features(ohlc_df, config=config)
    risk_features = extract_risk_features(ohlc_df, config=config)

    return {
        "indicator_features": indicator_features,
        "pattern_features": pattern_features,
        "trend_features": trend_features,
        "risk_features": risk_features,
        "decision_features": extract_decision_features(
            indicator_features,
            pattern_features,
            trend_features,
            risk_features,
            forecast_horizon_bars=forecast_horizon_bars,
        ),
    }
