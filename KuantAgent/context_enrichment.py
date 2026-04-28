from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


_TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


def derive_macro_timeframe(timeframe: str) -> str:
    mapping = {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "30m": "4h",
        "1h": "4h",
        "4h": "1d",
        "1d": "1w",
    }
    return mapping.get(timeframe, timeframe)


def _group_factor(timeframe: str) -> int:
    base_minutes = _TIMEFRAME_TO_MINUTES.get(timeframe)
    macro_minutes = _TIMEFRAME_TO_MINUTES.get(derive_macro_timeframe(timeframe))
    if not base_minutes or not macro_minutes or macro_minutes <= base_minutes:
        return 4
    return max(2, int(round(macro_minutes / base_minutes)))


def build_macro_context(
    normalized_df: pd.DataFrame,
    timeframe: str,
    macro_window_size: int = 16,
) -> Dict[str, Any] | None:
    """Aggregate the current window into a higher-timeframe context without extra data downloads."""

    if normalized_df.empty or len(normalized_df) < 12:
        return None

    df = normalized_df.copy().reset_index(drop=True)
    factor = _group_factor(timeframe)
    if len(df) < factor * 4:
        return None

    df["_macro_group"] = [idx // factor for idx in range(len(df))]
    macro_df = (
        df.groupby("_macro_group", as_index=False)
        .agg(
            {
                "Datetime": "last",
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            }
        )
        .tail(macro_window_size)
        .reset_index(drop=True)
    )
    if len(macro_df) < 4:
        return None

    close_start = float(macro_df.iloc[0]["Close"])
    close_end = float(macro_df.iloc[-1]["Close"])
    support_level = float(macro_df["Low"].tail(min(6, len(macro_df))).min())
    resistance_level = float(macro_df["High"].tail(min(6, len(macro_df))).max())
    slope = close_end - close_start

    if close_end > close_start * 1.01:
        macro_bias = "bullish"
    elif close_end < close_start * 0.99:
        macro_bias = "bearish"
    else:
        macro_bias = "neutral"

    return {
        "macro_timeframe": derive_macro_timeframe(timeframe),
        "macro_bias": macro_bias,
        "macro_slope": round(slope, 4),
        "macro_support": round(support_level, 4),
        "macro_resistance": round(resistance_level, 4),
        "macro_close_change_pct": round((close_end / close_start - 1.0) * 100.0, 4),
        "macro_kline_data": {
            "Datetime": macro_df["Datetime"].astype(str).tolist(),
            "Open": macro_df["Open"].tolist(),
            "High": macro_df["High"].tolist(),
            "Low": macro_df["Low"].tolist(),
            "Close": macro_df["Close"].tolist(),
        },
    }


def build_case_context(
    *,
    asset: str,
    timeframe: str,
    pattern_features: Dict[str, Any],
    trend_features: Dict[str, Any],
    risk_features: Dict[str, Any],
    decision_features: Dict[str, Any],
    macro_timeframe: str | None = None,
    macro_kline_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a lightweight case-memory object for prompt conditioning."""

    market_regime = str(trend_features.get("market_regime", "unknown"))
    trend_direction = str(trend_features.get("trend_direction", "unknown"))
    location_state = str(trend_features.get("location_state", "unknown"))
    breakout_state = str(trend_features.get("breakout_state", "unknown"))
    trend_alignment_state = str(trend_features.get("trend_alignment_state", "mixed_transition"))
    channel_direction = str(trend_features.get("channel_direction", "mixed"))
    swing_bias = str(trend_features.get("swing_bias", "mixed"))
    swing_quality_score = float(trend_features.get("swing_quality_score", 0.0) or 0.0)
    swing_compression_score = float(trend_features.get("swing_compression_score", 0.0) or 0.0)
    structure_break_state = str(trend_features.get("structure_break_state", "none"))
    channel_stability_score = float(trend_features.get("channel_stability_score", 0.0) or 0.0)
    level_reclaim_state = str(trend_features.get("level_reclaim_state", "none"))
    recent_breakout_followthrough_score = float(trend_features.get("recent_breakout_followthrough_score", 0.0) or 0.0)
    recent_breakout_failure_score = float(trend_features.get("recent_breakout_failure_score", 0.0) or 0.0)
    breakout_margin_pct = float(trend_features.get("breakout_margin_pct", 0.0) or 0.0)
    pattern_name = str(pattern_features.get("pattern", "none"))
    pattern_geometry_score = float(pattern_features.get("pattern_geometry_score", 0.0) or 0.0)
    breakout_authenticity_score = float(pattern_features.get("breakout_authenticity_score", 0.0) or 0.0)
    breakout_retest_quality = str(pattern_features.get("breakout_retest_quality", "unknown"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))
    breakout_quality = str(risk_features.get("breakout_quality", "unknown"))
    consensus_level = str(decision_features.get("consensus_level", "weak"))
    hard_case = bool(decision_features.get("hard_case", False))
    hard_case_score = float(decision_features.get("hard_case_score", 0.0) or 0.0)
    dominant_side = str(decision_features.get("dominant_side", "NEUTRAL")).upper()
    execution_grade = str(decision_features.get("execution_grade", "C"))
    reversal_bias = str(decision_features.get("reversal_bias", "none"))
    reversal_score = float(decision_features.get("reversal_score", 0.0) or 0.0)
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    continuation_bias = str(decision_features.get("continuation_bias", "none"))
    continuation_score = float(decision_features.get("continuation_score", 0.0) or 0.0)
    continuation_exhaustion_risk = str(decision_features.get("continuation_exhaustion_risk", "low"))
    short_horizon_bias = str(decision_features.get("short_horizon_bias", "none"))
    short_horizon_score = float(decision_features.get("short_horizon_score", 0.0) or 0.0)
    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)
    three_bar_path = decision_features.get("three_bar_path", [])
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_consistency = float(decision_features.get("three_bar_path_consistency", 0.0) or 0.0)
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    authority_owner = str(decision_features.get("authority_owner", "shared"))
    confirmed_structure_strength = str(decision_features.get("confirmed_structure_strength", "none"))
    authority_reasons = decision_features.get("authority_reasons", [])
    trend_failure_state = str(decision_features.get("trend_failure_state", "intact"))
    trend_failure_score = float(decision_features.get("trend_failure_score", 0.0) or 0.0)
    trend_failure_reasons = list(decision_features.get("trend_failure_reasons", []) or [])
    structure_semantic_label = str(decision_features.get("structure_semantic_label", "neutral_structure"))
    structure_semantic_bias = str(decision_features.get("structure_semantic_bias", "NONE"))
    structure_semantic_score = float(decision_features.get("structure_semantic_score", 0.0) or 0.0)
    structure_semantic_reasons = decision_features.get("structure_semantic_reasons", [])
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    path_semantic_role = str(decision_features.get("path_semantic_role", "support_only"))
    candidate_patterns = list(decision_features.get("candidate_patterns", []) or [])
    candidate_pattern_summaries = list(decision_features.get("candidate_pattern_summaries", []) or [])

    macro_bias = "neutral"
    macro_change_pct = 0.0
    if macro_kline_data and macro_kline_data.get("Close"):
        macro_close = macro_kline_data.get("Close", [])
        if len(macro_close) >= 2:
            start_close = float(macro_close[0])
            end_close = float(macro_close[-1])
            if start_close:
                macro_change_pct = round((end_close / start_close - 1.0) * 100.0, 4)
            if end_close > start_close * 1.01:
                macro_bias = "bullish"
            elif end_close < start_close * 0.99:
                macro_bias = "bearish"

    if market_regime in {"range", "compression"} and not breakout_confirmed:
        archetype = "range_wait_or_fake_break_watch"
        playbook = "Prefer waiting or cautious execution until breakout confirmation appears."
    elif reversal_confirmed and reversal_bias == "bullish_reversal":
        archetype = "bullish_structure_transition"
        playbook = "Treat as a bullish reversal candidate: let AI judge whether the new upside structure is strong enough to override the stale downside view."
    elif reversal_confirmed and reversal_bias == "bearish_reversal":
        archetype = "bearish_structure_transition"
        playbook = "Treat as a bearish reversal candidate: let AI judge whether the new downside structure is strong enough to override the stale upside view."
    elif continuation_bias == "bullish_continuation_candidate" and continuation_score >= 1.85:
        archetype = "bullish_continuation_pressure"
        playbook = "Treat resistance pressure and momentum crowding as possible continuation, not automatic exhaustion. Prefer asking whether the move is extending rather than fading."
    elif continuation_bias == "bearish_continuation_candidate" and continuation_score >= 1.85:
        archetype = "bearish_continuation_pressure"
        playbook = "Treat support pressure and downside momentum crowding as possible continuation, not automatic exhaustion. Prefer asking whether the move is extending rather than bouncing."
    elif trend_direction == "uptrend" and breakout_confirmed:
        archetype = "trend_continuation_bullish_breakout"
        playbook = "Favor continuation only if follow-through quality remains healthy."
    elif trend_direction == "downtrend" and breakout_confirmed:
        archetype = "trend_continuation_bearish_breakdown"
        playbook = "Favor downside continuation only if breakdown is not fragile."
    elif decision_authority_regime == "hard_skip_uncertainty":
        archetype = "hard_skip_uncertain_setup"
        playbook = "No trustworthy structure is present, so abstention is more rational than forcing a directional bet."
    elif decision_authority_regime == "structure_present_execution_uncertain":
        archetype = "structure_present_but_execution_uncertain"
        playbook = "A structure exists, but execution quality is weak; prefer cautious calibration over hard directional ambition."
    elif decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
        archetype = "confirmed_structure_pending_regime_handoff"
        playbook = "A confirmed new structure exists before full trend-failure maturity; treat this as an early regime handoff candidate rather than as a generic uncertain setup."
    elif decision_authority_regime == "confirmed_structure_priority" and trend_failure_state in {"probable_failure", "confirmed_failure"}:
        archetype = "confirmed_regime_handoff"
        playbook = "Treat this as a regime handoff: confirmed new structure plus old trend failure should outweigh stale continuation inertia."
    elif hard_case:
        archetype = "conflicted_hard_case"
        playbook = "Treat as a calibration-heavy setup; avoid large directional overrides without stacked evidence."
    else:
        archetype = "structured_directional_setup"
        playbook = "Base direction can be used, but location and confirmation still govern execution quality."

    lessons: List[str] = []
    if false_breakout_risk == "high":
        lessons.append("Historical analogs with high false-breakout risk often fail without confirmation.")
    if consensus_level == "strong":
        lessons.append("When multi-family evidence aligns strongly, unnecessary AI overrides usually hurt more than help.")
    if reversal_confirmed:
        lessons.append("Confirmed structure-transition cases deserve more AI authority than ordinary calibration cases.")
    if trend_failure_state in {"probable_failure", "confirmed_failure"}:
        lessons.append("Once the old trend has visibly failed, stale continuation evidence should be discounted.")
    elif decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
        lessons.append("High-quality confirmed structure can deserve directional authority before old-trend failure is fully mature.")
    if continuation_bias != "none" and continuation_score >= 1.85:
        lessons.append("When continuation structure remains intact, overbought or near-resistance conditions do not automatically imply a reversal.")
    if macro_bias != "neutral":
        lessons.append(f"Higher timeframe ({macro_timeframe or 'macro'}) bias currently leans {macro_bias}.")
    if pattern_name not in {"none", "neutral"}:
        lessons.append(f"Current dominant structure resembles a {pattern_name} case.")
    if not lessons:
        lessons.append("Use this case context as a bias-control reference, not as a hard label.")

    setup_quality = "mixed"
    if breakout_confirmed and false_breakout_risk == "low" and breakout_quality == "healthy":
        setup_quality = "confirmed"
    elif market_regime in {"range", "compression"} or false_breakout_risk == "high":
        setup_quality = "fragile"

    location_judgement = "acceptable"
    if dominant_side == "LONG" and location_state == "near_resistance" and breakout_state != "bullish_breakout":
        location_judgement = "poor_for_long"
    elif dominant_side == "SHORT" and location_state == "near_support" and breakout_state != "bearish_breakdown":
        location_judgement = "poor_for_short"
    elif location_state in {"near_support", "near_resistance"}:
        location_judgement = f"edge_location_{location_state}"

    structure_phase = "continuation"
    if reversal_confirmed:
        structure_phase = "confirmed_transition"
    elif reversal_bias != "none" or trend_alignment_state in {
        "bullish_short_vs_bearish_long",
        "bearish_short_vs_bullish_long",
    }:
        structure_phase = "possible_transition"
    elif market_regime in {"range", "compression"}:
        structure_phase = "compression_or_wait"

    ai_focus_points: List[str] = []
    if reversal_confirmed:
        ai_focus_points.append("Judge whether the new structure is strong enough to replace the stale prior trend.")
    elif reversal_bias != "none":
        ai_focus_points.append("Check whether early reversal clues are strong enough to matter or are only noise.")
    if pattern_geometry_score >= 0.45:
        ai_focus_points.append("Pattern geometry looks visually clean, so structure quality should matter more than isolated indicators.")
    if structure_semantic_label != "neutral_structure" and structure_semantic_score >= 0.5:
        ai_focus_points.append(f"Treat the setup as `{structure_semantic_label}` rather than as a generic pattern summary.")
    elif candidate_patterns:
        ai_focus_points.append(f"Do not anchor on a single pattern: secondary geometric candidates also exist ({', '.join(candidate_patterns[:3])}).")
    if structure_confirmation_tier == "confirmed":
        ai_focus_points.append("The structure is already confirmed, so short-horizon path noise should not casually overturn it.")
        if confirmed_structure_strength == "strong":
            ai_focus_points.append("This is a strong confirmed structure, so it should be allowed to contest stale trend inertia even before trend-failure evidence is fully mature.")
    if trend_failure_state in {"probable_failure", "confirmed_failure"}:
        ai_focus_points.append("The older regime is already failing, so treat stale trend inertia as weaker than usual.")
    elif decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
        ai_focus_points.append("Do not lump this into generic uncertainty: the structure is confirmed, only the old-regime failure evidence is still early.")
    elif structure_confirmation_tier == "developing":
        ai_focus_points.append("The structure is developing but not fully confirmed, so direction may be valid while execution still needs caution.")
    elif structure_confirmation_tier == "candidate" and candidate_pattern_summaries:
        ai_focus_points.append("Candidate structures exist but still need confirmation, so judge whether the geometry is improving or fading.")
    if breakout_authenticity_score >= 0.4:
        ai_focus_points.append("Breakout authenticity is high enough that confirmed structure may deserve more authority than stale trend inertia.")
    if breakout_retest_quality == "healthy":
        ai_focus_points.append("The breakout appears to survive an early retest, which makes continuation more credible.")
    if structure_break_state in {"bullish_break", "bearish_break"}:
        ai_focus_points.append(f"The latest swing structure has already shifted via `{structure_break_state}`, so the old trend label may be stale.")
    if trend_failure_reasons:
        ai_focus_points.append(f"Trend-failure evidence is explicit here: {trend_failure_reasons[0]}")
    if swing_compression_score >= 0.3:
        ai_focus_points.append("Recent swing amplitudes have compressed, so watch for a release move rather than treating the window as a flat range.")
    if level_reclaim_state in {"bullish_reclaim", "bearish_reclaim", "failed_bullish_breakout_reentry", "failed_bearish_breakdown_reentry"}:
        ai_focus_points.append(f"Level reclaim behavior is meaningful here: `{level_reclaim_state}`.")
    if continuation_bias != "none":
        ai_focus_points.append("Distinguish true continuation pressure from fake exhaustion: near resistance in an uptrend is not automatically bearish, and near support in a downtrend is not automatically bullish.")
    if channel_direction in {"ascending", "descending", "converging"}:
        ai_focus_points.append(f"Use the channel geometry ({channel_direction}) to judge whether this is continuation, compression, or a likely false break.")
    if breakout_margin_pct >= 0.18:
        ai_focus_points.append("The current move extends beyond the key level by a measurable margin, so breakout geometry deserves explicit attention.")
    if trend_alignment_state in {"bullish_short_vs_bearish_long", "bearish_short_vs_bullish_long"}:
        ai_focus_points.append("Resolve the conflict between short-term slope and longer-term trend direction.")
    if short_horizon_bias != "none":
        ai_focus_points.append("Because the forecast horizon is ultra-short, local support/resistance reactions may matter more than the slower structural trend.")
    if forecast_horizon_bars >= 3:
        ai_focus_points.append("Reason about the next three bars as a path: bar1 reaction, bar2 continuation, and bar3 follow-through may differ.")
        if three_bar_majority_bias in {"LONG", "SHORT"} and three_bar_majority_bias != dominant_side:
            ai_focus_points.append("The three-bar path majority conflicts with the base side, so check whether the base is overfitting the latest bar.")
        if path_semantic_role == "sequence_confirmed":
            ai_focus_points.append("The three-bar path is stable enough to help execution timing, but it should not outrank a confirmed higher-quality structure by default.")
    if location_judgement.startswith("poor_"):
        ai_focus_points.append("Be cautious because the current price location is poor for the algorithmic dominant side.")
    if not breakout_confirmed:
        ai_focus_points.append("Do not over-trust incomplete structures without breakout confirmation.")
    if false_breakout_risk == "high":
        ai_focus_points.append("Prioritize false-breakout defense because this setup is fragile.")
    if macro_bias != "neutral":
        ai_focus_points.append(f"Use the {macro_timeframe or 'macro'} backdrop to test whether the local signal is with or against the larger bias.")
    if not ai_focus_points:
        ai_focus_points.append("Treat this as a normal calibration task and avoid unnecessary overrides.")

    override_trigger_summary = "Require stacked evidence before any directional flip."
    if reversal_confirmed:
        override_trigger_summary = (
            "Override becomes more acceptable only if the new reversal structure, location, and confirmation jointly beat the stale base trend."
        )
    elif decision_authority_regime == "confirmed_structure_priority" and trend_failure_state in {"probable_failure", "confirmed_failure"}:
        override_trigger_summary = (
            "Override becomes acceptable when confirmed structure and explicit old-trend failure jointly show a regime handoff."
        )
    elif decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
        override_trigger_summary = (
            "Override becomes acceptable when a strong confirmed structure exists and the stale base case is no longer clearly superior, even if old-trend failure is only early."
        )
    elif continuation_bias != "none" and continuation_score >= 1.9:
        override_trigger_summary = (
            "Override becomes more acceptable only if continuation structure, breakout pressure, and momentum alignment jointly show that the move is extending rather than exhausting."
        )
    elif hard_case:
        override_trigger_summary = (
            "Only consider override when multiple evidence families align against the base case and the setup is execution-grade."
        )

    no_override_guardrail = "If evidence is mixed, preserve direction and only reduce confidence."
    if market_regime in {"range", "compression"}:
        no_override_guardrail = "In range/compression conditions, avoid confident overrides unless a real breakout or clear transition is present."

    return {
        "asset": asset,
        "timeframe": timeframe,
        "macro_timeframe": macro_timeframe or "",
        "macro_bias": macro_bias,
        "macro_change_pct": macro_change_pct,
        "case_archetype": archetype,
        "case_playbook": playbook,
        "case_lessons": lessons[:3],
        "structure_phase": structure_phase,
        "setup_quality": setup_quality,
        "location_judgement": location_judgement,
        "trend_alignment_state": trend_alignment_state,
        "dominant_side": dominant_side,
        "execution_grade": execution_grade,
        "hard_case_score": round(hard_case_score, 4),
        "reversal_score": round(reversal_score, 4),
        "pattern_geometry_score": round(pattern_geometry_score, 4),
        "breakout_authenticity_score": round(breakout_authenticity_score, 4),
        "breakout_retest_quality": breakout_retest_quality,
        "channel_direction": channel_direction,
        "swing_bias": swing_bias,
        "swing_quality_score": round(swing_quality_score, 4),
        "swing_compression_score": round(swing_compression_score, 4),
        "structure_break_state": structure_break_state,
        "channel_stability_score": round(channel_stability_score, 4),
        "level_reclaim_state": level_reclaim_state,
        "recent_breakout_followthrough_score": round(recent_breakout_followthrough_score, 4),
        "recent_breakout_failure_score": round(recent_breakout_failure_score, 4),
        "breakout_margin_pct": round(breakout_margin_pct, 4),
        "decision_authority_regime": decision_authority_regime,
        "authority_owner": authority_owner,
        "confirmed_structure_strength": confirmed_structure_strength,
        "authority_reasons": authority_reasons[:3],
        "trend_failure_state": trend_failure_state,
        "trend_failure_score": round(trend_failure_score, 4),
        "trend_failure_reasons": trend_failure_reasons[:3],
        "structure_semantic_label": structure_semantic_label,
        "structure_semantic_bias": structure_semantic_bias,
        "structure_semantic_score": round(structure_semantic_score, 4),
        "structure_semantic_reasons": structure_semantic_reasons[:3],
        "structure_confirmation_tier": structure_confirmation_tier,
        "path_semantic_role": path_semantic_role,
        "candidate_patterns": candidate_patterns[:5],
        "candidate_pattern_summaries": candidate_pattern_summaries[:5],
        "continuation_bias": continuation_bias,
        "continuation_score": round(continuation_score, 4),
        "continuation_exhaustion_risk": continuation_exhaustion_risk,
        "short_horizon_bias": short_horizon_bias,
        "short_horizon_score": round(short_horizon_score, 4),
        "three_bar_path": three_bar_path,
        "three_bar_majority_bias": three_bar_majority_bias,
        "three_bar_path_consistency": round(three_bar_path_consistency, 4),
        "ai_focus_points": ai_focus_points[:4],
        "override_trigger_summary": override_trigger_summary,
        "no_override_guardrail": no_override_guardrail,
    }
