"""
Final decision agent for KuantAgent.

This version implements an "algorithm-led, AI-enhanced" decision process:
deterministic features provide the primary base case, while the LLM acts as a
disciplined calibrator that may follow, reduce confidence, or override only
under strict conditions.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Dict, Tuple

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


def invoke_with_retry(call_fn, *args, retries=4, wait_sec=6):
    """Retry transient model-call failures so one backend hiccup does not kill a long benchmark run."""

    for attempt in range(retries):
        try:
            return call_fn(*args)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
            print(
                f"Decision agent transient error: {exc}. "
                f"Retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"Decision agent error: {exc}. "
                f"Retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...",
                file=sys.stderr,
                flush=True,
            )
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Decision agent exceeded maximum retries")


def _extract_json_block(text: str) -> Dict[str, object]:
    """Best-effort JSON extraction from model output."""

    if not text:
        return {}
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _derive_base_decision(decision_features: Dict[str, object]) -> Tuple[str, float, str]:
    """
    Build the algorithm-led base case before AI intervention.

    If the algorithm abstains, we still produce a side for execution purposes,
    but keep confidence low and mark the reason.
    """

    dominant_side = str(decision_features.get("dominant_side", "NEUTRAL")).upper()
    raw_dominant_side = str(decision_features.get("raw_dominant_side", "NEUTRAL")).upper()
    long_score = float(decision_features.get("long_score", 0.0) or 0.0)
    short_score = float(decision_features.get("short_score", 0.0) or 0.0)
    base_confidence = float(decision_features.get("algorithmic_confidence", 0.0) or 0.0)
    signal_gate = str(decision_features.get("signal_gate", "directional"))

    if dominant_side in {"LONG", "SHORT"}:
        return dominant_side, base_confidence, "algorithm_directional"

    fallback_side = raw_dominant_side
    if fallback_side not in {"LONG", "SHORT"}:
        fallback_side = "LONG" if long_score >= short_score else "SHORT"

    fallback_confidence = min(0.18, max(base_confidence, 0.06))
    if signal_gate == "abstain":
        return fallback_side, fallback_confidence, "algorithm_abstained_fallback_side"
    return fallback_side, fallback_confidence, "algorithm_fallback_side"


def _derive_structure_expert_view(decision_features: Dict[str, object]) -> Dict[str, object]:
    """Independent structure expert used by the arbiter before AI review."""

    confirmed_structure_bias = str(decision_features.get("confirmed_structure_bias", "NONE")).upper()
    confirmed_structure_score = float(decision_features.get("confirmed_structure_score", 0.0) or 0.0)
    confirmed_structure_strength = str(decision_features.get("confirmed_structure_strength", "none"))
    structure_semantic_bias = str(decision_features.get("structure_semantic_bias", "NONE")).upper()
    structure_semantic_score = float(decision_features.get("structure_semantic_score", 0.0) or 0.0)
    structure_semantic_label = str(decision_features.get("structure_semantic_label", "neutral_structure"))
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    trend_failure_state = str(decision_features.get("trend_failure_state", "intact"))
    reversal_bias = str(decision_features.get("reversal_bias", "none"))
    reversal_score = float(decision_features.get("reversal_score", 0.0) or 0.0)
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(decision_features.get("three_bar_path_score", 0.0) or 0.0)

    view = {
        "decision": "NONE",
        "confidence": 0.0,
        "reason": "no_structure_edge",
        "strength": "none",
    }

    if (
        confirmed_structure_bias in {"LONG", "SHORT"}
        and confirmed_structure_strength == "strong"
        and structure_confirmation_tier == "confirmed"
    ):
        confidence = 0.24 + min(0.16, max(confirmed_structure_score, structure_semantic_score) * 0.18)
        if decision_authority_regime == "confirmed_structure_priority":
            confidence += 0.08
        elif decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
            confidence += 0.05
        if trend_failure_state in {"early_failure", "probable_failure", "confirmed_failure"}:
            confidence += 0.04
        if three_bar_majority_bias == confirmed_structure_bias and three_bar_path_score >= 0.55:
            confidence += 0.03
        return {
            "decision": confirmed_structure_bias,
            "confidence": _clip(confidence, 0.18, 0.52),
            "reason": f"strong_confirmed_structure:{structure_semantic_label}",
            "strength": "strong",
        }

    if (
        structure_semantic_bias in {"LONG", "SHORT"}
        and structure_confirmation_tier == "confirmed"
        and structure_semantic_score >= 0.72
    ):
        confidence = 0.2 + min(0.12, structure_semantic_score * 0.16)
        return {
            "decision": structure_semantic_bias,
            "confidence": _clip(confidence, 0.16, 0.42),
            "reason": f"confirmed_semantic_structure:{structure_semantic_label}",
            "strength": "medium",
        }

    if reversal_confirmed and reversal_score >= 2.25:
        reversal_decision = "LONG" if reversal_bias == "bullish_reversal" else "SHORT" if reversal_bias == "bearish_reversal" else "NONE"
        if reversal_decision in {"LONG", "SHORT"}:
            return {
                "decision": reversal_decision,
                "confidence": _clip(0.16 + min(0.12, reversal_score * 0.06), 0.14, 0.34),
                "reason": "confirmed_reversal_structure",
                "strength": "medium",
            }

    return view


def _arbitrate_experts(
    algorithm_decision: str,
    algorithm_confidence: float,
    algorithm_source: str,
    structure_view: Dict[str, object],
    decision_features: Dict[str, object],
) -> Tuple[str, float, str, Dict[str, object]]:
    """Arbitrate between the algorithm expert and structure expert."""

    structure_decision = str(structure_view.get("decision", "NONE")).upper()
    structure_confidence = float(structure_view.get("confidence", 0.0) or 0.0)
    structure_strength = str(structure_view.get("strength", "none"))
    structure_reason = str(structure_view.get("reason", "no_structure_edge"))
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    trend_failure_state = str(decision_features.get("trend_failure_state", "intact"))
    dominance_ratio = float(decision_features.get("dominance_ratio", 0.0) or 0.0)
    consensus_level = str(decision_features.get("consensus_level", "weak"))
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))

    arbitration = {
        "winner": "algorithm",
        "structure_view": structure_view,
        "reason": "no_structure_override",
    }

    if structure_decision not in {"LONG", "SHORT"}:
        return algorithm_decision, algorithm_confidence, algorithm_source, arbitration

    if structure_decision == algorithm_decision:
        merged_confidence = _clip(max(algorithm_confidence, structure_confidence), 0.06, 0.62)
        arbitration["winner"] = "consensus"
        arbitration["reason"] = "algorithm_and_structure_align"
        return algorithm_decision, merged_confidence, "expert_arbiter_consensus", arbitration

    if (
        decision_authority_regime in {"confirmed_structure_priority", "confirmed_structure_but_trend_failure_incomplete"}
        and structure_strength == "strong"
        and (
            trend_failure_state in {"early_failure", "probable_failure", "confirmed_failure"}
            or decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
        )
        and (
            algorithm_confidence <= 0.24
            or dominance_ratio <= 0.22
            or three_bar_majority_bias == structure_decision
        )
    ):
        arbitration["winner"] = "structure"
        arbitration["reason"] = structure_reason
        return (
            structure_decision,
            _clip(max(algorithm_confidence, structure_confidence), 0.18, 0.58),
            "expert_arbiter_structure_wins",
            arbitration,
        )

    if (
        structure_strength == "medium"
        and consensus_level != "strong"
        and algorithm_confidence <= 0.14
        and dominance_ratio <= 0.18
    ):
        arbitration["winner"] = "structure"
        arbitration["reason"] = structure_reason
        return (
            structure_decision,
            _clip(max(algorithm_confidence, structure_confidence), 0.14, 0.44),
            "expert_arbiter_soft_structure_wins",
            arbitration,
        )

    return algorithm_decision, algorithm_confidence, algorithm_source, arbitration


def _count_override_signals(
    base_decision: str,
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
    risk_features: Dict[str, object],
    decision_features: Dict[str, object],
) -> Tuple[int, list[str]]:
    """Count strong counter-signals before allowing AI to flip the algorithmic base case."""

    reasons: list[str] = []
    trend_direction = str(trend_features.get("trend_direction", "unknown"))
    location_state = str(trend_features.get("location_state", "unknown"))
    breakout_state = str(trend_features.get("breakout_state", "unknown"))
    market_regime = str(trend_features.get("market_regime", "unknown"))
    pattern_bias = str(pattern_features.get("pattern_bias", "neutral"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))
    consensus_level = str(decision_features.get("consensus_level", "weak"))
    long_votes = int(decision_features.get("long_votes", 0) or 0)
    short_votes = int(decision_features.get("short_votes", 0) or 0)
    indicator_long_score = float(decision_features.get("indicator_long_score", 0.0) or 0.0)
    indicator_short_score = float(decision_features.get("indicator_short_score", 0.0) or 0.0)
    reversal_bias = str(decision_features.get("reversal_bias", "none"))
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    continuation_bias = str(decision_features.get("continuation_bias", "none"))
    continuation_score = float(decision_features.get("continuation_score", 0.0) or 0.0)
    short_horizon_bias = str(decision_features.get("short_horizon_bias", "none"))
    short_horizon_score = float(decision_features.get("short_horizon_score", 0.0) or 0.0)
    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(decision_features.get("three_bar_path_score", 0.0) or 0.0)
    three_bar_path_consistency = float(decision_features.get("three_bar_path_consistency", 0.0) or 0.0)
    confirmed_structure_bias = str(decision_features.get("confirmed_structure_bias", "NONE")).upper()
    confirmed_structure_score = float(decision_features.get("confirmed_structure_score", 0.0) or 0.0)
    confirmed_structure_strength = str(decision_features.get("confirmed_structure_strength", "none"))
    structure_semantic_label = str(decision_features.get("structure_semantic_label", "neutral_structure"))
    structure_semantic_bias = str(decision_features.get("structure_semantic_bias", "NONE")).upper()
    structure_semantic_score = float(decision_features.get("structure_semantic_score", 0.0) or 0.0)
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    channel_dominance_bias = str(decision_features.get("channel_dominance_bias", "NONE")).upper()
    channel_dominance_score = float(decision_features.get("channel_dominance_score", 0.0) or 0.0)
    structure_conflict = bool(decision_features.get("structure_conflict", False))
    signal_conflict_count = int(decision_features.get("signal_conflict_count", 0) or 0)
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    authority_owner = str(decision_features.get("authority_owner", "shared"))

    override_count = 0

    if base_decision == "LONG":
        if trend_direction == "downtrend":
            override_count += 1
            reasons.append("trend structure strongly contradicts LONG")
        if indicator_short_score >= indicator_long_score + 0.45:
            override_count += 1
            reasons.append("indicator momentum strongly contradicts LONG")
        if location_state == "near_resistance" and breakout_state != "bullish_breakout":
            override_count += 1
            reasons.append("LONG location is poor without breakout confirmation")
        if false_breakout_risk == "high":
            override_count += 1
            reasons.append("false breakout risk is high")
        if short_votes >= 2 and long_votes <= 1:
            override_count += 1
            reasons.append("majority of evidence families lean SHORT")
        if (
            pattern_bias == "bearish"
            and breakout_confirmed
            and market_regime in {"trend", "transition"}
        ):
            override_count += 1
            reasons.append("confirmed bearish structure contradicts LONG")
        if reversal_confirmed and reversal_bias == "bearish_reversal":
            override_count += 1
            reasons.append("confirmed bearish reversal structure contradicts LONG")
        if continuation_bias == "bearish_continuation_candidate" and continuation_score >= 1.9:
            override_count += 1
            reasons.append("downside continuation structure contradicts LONG")
        if confirmed_structure_bias == "SHORT" and confirmed_structure_score >= 0.24:
            override_count += 1
            reasons.append("confirmed bearish structure has stronger priority than the current LONG base")
        if (
            decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
            and confirmed_structure_strength == "strong"
            and confirmed_structure_bias == "SHORT"
        ):
            override_count += 1
            reasons.append("strong confirmed bearish structure is already contesting the stale LONG base before full trend-failure maturity")
        if structure_semantic_bias == "SHORT" and structure_semantic_score >= 0.58:
            override_count += 1
            reasons.append(f"structural semantic `{structure_semantic_label}` contradicts LONG")
        if (
            channel_dominance_bias == "SHORT"
            and channel_dominance_score >= 0.26
            and not reversal_confirmed
        ):
            override_count += 1
            reasons.append("dominant bearish channel suggests bullish rebounds are still counter-trend only")
        if (
            forecast_horizon_bars >= 3
            and three_bar_majority_bias == "SHORT"
            and three_bar_path_score >= 0.75
            and three_bar_path_consistency >= 0.99
            and structure_confirmation_tier != "confirmed"
        ):
            override_count += 1
            reasons.append("three-bar path majority contradicts LONG")
        if (
            forecast_horizon_bars <= 2
            and short_horizon_bias == "bearish_pullback_candidate"
            and short_horizon_score >= 1.35
        ):
            override_count += 1
            reasons.append("short-horizon pullback evidence contradicts LONG")
    else:
        if trend_direction == "uptrend":
            override_count += 1
            reasons.append("trend structure strongly contradicts SHORT")
        if indicator_long_score >= indicator_short_score + 0.45:
            override_count += 1
            reasons.append("indicator momentum strongly contradicts SHORT")
        if location_state == "near_support" and breakout_state != "bearish_breakdown":
            override_count += 1
            reasons.append("SHORT location is poor without breakdown confirmation")
        if false_breakout_risk == "high":
            override_count += 1
            reasons.append("false breakout risk is high")
        if long_votes >= 2 and short_votes <= 1:
            override_count += 1
            reasons.append("majority of evidence families lean LONG")
        if (
            pattern_bias == "bullish"
            and breakout_confirmed
            and market_regime in {"trend", "transition"}
        ):
            override_count += 1
            reasons.append("confirmed bullish structure contradicts SHORT")
        if reversal_confirmed and reversal_bias == "bullish_reversal":
            override_count += 1
            reasons.append("confirmed bullish reversal structure contradicts SHORT")
        if continuation_bias == "bullish_continuation_candidate" and continuation_score >= 1.9:
            override_count += 1
            reasons.append("upside continuation structure contradicts SHORT")
        if confirmed_structure_bias == "LONG" and confirmed_structure_score >= 0.24:
            override_count += 1
            reasons.append("confirmed bullish structure has stronger priority than the current SHORT base")
        if (
            decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
            and confirmed_structure_strength == "strong"
            and confirmed_structure_bias == "LONG"
        ):
            override_count += 1
            reasons.append("strong confirmed bullish structure is already contesting the stale SHORT base before full trend-failure maturity")
        if structure_semantic_bias == "LONG" and structure_semantic_score >= 0.58:
            override_count += 1
            reasons.append(f"structural semantic `{structure_semantic_label}` contradicts SHORT")
        if (
            channel_dominance_bias == "LONG"
            and channel_dominance_score >= 0.26
            and not reversal_confirmed
        ):
            override_count += 1
            reasons.append("dominant bullish channel suggests bearish pullbacks are still counter-trend only")
        if (
            forecast_horizon_bars >= 3
            and three_bar_majority_bias == "LONG"
            and three_bar_path_score >= 0.75
            and three_bar_path_consistency >= 0.99
            and structure_confirmation_tier != "confirmed"
        ):
            override_count += 1
            reasons.append("three-bar path majority contradicts SHORT")
        if (
            forecast_horizon_bars <= 2
            and short_horizon_bias == "bullish_rebound_candidate"
            and short_horizon_score >= 1.35
        ):
            override_count += 1
            reasons.append("short-horizon rebound evidence contradicts SHORT")

    # Strong algorithmic consensus raises the threshold by making overrides rarer.
    if consensus_level == "strong":
        reasons.append("algorithmic consensus is strong, so override threshold is higher")
    if structure_conflict and signal_conflict_count >= 1:
        reasons.append("mixed structural conflict raises the bar for any directional flip")
    if authority_owner == "algorithm":
        reasons.append("this sample belongs to trend-inertia priority, so override authority is intentionally narrow")
    elif authority_owner == "ai_structure":
        reasons.append("this sample belongs to confirmed-structure priority, so fresh structure deserves extra weight")
    if decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete":
        reasons.append("confirmed structure is already strong enough to contest the stale base case before trend-failure evidence fully matures")

    return override_count, reasons


def _override_allowed(base_decision: str, proposed_decision: str, decision_features: Dict[str, object]) -> bool:
    """Programmatic guardrail: AI may only flip when the algorithmic base case is weak enough."""

    if proposed_decision not in {"LONG", "SHORT"} or proposed_decision == base_decision:
        return False

    consensus_level = str(decision_features.get("consensus_level", "weak"))
    dominance_ratio = float(decision_features.get("dominance_ratio", 0.0) or 0.0)
    algorithmic_confidence = float(decision_features.get("algorithmic_confidence", 0.0) or 0.0)
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    reversal_score = float(decision_features.get("reversal_score", 0.0) or 0.0)
    continuation_bias = str(decision_features.get("continuation_bias", "none"))
    continuation_score = float(decision_features.get("continuation_score", 0.0) or 0.0)
    short_horizon_bias = str(decision_features.get("short_horizon_bias", "none"))
    short_horizon_score = float(decision_features.get("short_horizon_score", 0.0) or 0.0)
    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(decision_features.get("three_bar_path_score", 0.0) or 0.0)
    three_bar_path_consistency = float(decision_features.get("three_bar_path_consistency", 0.0) or 0.0)
    confirmed_structure_bias = str(decision_features.get("confirmed_structure_bias", "NONE")).upper()
    confirmed_structure_score = float(decision_features.get("confirmed_structure_score", 0.0) or 0.0)
    confirmed_structure_strength = str(decision_features.get("confirmed_structure_strength", "none"))
    structure_semantic_bias = str(decision_features.get("structure_semantic_bias", "NONE")).upper()
    structure_semantic_score = float(decision_features.get("structure_semantic_score", 0.0) or 0.0)
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    channel_dominance_bias = str(decision_features.get("channel_dominance_bias", "NONE")).upper()
    channel_dominance_score = float(decision_features.get("channel_dominance_score", 0.0) or 0.0)
    structure_conflict = bool(decision_features.get("structure_conflict", False))
    signal_conflict_count = int(decision_features.get("signal_conflict_count", 0) or 0)
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    authority_owner = str(decision_features.get("authority_owner", "shared"))

    # Keep overrides rare: algorithms still own the base direction in the
    # intended 80/20 design. AI may intervene only when the base edge is weak.
    if consensus_level == "strong" and dominance_ratio >= 0.38:
        return False
    if algorithmic_confidence >= 0.38 and dominance_ratio >= 0.28:
        return False
    if decision_authority_regime == "hard_skip_uncertainty":
        return False
    if structure_confirmation_tier == "confirmed" and structure_semantic_bias in {"LONG", "SHORT"}:
        if proposed_decision != structure_semantic_bias:
            return False
    if authority_owner == "algorithm" and proposed_decision != base_decision:
        return False
    if (
        structure_conflict
        and signal_conflict_count >= 1
        and confirmed_structure_bias != proposed_decision
    ):
        return False
    if (
        channel_dominance_bias not in {"NONE", "NEUTRAL"}
        and channel_dominance_bias != proposed_decision
        and channel_dominance_score >= 0.26
        and not reversal_confirmed
    ):
        return False
    if (
        confirmed_structure_bias == proposed_decision
        and confirmed_structure_score >= 0.24
        and algorithmic_confidence <= 0.26
    ):
        return True
    if (
        decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
        and confirmed_structure_strength == "strong"
        and confirmed_structure_bias == proposed_decision
        and (
            confirmed_structure_score >= 0.18
            or structure_semantic_score >= 0.72
        )
        and algorithmic_confidence <= 0.32
        and dominance_ratio <= 0.34
    ):
        return True
    if (
        structure_semantic_bias == proposed_decision
        and structure_semantic_score >= 0.64
        and algorithmic_confidence <= 0.24
        and dominance_ratio <= 0.3
    ):
        return True
    if (
        authority_owner == "ai_structure"
        and confirmed_structure_bias == proposed_decision
        and confirmed_structure_score >= 0.3
        and algorithmic_confidence <= 0.3
    ):
        return True
    if reversal_confirmed and reversal_score >= 2.25 and algorithmic_confidence <= 0.24:
        return True
    continuation_direction_ok = (
        (continuation_bias == "bullish_continuation_candidate" and proposed_decision == "LONG")
        or (continuation_bias == "bearish_continuation_candidate" and proposed_decision == "SHORT")
    )
    if (
        continuation_direction_ok
        and continuation_score >= 1.9
        and algorithmic_confidence <= 0.16
        and dominance_ratio <= 0.24
    ):
        return True
    three_bar_direction_ok = (
        forecast_horizon_bars >= 3
        and three_bar_majority_bias == proposed_decision
        and three_bar_path_score >= 0.82
        and three_bar_path_consistency >= 0.99
        and structure_confirmation_tier != "confirmed"
    )
    if three_bar_direction_ok and algorithmic_confidence <= 0.18 and dominance_ratio <= 0.26:
        return True
    if (
        forecast_horizon_bars <= 2
        and short_horizon_score >= 1.45
        and short_horizon_bias != "none"
        and algorithmic_confidence <= 0.12
    ):
        return True
    if forecast_horizon_bars < 3 and algorithmic_confidence <= 0.08 and dominance_ratio <= 0.16:
        return True
    return False


def _determine_intervention_policy(
    decision_features: Dict[str, object],
    base_source: str,
) -> Tuple[str, str]:
    """
    Decide how much freedom the AI should have.

    Policies:
    - confidence_only: strong algorithmic edge, AI may only calibrate confidence
    - limited_override: weak/moderate edge, AI may override with strict evidence
    - tie_breaker: algorithm abstained or watchlist-quality setup, AI may resolve ambiguity
    - hard_case_override: hard reversal/structure-transition sample where AI may gain extra flip authority
    """

    signal_gate = str(decision_features.get("signal_gate", "directional"))
    consensus_level = str(decision_features.get("consensus_level", "weak"))
    algorithmic_confidence = float(decision_features.get("algorithmic_confidence", 0.0) or 0.0)
    dominance_ratio = float(decision_features.get("dominance_ratio", 0.0) or 0.0)
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    reversal_score = float(decision_features.get("reversal_score", 0.0) or 0.0)
    continuation_bias = str(decision_features.get("continuation_bias", "none"))
    continuation_score = float(decision_features.get("continuation_score", 0.0) or 0.0)
    hard_case_score = float(decision_features.get("hard_case_score", 0.0) or 0.0)
    short_horizon_bias = str(decision_features.get("short_horizon_bias", "none"))
    short_horizon_score = float(decision_features.get("short_horizon_score", 0.0) or 0.0)
    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(decision_features.get("three_bar_path_score", 0.0) or 0.0)
    three_bar_path_consistency = float(decision_features.get("three_bar_path_consistency", 0.0) or 0.0)
    confirmed_structure_bias = str(decision_features.get("confirmed_structure_bias", "NONE")).upper()
    confirmed_structure_score = float(decision_features.get("confirmed_structure_score", 0.0) or 0.0)
    confirmed_structure_strength = str(decision_features.get("confirmed_structure_strength", "none"))
    structure_semantic_score = float(decision_features.get("structure_semantic_score", 0.0) or 0.0)
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    structure_conflict = bool(decision_features.get("structure_conflict", False))
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    authority_owner = str(decision_features.get("authority_owner", "shared"))
    trend_failure_state = str(decision_features.get("trend_failure_state", "intact"))
    trend_failure_score = float(decision_features.get("trend_failure_score", 0.0) or 0.0)

    if signal_gate == "abstain" or consensus_level == "watchlist" or "abstained" in base_source:
        return (
            "tie_breaker",
            "Algorithmic evidence is too conflicted or too weak, so AI may resolve the side if it sees a clearer setup.",
        )

    if (
        decision_authority_regime == "trend_inertia_priority"
        and authority_owner == "algorithm"
        and trend_failure_state == "intact"
    ):
        return (
            "confidence_only",
            "Dominant channel inertia still owns this sample, so AI should mainly calibrate confidence and avoid fighting the prevailing structure.",
        )

    if decision_authority_regime == "hard_skip_uncertainty":
        return (
            "confidence_only",
            "No trustworthy structure is present, so selective abstention should dominate and AI should avoid forcing a directional flip.",
        )

    if decision_authority_regime == "structure_present_execution_uncertain":
        return (
            "limited_override",
            "A structure exists, but execution quality is weak, so AI may calibrate direction cautiously without treating this as a full hard-skip setup.",
        )

    if (
        decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
        and confirmed_structure_bias in {"LONG", "SHORT"}
        and confirmed_structure_strength == "strong"
        and (
            confirmed_structure_score >= 0.18
            or structure_semantic_score >= 0.72
        )
    ):
        return (
            "hard_case_override",
            "A strong confirmed structure is already present, so AI may let it contest stale inertia even before old-trend failure is fully mature.",
        )

    if structure_conflict and confirmed_structure_score < 0.24:
        return (
            "confidence_only",
            "Signals are internally conflicted, so AI should calibrate confidence rather than force a directional flip.",
        )

    if (
        decision_authority_regime == "confirmed_structure_priority"
        and trend_failure_state in {"probable_failure", "confirmed_failure"}
        and confirmed_structure_bias in {"LONG", "SHORT"}
        and confirmed_structure_score >= 0.22
        and trend_failure_score >= 0.28
    ):
        return (
            "hard_case_override",
            "Confirmed structure plus explicit old-trend failure gives AI real authority to replace stale trend inertia.",
        )

    if confirmed_structure_bias in {"LONG", "SHORT"} and confirmed_structure_score >= 0.24 and algorithmic_confidence <= 0.26:
        return (
            "hard_case_override",
            "This sample contains a confirmed reversal or breakout structure, so AI may flip only if it agrees with the confirmed structure rather than stale trend inertia.",
        )

    if reversal_confirmed and reversal_score >= 2.1 and hard_case_score >= 0.28:
        return (
            "hard_case_override",
            "This is a confirmed structure-transition hard case, so AI may gain extra authority to flip direction if the new regime is clearly stronger than the stale algorithmic read.",
        )

    if continuation_bias != "none" and continuation_score >= 1.9 and algorithmic_confidence <= 0.16:
        return (
            "hard_case_override",
            "This sample has strong continuation pressure against the stale base read, so AI may flip if it can justify continuation beating apparent exhaustion.",
        )

    if (
        forecast_horizon_bars >= 3
        and three_bar_majority_bias in {"LONG", "SHORT"}
        and three_bar_path_score >= 0.82
        and three_bar_path_consistency >= 0.99
        and algorithmic_confidence <= 0.18
        and structure_confirmation_tier != "confirmed"
    ):
        return (
            "hard_case_override",
            "This is a three-bar path case: AI may flip only if bar1 reaction, bar2 continuation, and bar3 follow-through jointly support the opposite side.",
        )

    if (
        forecast_horizon_bars <= 2
        and short_horizon_bias != "none"
        and short_horizon_score >= 1.45
        and algorithmic_confidence <= 0.12
    ):
        return (
            "hard_case_override",
            "This is a short-horizon support/resistance reaction case, so AI may prefer the immediate rebound/pullback signal over a slower stale trend read.",
        )

    if consensus_level == "strong" or (algorithmic_confidence >= 0.18 and dominance_ratio >= 0.18):
        return (
            "confidence_only",
            "Algorithmic base case is already meaningful, so AI should mainly calibrate confidence and avoid flipping direction.",
        )

    return (
        "limited_override",
        "Algorithmic edge is only weak or moderate, so AI may override only when multiple strong counter-signals align.",
    )


def _route_ai_intervention(
    decision_features: Dict[str, object],
    base_decision: str,
    base_confidence: float,
) -> Tuple[str, str]:
    """Choose whether this sample needs AI review or can stay algorithm-only."""

    ai_route_hint = str(decision_features.get("ai_route_hint", "algorithm_first_ai_optional"))
    ai_intervention_reason = str(
        decision_features.get(
            "ai_intervention_reason",
            "algorithm edge is usable but may still benefit from calibration",
        )
    )
    consensus_level = str(decision_features.get("consensus_level", "weak"))
    hard_case = bool(decision_features.get("hard_case", False))
    simple_case = bool(decision_features.get("simple_case", False))
    signal_gate = str(decision_features.get("signal_gate", "directional"))

    if (
        ai_route_hint == "algorithm_only"
        and simple_case
        and not hard_case
        and signal_gate == "directional"
        and base_decision in {"LONG", "SHORT"}
        and consensus_level == "strong"
        and base_confidence >= 0.12
    ):
        return (
            "algorithm_only",
            f"AI review skipped because this is a simple strong-edge sample: {ai_intervention_reason}.",
        )

    return (
        "ai_calibrator",
        f"AI review enabled because this sample is not a clean simple-case pass-through: {ai_intervention_reason}.",
    )


def _compose_final_result(
    *,
    base_decision: str,
    base_confidence: float,
    base_source: str,
    ai_output: Dict[str, object],
    indicator_features: Dict[str, object],
    pattern_features: Dict[str, object],
    trend_features: Dict[str, object],
    risk_features: Dict[str, object],
    decision_features: Dict[str, object],
    case_context: Dict[str, object] | None = None,
    macro_context: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Combine algorithmic base case with bounded AI intervention."""

    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)

    ai_action = str(ai_output.get("action", "follow")).lower()
    proposed_override = str(ai_output.get("override_decision", "NONE")).upper()
    signal_quality = str(ai_output.get("signal_quality", "medium")).lower()
    execution_advice = str(ai_output.get("execution_advice", "cautious")).lower()
    confidence_adjustment = ai_output.get("confidence_adjustment", 0.0)
    risk_reward_ratio = ai_output.get("risk_reward_ratio", 1.5)
    reasoning_summary = str(ai_output.get("reasoning_summary", "")).strip()

    try:
        confidence_adjustment = float(confidence_adjustment)
    except Exception:
        confidence_adjustment = 0.0
    confidence_adjustment = _clip(confidence_adjustment, -0.25, 0.1)

    try:
        risk_reward_ratio = float(risk_reward_ratio)
    except Exception:
        risk_reward_ratio = 1.5
    risk_reward_ratio = round(_clip(risk_reward_ratio, 1.2, 1.8), 2)

    if signal_quality not in {"high", "medium", "low"}:
        signal_quality = "medium"
    if execution_advice not in {"execute", "cautious", "skip"}:
        execution_advice = "cautious"

    override_signal_count, override_reasons = _count_override_signals(
        base_decision=base_decision,
        indicator_features=indicator_features,
        pattern_features=pattern_features,
        trend_features=trend_features,
        risk_features=risk_features,
        decision_features=decision_features,
    )
    intervention_policy, policy_reason = _determine_intervention_policy(
        decision_features=decision_features,
        base_source=base_source,
    )
    override_gate_open = _override_allowed(base_decision, proposed_override, decision_features)
    override_applied = False
    market_regime = str(trend_features.get("market_regime", "unknown"))
    breakout_confirmed = bool(pattern_features.get("breakout_confirmed", False))
    reversal_confirmed = bool(decision_features.get("reversal_confirmed", False))
    reversal_score = float(decision_features.get("reversal_score", 0.0) or 0.0)
    continuation_bias = str(decision_features.get("continuation_bias", "none"))
    continuation_score = float(decision_features.get("continuation_score", 0.0) or 0.0)
    short_horizon_bias = str(decision_features.get("short_horizon_bias", "none"))
    short_horizon_score = float(decision_features.get("short_horizon_score", 0.0) or 0.0)
    forecast_horizon_bars = int(decision_features.get("forecast_horizon_bars", 1) or 1)
    three_bar_majority_bias = str(decision_features.get("three_bar_majority_bias", "MIXED"))
    three_bar_path_score = float(decision_features.get("three_bar_path_score", 0.0) or 0.0)
    three_bar_path_consistency = float(decision_features.get("three_bar_path_consistency", 0.0) or 0.0)
    confirmed_structure_bias = str(decision_features.get("confirmed_structure_bias", "NONE")).upper()
    confirmed_structure_score = float(decision_features.get("confirmed_structure_score", 0.0) or 0.0)
    structure_confirmation_tier = str(decision_features.get("structure_confirmation_tier", "none"))
    structure_semantic_bias = str(decision_features.get("structure_semantic_bias", "NONE")).upper()
    channel_dominance_bias = str(decision_features.get("channel_dominance_bias", "NONE")).upper()
    channel_dominance_score = float(decision_features.get("channel_dominance_score", 0.0) or 0.0)
    structure_conflict = bool(decision_features.get("structure_conflict", False))
    decision_authority_regime = str(decision_features.get("decision_authority_regime", "balanced_calibration"))
    authority_owner = str(decision_features.get("authority_owner", "shared"))
    if (
        ai_action == "override"
        and override_gate_open
        and proposed_override in {"LONG", "SHORT"}
        and proposed_override != base_decision
    ):
        short_horizon_direction_ok = (
            (short_horizon_bias == "bullish_rebound_candidate" and proposed_override == "LONG")
            or (short_horizon_bias == "bearish_pullback_candidate" and proposed_override == "SHORT")
        )
        continuation_direction_ok = (
            (continuation_bias == "bullish_continuation_candidate" and proposed_override == "LONG")
            or (continuation_bias == "bearish_continuation_candidate" and proposed_override == "SHORT")
        )
        confirmed_structure_direction_ok = (
            confirmed_structure_bias == proposed_override and confirmed_structure_score >= 0.24
        )
        strong_confirmed_structure_direction_ok = (
            decision_authority_regime == "confirmed_structure_but_trend_failure_incomplete"
            and confirmed_structure_strength == "strong"
            and confirmed_structure_bias == proposed_override
            and (
                confirmed_structure_score >= 0.18
                or (
                    structure_semantic_bias == proposed_override
                    and structure_confirmation_tier == "confirmed"
                )
            )
        )
        three_bar_direction_ok = (
            forecast_horizon_bars >= 3
            and three_bar_majority_bias == proposed_override
            and three_bar_path_score >= 0.82
            and three_bar_path_consistency >= 0.99
            and structure_confirmation_tier != "confirmed"
        )
        channel_supports_override = (
            channel_dominance_bias in {"NONE", "NEUTRAL", proposed_override}
            or channel_dominance_score < 0.26
        )
        authority_supports_override = (
            authority_owner in {"shared", "ai_structure", "ai_calibrator"}
            and decision_authority_regime != "hard_skip_uncertainty"
        )
        semantic_direction_ok = (
            structure_semantic_bias == proposed_override
            and structure_confirmation_tier in {"developing", "confirmed"}
        )
        if (
            intervention_policy == "tie_breaker"
            and override_signal_count >= 2
            and signal_quality == "high"
            and execution_advice == "execute"
            and breakout_confirmed
            and market_regime not in {"range", "compression"}
            and channel_supports_override
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "limited_override"
            and override_signal_count >= 3
            and signal_quality == "high"
            and execution_advice == "execute"
            and breakout_confirmed
            and market_regime not in {"range", "compression"}
            and base_confidence <= 0.08
            and channel_supports_override
            and not structure_conflict
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and reversal_confirmed
            and reversal_score >= 2.1
            and market_regime != "compression"
            and base_confidence <= 0.24
            and channel_supports_override
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and strong_confirmed_structure_direction_ok
            and base_confidence <= 0.32
            and channel_supports_override
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and confirmed_structure_direction_ok
            and base_confidence <= 0.26
            and channel_supports_override
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and semantic_direction_ok
            and structure_confirmation_tier == "confirmed"
            and base_confidence <= 0.24
            and channel_supports_override
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and forecast_horizon_bars <= 2
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and short_horizon_bias != "none"
            and short_horizon_score >= 1.45
            and short_horizon_direction_ok
            and base_confidence <= 0.12
            and channel_supports_override
            and not structure_conflict
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and continuation_bias != "none"
            and continuation_score >= 1.9
            and continuation_direction_ok
            and base_confidence <= 0.16
            and channel_supports_override
            and not structure_conflict
            and authority_supports_override
        ):
            override_applied = True
        elif (
            intervention_policy == "hard_case_override"
            and override_signal_count >= 2
            and signal_quality in {"high", "medium"}
            and execution_advice in {"execute", "cautious"}
            and three_bar_direction_ok
            and base_confidence <= 0.18
            and channel_supports_override
            and not structure_conflict
            and authority_supports_override
        ):
            override_applied = True

    final_decision = base_decision
    final_confidence = base_confidence
    skip_execution = False

    if intervention_policy == "confidence_only":
        if ai_action in {"reduce_confidence", "override"}:
            final_confidence = _clip(base_confidence + min(confidence_adjustment, -0.04), 0.02, 0.75)
        else:
            final_confidence = _clip(base_confidence + max(confidence_adjustment, 0.0), 0.02, 0.72)
    elif ai_action == "reduce_confidence":
        final_confidence = _clip(base_confidence + min(confidence_adjustment, -0.04), 0.02, 0.75)
    elif override_applied:
        final_decision = proposed_override
        # Even when override is allowed, keep confidence disciplined because AI is the secondary layer.
        final_confidence = _clip(max(base_confidence, 0.18) + confidence_adjustment, 0.12, 0.62)
    else:
        final_confidence = _clip(base_confidence + max(confidence_adjustment, 0.0), 0.02, 0.72)

    if decision_features.get("signal_gate") == "abstain":
        final_confidence = min(final_confidence, 0.22)
    if str(decision_features.get("consensus_level", "weak")) == "strong" and not override_applied:
        final_confidence = min(final_confidence, max(base_confidence + 0.06, final_confidence))

    location_state = str(trend_features.get("location_state", "unknown"))
    breakout_state = str(trend_features.get("breakout_state", "unknown"))
    poor_long_location = final_decision == "LONG" and location_state == "near_resistance" and breakout_state != "bullish_breakout"
    poor_short_location = final_decision == "SHORT" and location_state == "near_support" and breakout_state != "bearish_breakdown"
    false_breakout_risk = str(risk_features.get("false_breakout_risk", "unknown"))
    weak_environment = market_regime in {"range", "compression"}
    base_is_weak = base_confidence <= 0.08

    low_quality_skip_candidate = (
        signal_quality == "low"
        and execution_advice == "skip"
        and intervention_policy in {"limited_override", "tie_breaker", "hard_case_override"}
        and base_is_weak
        and weak_environment
        and not breakout_confirmed
        and not reversal_confirmed
        and (poor_long_location or poor_short_location)
    )
    medium_quality_skip_candidate = (
        not low_quality_skip_candidate
        and intervention_policy in {"limited_override", "tie_breaker", "hard_case_override"}
        and execution_advice in {"skip", "cautious"}
        and signal_quality in {"low", "medium"}
        and base_is_weak
        and not breakout_confirmed
        and not reversal_confirmed
        and (
            weak_environment
            or poor_long_location
            or poor_short_location
            or false_breakout_risk == "high"
            or override_signal_count >= 2
        )
    )
    if (
        forecast_horizon_bars <= 2
        and short_horizon_bias != "none"
        and short_horizon_score >= 1.45
    ):
        low_quality_skip_candidate = False
        medium_quality_skip_candidate = False
    if continuation_bias != "none" and continuation_score >= 1.9 and not poor_long_location and not poor_short_location:
        medium_quality_skip_candidate = False
    if low_quality_skip_candidate:
        skip_execution = True
        final_confidence = min(final_confidence, 0.05)
    elif medium_quality_skip_candidate:
        skip_execution = True
        final_confidence = min(final_confidence, 0.08)

    if execution_advice == "execute":
        executable_quality_ok = (
            signal_quality == "high"
            and (
                breakout_confirmed
                or (
                    not weak_environment
                    and not poor_long_location
                    and not poor_short_location
                    and false_breakout_risk != "high"
                    and base_confidence >= 0.08
                )
            )
        )
        if not executable_quality_ok and not override_applied:
            execution_advice = "cautious"

    if reversal_confirmed and execution_advice == "skip":
        execution_advice = "cautious"
    if (
        forecast_horizon_bars <= 2
        and short_horizon_bias != "none"
        and short_horizon_score >= 1.45
        and execution_advice == "skip"
    ):
        execution_advice = "cautious"
    if continuation_bias != "none" and continuation_score >= 1.9 and execution_advice == "skip":
        execution_advice = "cautious"
    if reversal_confirmed and override_applied:
        final_confidence = max(final_confidence, 0.18)
    if (
        forecast_horizon_bars <= 2
        and short_horizon_bias != "none"
        and short_horizon_score >= 1.45
        and not skip_execution
    ):
        final_confidence = max(final_confidence, 0.1)
    if continuation_bias != "none" and continuation_score >= 1.9 and not skip_execution:
        final_confidence = max(final_confidence, 0.12)

    final_confidence = round(_clip(final_confidence, 0.02, 0.8), 4)

    if not reasoning_summary:
        reasoning_summary = "AI calibration output was sparse, so the algorithmic base case was preserved."

    justification_parts = [
        f"Base decision: {base_decision} ({base_source})",
        f"Intervention policy: {intervention_policy}",
        f"AI action: {ai_action}",
        policy_reason,
        reasoning_summary,
    ]
    if override_applied:
        justification_parts.append(
            "Override applied because at least two strong counter-signals were present: "
            + "; ".join(override_reasons[:3])
        )
    elif ai_action == "override":
        justification_parts.append(
            "Override rejected; the current intervention policy or counter-signal strength did not justify a flip."
        )
    elif ai_action == "reduce_confidence":
        justification_parts.append("Direction preserved, but confidence was reduced due to setup quality concerns.")
    if reversal_confirmed:
        justification_parts.append("This sample was treated as a structure-transition hard case, so reversal evidence received extra review weight.")
    if continuation_bias != "none" and continuation_score >= 1.9:
        justification_parts.append("Continuation-vs-exhaustion was treated as a key review axis so the model would not automatically fade a strong trend just because price looked stretched.")
    if forecast_horizon_bars >= 3:
        justification_parts.append(
            f"Three-bar path review: {decision_features.get('three_bar_path', [])}, majority={three_bar_majority_bias}, consistency={three_bar_path_consistency}."
        )
    if skip_execution:
        justification_parts.append(
            "Execution is downgraded to skip because this is a low-quality setup with poor location, weak confirmation, and insufficient edge."
        )

    return {
        "forecast_horizon": f"next {forecast_horizon_bars} bar" + ("s" if forecast_horizon_bars != 1 else ""),
        "forecast_horizon_bars": forecast_horizon_bars,
        "model_role": "decision_calibrator",
        "decision": final_decision,
        "confidence": final_confidence,
        "justification": " ".join(part for part in justification_parts if part).strip(),
        "risk_reward_ratio": risk_reward_ratio,
        "base_decision": base_decision,
        "base_confidence": round(base_confidence, 4),
        "base_source": base_source,
        "expert_arbitration": decision_features.get("expert_arbitration", {}),
        "algorithm_expert_decision": decision_features.get("algorithm_expert_decision", ""),
        "algorithm_expert_confidence": float(decision_features.get("algorithm_expert_confidence", 0.0) or 0.0),
        "structure_expert_view": decision_features.get("structure_expert_view", {}),
        "intervention_policy": intervention_policy,
        "ai_action": ai_action,
        "signal_quality": signal_quality,
        "execution_advice": "skip" if skip_execution else execution_advice,
        "override_candidate": proposed_override,
        "override_signal_count": override_signal_count,
        "override_gate_open": override_gate_open,
        "override_applied": override_applied,
        "calibration_applied": ai_action in {"reduce_confidence", "override"} or override_applied,
        "hard_case_score": float(decision_features.get("hard_case_score", 0.0) or 0.0),
        "execution_grade": str(decision_features.get("execution_grade", "C")),
        "reversal_bias": str(decision_features.get("reversal_bias", "none")),
        "reversal_score": float(decision_features.get("reversal_score", 0.0) or 0.0),
        "reversal_confirmed": bool(decision_features.get("reversal_confirmed", False)),
        "continuation_bias": str(decision_features.get("continuation_bias", "none")),
        "continuation_score": float(decision_features.get("continuation_score", 0.0) or 0.0),
        "short_horizon_bias": str(decision_features.get("short_horizon_bias", "none")),
        "short_horizon_score": float(decision_features.get("short_horizon_score", 0.0) or 0.0),
        "pattern_geometry_score": float(decision_features.get("pattern_geometry_score", 0.0) or 0.0),
        "breakout_authenticity_score": float(decision_features.get("breakout_authenticity_score", 0.0) or 0.0),
        "breakout_body_ratio": float(decision_features.get("breakout_body_ratio", 0.0) or 0.0),
        "breakout_retest_quality": str(decision_features.get("breakout_retest_quality", "unknown")),
        "structure_semantic_label": str(decision_features.get("structure_semantic_label", "neutral_structure")),
        "structure_semantic_bias": str(decision_features.get("structure_semantic_bias", "NONE")),
        "structure_semantic_score": float(decision_features.get("structure_semantic_score", 0.0) or 0.0),
        "structure_confirmation_tier": str(decision_features.get("structure_confirmation_tier", "none")),
        "confirmed_structure_strength": str(decision_features.get("confirmed_structure_strength", "none")),
        "path_semantic_role": str(decision_features.get("path_semantic_role", "support_only")),
        "three_bar_path": decision_features.get("three_bar_path", []),
        "three_bar_majority_bias": three_bar_majority_bias,
        "three_bar_path_consistency": three_bar_path_consistency,
        "three_bar_path_score": three_bar_path_score,
        "trend_alignment_state": str(decision_features.get("trend_alignment_state", "mixed_transition")),
        "decision_authority_regime": decision_authority_regime,
        "authority_owner": authority_owner,
        "case_context": case_context or {},
        "macro_context": macro_context or {},
    }


def create_final_trade_decider(llm):
    """Create the final decision node used in the graph."""

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        indicator_features = state.get("indicator_features", {})
        pattern_features = state.get("pattern_features", {})
        trend_features = state.get("trend_features", {})
        risk_features = state.get("risk_features", {})
        decision_features = dict(state.get("decision_features", {}))
        macro_timeframe = state.get("macro_timeframe", "")
        macro_kline_data = state.get("macro_kline_data", {})
        case_context = state.get("case_context", {})
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        algorithm_decision, algorithm_confidence, algorithm_source = _derive_base_decision(decision_features)
        structure_view = _derive_structure_expert_view(decision_features)
        base_decision, base_confidence, base_source, arbitration_summary = _arbitrate_experts(
            algorithm_decision=algorithm_decision,
            algorithm_confidence=algorithm_confidence,
            algorithm_source=algorithm_source,
            structure_view=structure_view,
            decision_features=decision_features,
        )
        decision_features["expert_arbitration"] = arbitration_summary
        decision_features["algorithm_expert_decision"] = algorithm_decision
        decision_features["algorithm_expert_confidence"] = round(algorithm_confidence, 4)
        decision_features["structure_expert_view"] = structure_view
        override_signal_count, override_reasons = _count_override_signals(
            base_decision=base_decision,
            indicator_features=indicator_features,
            pattern_features=pattern_features,
            trend_features=trend_features,
            risk_features=risk_features,
            decision_features=decision_features,
        )
        consensus_level = str(decision_features.get("consensus_level", "weak"))
        intervention_policy, policy_reason = _determine_intervention_policy(
            decision_features=decision_features,
            base_source=base_source,
        )
        decision_route, route_reason = _route_ai_intervention(
            decision_features=decision_features,
            base_decision=base_decision,
            base_confidence=base_confidence,
        )

        if decision_route == "algorithm_only":
            execution_grade = str(decision_features.get("execution_grade", "B"))
            synthetic_ai_output = {
                "action": "follow",
                "override_decision": "NONE",
                "signal_quality": "high" if execution_grade == "A" else "medium",
                "execution_advice": "execute" if execution_grade == "A" else "cautious",
                "confidence_adjustment": 0.0,
                "risk_reward_ratio": 1.5,
                "reasoning_summary": route_reason,
            }
            final_result = _compose_final_result(
                base_decision=base_decision,
                base_confidence=base_confidence,
                base_source=base_source,
                ai_output=synthetic_ai_output,
                indicator_features=indicator_features,
                pattern_features=pattern_features,
                trend_features=trend_features,
                risk_features=risk_features,
                decision_features=decision_features,
                case_context=case_context,
                macro_context={
                    "macro_timeframe": macro_timeframe,
                    "macro_bias": case_context.get("macro_bias", "") if isinstance(case_context, dict) else "",
                    "macro_change_pct": case_context.get("macro_change_pct", 0.0) if isinstance(case_context, dict) else 0.0,
                },
            )
            final_result["decision_route"] = decision_route
            final_result["route_reason"] = route_reason
            final_result["ai_review_skipped"] = True
            final_result["calibration_mode"] = "algorithm_passthrough"
            return {
                "final_trade_decision": json.dumps(final_result, ensure_ascii=False, indent=2),
                "messages": [],
                "decision_prompt": "",
            }

        prompt = f"""
You are the final calibration layer in a short-horizon quantitative trading system.
You are analyzing the current {time_frame} market window for {stock_name}.

You are NOT the primary judge. The algorithm analysis layer already produced a base case.
Your job is to behave like a disciplined trading supervisor and choose one of:
- follow
- reduce_confidence
- override

Role split:
1. Deterministic algorithms own the primary directional judgment.
2. You review market regime, setup quality, location, confirmation, and risk.
3. You may only recommend override when multiple strong counter-signals clearly exist.
4. If evidence is mixed, keep the base side and reduce confidence instead of forcing a flip.

Expert checklist:
A. Market regime
- trend, range, compression, or transition

B. Location quality
- LONG is better near support than near resistance
- SHORT is better near resistance than near support

C. Confirmation quality
- confirmed breakout/breakdown > completed but unconfirmed pattern
- unconfirmed patterns are supporting evidence only
- when a reversal or breakout is confirmed, it should usually outrank stale opposing momentum unless the dominant channel is still clearly intact

D. Risk review
- high volatility, weak trend, fragile breakout, and high false-breakout risk should reduce confidence

E. Intervention discipline
- If policy is `confidence_only`, do not flip direction. Only choose follow or reduce_confidence.
- If policy is `limited_override`, you may override only when multiple strong counter-signals clearly align AND the setup is execution-grade.
- If policy is `tie_breaker`, the algorithm is weak/abstaining; prefer `skip` or `cautious` unless the counter-case is exceptionally clear and confirmed.
- If policy is `hard_case_override`, this is a structure-transition sample. You may override when reversal evidence is stacked, recent, and more trustworthy than the stale trend-following base case.
- If strong `continuation_bias` is present, explicitly decide whether the move is a real continuation or only apparent exhaustion. Do not fade a strong trend only because oscillators look stretched.
- If `forecast_horizon_bars >= 3`, reason in three steps: bar1 reaction, bar2 continuation, and bar3 follow-through. The final side should maximize the three-bar step score, not merely the last-bar cumulative return.
- If `forecast_horizon_bars <= 2`, short-horizon support bounces and resistance rejections deserve extra attention. These micro-reaction signals may temporarily beat the slower structural trend read.
- If the evidence is mixed but not decisive, prefer reduce_confidence over override.
- In weak range/compression setups with poor location or missing confirmation, prefer `skip`.
- Do not overuse `skip` on confirmed structure-transition cases; when reversal evidence is real, `cautious` is usually better than suppressing the trade entirely.

Your output must be valid JSON only:
{{
  "action": "<follow | reduce_confidence | override>",
  "override_decision": "<LONG | SHORT | NONE>",
  "signal_quality": "<high | medium | low>",
  "execution_advice": "<execute | cautious | skip>",
  "confidence_adjustment": "<float between -0.25 and 0.10>",
  "risk_reward_ratio": "<float between 1.2 and 1.8>",
  "reasoning_summary": "<brief reasoning mentioning regime, location, confirmation, and risk>"
}}

Quality guidelines:
- high: confirmed setup, acceptable location, manageable risk, and enough asymmetry to act
- medium: direction is acceptable but setup quality is mixed; prefer cautious over aggressive execution
- low: weak range/compression setup, poor location, missing confirmation, or conflict is too high

Execution guidelines:
- execute: quality is high enough that the system should treat this as a real tradable signal
- cautious: keep the direction but treat the setup as lower conviction
- skip: the setup is too weak/fragile to count as an execution-grade signal even if a side can still be described

Important restrictions:
- Do NOT label a setup `high` if breakout confirmation is missing and the market is still in range/compression.
- Do NOT output `execute` for poor-location setups (LONG near resistance without breakout, SHORT near support without breakdown).
- For watchlist or abstain-like setups, `skip` is usually safer than forcing a strong execution-grade call.
- If `reversal_confirmed=true`, weigh short-vs-long trend inflection and support/resistance rejection heavily; these are the main places where AI is allowed to beat the stale algorithmic base case.
- If `continuation_bias` is strong, weigh breakout pressure, trend structure integrity, and momentum continuation heavily; these are the main places where AI may keep or restore a trend-following LONG/SHORT despite superficial exhaustion cues.
- If `decision_authority_regime=confirmed_structure_priority`, treat confirmed structure as the lead evidence family. Fresh breakout geometry may deserve more authority than stale inertia.
- If `decision_authority_regime=confirmed_structure_but_trend_failure_incomplete`, do not collapse this into generic uncertainty. A strong confirmed structure may already contest the stale base case even before old-trend failure is fully mature.
- If `trend_failure_state` is `probable_failure` or `confirmed_failure`, explicitly discount stale trend continuation. Confirmed new structure may replace the old regime.
- If `decision_authority_regime=trend_inertia_priority`, do not fight the prevailing channel unless a truly confirmed new structure exists.
- If `decision_authority_regime=hard_skip_uncertainty`, risk control wins completely: do not force a directional bet.
- If `decision_authority_regime=structure_present_execution_uncertain`, a structure exists but execution quality is weak, so prefer cautious calibration over hard abstention.
- If `structure_confirmation_tier=confirmed`, confirmed structure outranks a conflicting three-bar path unless the structure itself is shown to be false.
- If `structure_semantic_label` is not `neutral_structure`, treat it as a market-structure object, not as a decorative tag. It describes what the geometry is doing, such as release, reclaim, false break, or structural break.
- If `candidate_patterns` contains multiple credible structures, do not anchor on the single top-ranked pattern. Secondary candidates may explain why the chart still behaves like an early reversal or continuation setup.
- If the semantic label is an early structure such as `developing_*` or `emerging_*`, use it mainly to calibrate confidence and execution quality. Do not let early structure alone justify a directional flip.
- If `confirmed_structure_bias` is present, treat it as a privileged signal. Confirmed V-reversal, flag breakout, or neckline break should beat stale trend inertia more often than ambiguous oscillator drift.
- If `confirmed_structure_strength=strong`, treat the confirmed structure as a near-lead signal even when `trend_failure_state=early_failure`. Do not force it back into the same bucket as an ordinary developing setup.
- If `channel_dominance_bias` is strong and opposite to a local bounce/rejection, treat that local move as counter-trend unless a real reversal is confirmed.
- If `structure_conflict=true`, do not force an override unless the override side is also backed by confirmed structure.
- If `three_bar_majority_bias` conflicts with the base side, only override when all three projected bars align and the path is supported by regime, location, and momentum. A 2-vs-1 path is context, not enough reason to flip by itself.
- If `short_horizon_bias` is present and the forecast horizon is only 1-2 bars, do not automatically suppress the trade just because the broader trend still points the other way.

Current algorithm-led base case:
{{
  "base_decision": "{base_decision}",
  "base_confidence": {round(base_confidence, 4)},
  "base_source": "{base_source}"
}}

Programmatic calibration context:
{{
  "consensus_level": "{consensus_level}",
  "intervention_policy": "{intervention_policy}",
  "policy_reason": "{policy_reason}",
  "override_signal_count": {override_signal_count},
  "override_reasons": {json.dumps(override_reasons[:5], ensure_ascii=False)},
  "expert_arbitration": {json.dumps(decision_features.get("expert_arbitration", {}), ensure_ascii=False)},
  "algorithm_expert_decision": "{decision_features.get("algorithm_expert_decision", "")}",
  "algorithm_expert_confidence": {decision_features.get("algorithm_expert_confidence", 0.0)},
  "structure_expert_view": {json.dumps(decision_features.get("structure_expert_view", {}), ensure_ascii=False)}
}}

AI decision brief:
{json.dumps(
    {
        "case_archetype": case_context.get("case_archetype", ""),
        "structure_phase": case_context.get("structure_phase", ""),
        "setup_quality": case_context.get("setup_quality", ""),
        "location_judgement": case_context.get("location_judgement", ""),
        "trend_alignment_state": case_context.get("trend_alignment_state", ""),
        "dominant_side": case_context.get("dominant_side", ""),
        "execution_grade": case_context.get("execution_grade", ""),
        "hard_case_score": case_context.get("hard_case_score", 0.0),
        "reversal_score": case_context.get("reversal_score", 0.0),
        "continuation_bias": decision_features.get("continuation_bias", "none"),
        "continuation_score": decision_features.get("continuation_score", 0.0),
        "short_horizon_bias": decision_features.get("short_horizon_bias", "none"),
        "short_horizon_score": decision_features.get("short_horizon_score", 0.0),
        "confirmed_structure_bias": decision_features.get("confirmed_structure_bias", "NONE"),
        "confirmed_structure_score": decision_features.get("confirmed_structure_score", 0.0),
        "confirmed_structure_strength": decision_features.get("confirmed_structure_strength", "none"),
        "trend_failure_state": decision_features.get("trend_failure_state", "intact"),
        "trend_failure_score": decision_features.get("trend_failure_score", 0.0),
        "trend_failure_reasons": decision_features.get("trend_failure_reasons", []),
        "structure_semantic_label": decision_features.get("structure_semantic_label", "neutral_structure"),
        "structure_semantic_bias": decision_features.get("structure_semantic_bias", "NONE"),
        "structure_semantic_score": decision_features.get("structure_semantic_score", 0.0),
        "structure_semantic_reasons": decision_features.get("structure_semantic_reasons", []),
        "structure_confirmation_tier": decision_features.get("structure_confirmation_tier", "none"),
        "path_semantic_role": decision_features.get("path_semantic_role", "support_only"),
        "candidate_patterns": decision_features.get("candidate_patterns", []),
        "candidate_pattern_summaries": decision_features.get("candidate_pattern_summaries", []),
        "breakout_authenticity_score": decision_features.get("breakout_authenticity_score", 0.0),
        "breakout_body_ratio": decision_features.get("breakout_body_ratio", 0.0),
        "breakout_retest_quality": decision_features.get("breakout_retest_quality", "unknown"),
        "channel_dominance_bias": decision_features.get("channel_dominance_bias", "NONE"),
        "channel_dominance_score": decision_features.get("channel_dominance_score", 0.0),
        "decision_authority_regime": decision_features.get("decision_authority_regime", "balanced_calibration"),
        "authority_owner": decision_features.get("authority_owner", "shared"),
        "authority_reasons": decision_features.get("authority_reasons", []),
        "structure_conflict": decision_features.get("structure_conflict", False),
        "signal_conflict_count": decision_features.get("signal_conflict_count", 0),
        "three_bar_path": decision_features.get("three_bar_path", []),
        "three_bar_majority_bias": decision_features.get("three_bar_majority_bias", "MIXED"),
        "three_bar_path_consistency": decision_features.get("three_bar_path_consistency", 0.0),
        "three_bar_path_score": decision_features.get("three_bar_path_score", 0.0),
        "three_bar_path_reasons": decision_features.get("three_bar_path_reasons", []),
        "ai_focus_points": case_context.get("ai_focus_points", []),
        "override_trigger_summary": case_context.get("override_trigger_summary", ""),
        "no_override_guardrail": case_context.get("no_override_guardrail", ""),
    },
    indent=2,
    ensure_ascii=False,
)}

Structured indicator features:
{json.dumps(indicator_features, indent=2)}

Structured pattern features:
{json.dumps(pattern_features, indent=2)}

Structured trend features:
{json.dumps(trend_features, indent=2)}

Structured risk features:
{json.dumps(risk_features, indent=2)}

Pre-fused decision context:
{json.dumps(decision_features, indent=2)}

Macro timeframe context:
{json.dumps({"macro_timeframe": macro_timeframe, "macro_kline_data": macro_kline_data}, indent=2)}

Case-memory context:
{json.dumps(case_context, indent=2)}

Indicator report:
{indicator_report}

Pattern report:
{pattern_report}

Trend report:
{trend_report}
"""

        response = invoke_with_retry(llm.invoke, prompt)
        ai_output = _extract_json_block(getattr(response, "content", ""))
        final_result = _compose_final_result(
            base_decision=base_decision,
            base_confidence=base_confidence,
            base_source=base_source,
            ai_output=ai_output,
            indicator_features=indicator_features,
            pattern_features=pattern_features,
            trend_features=trend_features,
            risk_features=risk_features,
            decision_features=decision_features,
            case_context=case_context,
            macro_context={
                "macro_timeframe": macro_timeframe,
                "macro_bias": case_context.get("macro_bias", "") if isinstance(case_context, dict) else "",
                "macro_change_pct": case_context.get("macro_change_pct", 0.0) if isinstance(case_context, dict) else 0.0,
            },
        )
        final_result["decision_route"] = decision_route
        final_result["route_reason"] = route_reason
        final_result["ai_review_skipped"] = False
        final_result["calibration_mode"] = intervention_policy

        return {
            "final_trade_decision": json.dumps(final_result, ensure_ascii=False, indent=2),
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node
