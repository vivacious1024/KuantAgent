from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _log(message: str) -> None:
    print(f"[run_full_graph_sample] {message}", file=sys.stderr, flush=True)


def _normalize_ohlc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "Datetime",
        "datetime": "Datetime",
        "timestamp": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    }
    normalized = df.copy()
    normalized.columns = [rename_map.get(str(col).strip().lower(), str(col)) for col in normalized.columns]
    required = ["Datetime", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized = normalized[required].copy()
    normalized["Datetime"] = pd.to_datetime(normalized["Datetime"])
    for col in ["Open", "High", "Low", "Close"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=required).sort_values("Datetime").reset_index(drop=True)
    return normalized


def _build_kline_dict(df: pd.DataFrame) -> Dict[str, list[Any]]:
    return {
        "Datetime": df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": df["Open"].tolist(),
        "High": df["High"].tolist(),
        "Low": df["Low"].tolist(),
        "Close": df["Close"].tolist(),
    }


def _display_timeframe(timeframe: str) -> str:
    if timeframe.endswith("h"):
        return f"{timeframe[:-1]} hour"
    if timeframe.endswith("m"):
        return f"{timeframe[:-1]} min"
    if timeframe.endswith("d"):
        return f"{timeframe[:-1]} day"
    if timeframe == "1w":
        return "1 week"
    if timeframe == "1mo":
        return "1 month"
    return timeframe


def _extract_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _apply_llm_overrides(
    base_config: Dict[str, Any],
    *,
    llm_preset: str,
    agent_llm_model: str,
    graph_llm_model: str,
    vision_llm_model: str,
    qwen_base_url: str,
    qwen_api_env_name: str,
) -> Dict[str, Any]:
    config = dict(base_config)
    preset = (llm_preset or "default").strip().lower()

    preset_defaults: Dict[str, str] = {}
    if preset == "siliconflow_qwen":
        preset_defaults = {
            "agent_llm_model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            "graph_llm_model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            "vision_llm_model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            "qwen_base_url": "https://api.siliconflow.cn/v1",
            "qwen_api_env_name": "SILICONFLOW_API_KEY",
        }
    elif preset == "mimo":
        preset_defaults = {
            "agent_llm_model": "mimo-v2.5-pro",
            "graph_llm_model": "mimo-v2.5-pro",
            "vision_llm_model": "mimo-v2.5-pro",
            "qwen_base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "qwen_api_env_name": "MIMO_API_KEY",
        }
    elif preset not in {"default", "custom"}:
        raise ValueError(f"Unsupported llm preset: {llm_preset}")

    for key, value in preset_defaults.items():
        config[key] = value

    explicit_overrides = {
        "agent_llm_model": agent_llm_model,
        "graph_llm_model": graph_llm_model,
        "vision_llm_model": vision_llm_model,
        "qwen_base_url": qwen_base_url,
        "qwen_api_env_name": qwen_api_env_name,
    }
    for key, value in explicit_overrides.items():
        if str(value or "").strip():
            config[key] = str(value).strip()

    return config


def run_full_graph_sample(
    project_root: Path,
    csv_path: Path,
    asset: str,
    timeframe: str,
    window_size: int,
    future_horizon: int,
    neutral_threshold_pct: float,
    llm_preset: str = "default",
    agent_llm_model: str = "",
    graph_llm_model: str = "",
    vision_llm_model: str = "",
    qwen_base_url: str = "",
    qwen_api_env_name: str = "",
) -> Dict[str, Any]:
    _log(f"Preparing sample {csv_path.name} for project {project_root.name}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from default_config import DEFAULT_CONFIG  # type: ignore
    from trading_graph import TradingGraph  # type: ignore
    import static_util  # type: ignore

    raw_df = pd.read_csv(csv_path)
    normalized = _normalize_ohlc_dataframe(raw_df)
    if len(normalized) < window_size + future_horizon:
        raise ValueError(
            f"Not enough rows for benchmark split: need at least {window_size + future_horizon}, got {len(normalized)}"
        )

    input_df = normalized.iloc[:window_size].reset_index(drop=True)
    future_df = normalized.iloc[window_size : window_size + future_horizon].reset_index(drop=True)
    kline_dict = _build_kline_dict(input_df)
    is_kuant_project = project_root.name.lower() == "kuantagent"
    macro_context: Dict[str, Any] = {}
    case_context: Dict[str, Any] = {}
    if is_kuant_project:
        from context_enrichment import build_case_context, build_macro_context  # type: ignore

        macro_context = build_macro_context(input_df, timeframe=timeframe) or {}
        case_context = build_case_context(
            asset=asset,
            timeframe=timeframe,
            pattern_features={},
            trend_features={},
            risk_features={},
            decision_features={},
        )
    display_timeframe = _display_timeframe(timeframe)
    run_config = _apply_llm_overrides(
        DEFAULT_CONFIG,
        llm_preset=llm_preset,
        agent_llm_model=agent_llm_model,
        graph_llm_model=graph_llm_model,
        vision_llm_model=vision_llm_model,
        qwen_base_url=qwen_base_url,
        qwen_api_env_name=qwen_api_env_name,
    )
    use_multimodal_images = bool(run_config.get("use_multimodal_images", True))

    pattern_image = ""
    trend_image = ""
    if use_multimodal_images:
        _log("Generating chart images")
        p_image = static_util.generate_kline_image(kline_dict)
        t_image = static_util.generate_trend_image(kline_dict)
        pattern_image = p_image["pattern_image"]
        trend_image = t_image["trend_image"]
    else:
        _log("Structured-only mode enabled; skipping chart generation")

    initial_state = {
        "kline_data": kline_dict,
        "analysis_results": None,
        "messages": [],
        "time_frame": display_timeframe,
        "stock_name": asset,
        "pattern_image": pattern_image,
        "trend_image": trend_image,
    }
    if is_kuant_project:
        initial_state.update(
            {
                "forecast_horizon_bars": future_horizon,
                "macro_timeframe": str(macro_context.get("macro_timeframe", "")),
                "macro_kline_data": macro_context.get("macro_kline_data", {}),
                "case_context": case_context,
            }
        )

    _log("Initializing TradingGraph")
    trading_graph = TradingGraph(config=run_config)
    _log("Invoking LangGraph workflow")
    final_state = trading_graph.graph.invoke(initial_state)
    _log("Graph invocation completed")

    input_close = float(input_df.iloc[-1]["Close"])
    future_close = float(future_df.iloc[-1]["Close"])
    future_return_pct = float((future_close / input_close - 1.0) * 100.0)
    true_direction = "LONG" if future_return_pct >= 0 else "SHORT"
    is_neutral_move = int(abs(future_return_pct) < neutral_threshold_pct)

    final_decision_raw = final_state.get("final_trade_decision", "")
    final_decision_json = _extract_json_block(final_decision_raw)
    if is_kuant_project and "case_context" not in final_decision_json:
        final_decision_json["case_context"] = final_state.get("case_context", case_context)
    if is_kuant_project and "macro_context" not in final_decision_json and macro_context:
        final_decision_json["macro_context"] = {
            key: value for key, value in macro_context.items() if key != "macro_kline_data"
        }
    predicted = str(final_decision_json.get("decision", "")).upper()
    comparison_closes = [input_close] + [float(value) for value in future_df["Close"].tolist()]
    future_step_directions = [
        "LONG" if next_close >= prev_close else "SHORT"
        for prev_close, next_close in zip(comparison_closes, comparison_closes[1:])
    ]
    horizon_correct_count = int(sum(1 for direction in future_step_directions if predicted == direction))
    horizon_total_count = len(future_step_directions)
    horizon_step_accuracy = 0.0 if horizon_total_count == 0 else horizon_correct_count / horizon_total_count
    final_direction_correct = int(predicted == true_direction)
    horizon_majority_correct = final_direction_correct
    if future_horizon > 1 and horizon_total_count > 0:
        horizon_majority_correct = int(horizon_correct_count >= ((horizon_total_count // 2) + 1))
    confidence = final_decision_json.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None

    return {
        "sample_file": str(csv_path),
        "asset": asset,
        "timeframe": timeframe,
        "future_horizon": future_horizon,
        "predicted": predicted,
        "true_direction": true_direction,
        "final_direction_correct": final_direction_correct,
        "horizon_majority_correct": horizon_majority_correct,
        "correct": horizon_majority_correct,
        "confidence": confidence,
        "future_return_pct": round(future_return_pct, 4),
        "horizon_correct_count": horizon_correct_count,
        "horizon_total_count": horizon_total_count,
        "horizon_step_accuracy": round(horizon_step_accuracy, 4),
        "future_step_directions": future_step_directions,
        "is_neutral_move": is_neutral_move,
        "raw_decision": final_decision_raw,
        "parsed_decision": final_decision_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full TradingGraph sample for KuantAgent or QuantAgent.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--neutral-threshold-pct", type=float, default=0.15)
    parser.add_argument(
        "--llm-preset",
        default="default",
        choices=["default", "siliconflow_qwen", "mimo", "custom"],
        help="Predefined full-system LLM endpoint/model preset.",
    )
    parser.add_argument("--agent-llm-model", default="", help="Optional override for agent LLM model name.")
    parser.add_argument("--graph-llm-model", default="", help="Optional override for graph LLM model name.")
    parser.add_argument("--vision-llm-model", default="", help="Optional override for vision LLM model name.")
    parser.add_argument("--qwen-base-url", default="", help="Optional override for qwen-compatible base URL.")
    parser.add_argument("--qwen-api-env-name", default="", help="Optional preferred environment variable for the qwen-compatible API key.")
    args = parser.parse_args()

    result = run_full_graph_sample(
        project_root=Path(args.project_root).resolve(),
        csv_path=Path(args.csv_path).resolve(),
        asset=args.asset,
        timeframe=args.timeframe,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        neutral_threshold_pct=args.neutral_threshold_pct,
        llm_preset=args.llm_preset,
        agent_llm_model=args.agent_llm_model,
        graph_llm_model=args.graph_llm_model,
        vision_llm_model=args.vision_llm_model,
        qwen_base_url=args.qwen_base_url,
        qwen_api_env_name=args.qwen_api_env_name,
    )
    # Emit ASCII-safe JSON so Windows GBK consoles cannot crash on model outputs
    # containing symbols such as superscripts, math characters, or non-ASCII text.
    sys.stdout.write(json.dumps(result, ensure_ascii=True, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
