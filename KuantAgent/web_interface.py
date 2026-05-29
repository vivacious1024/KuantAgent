import json
import os
import re
import traceback
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_file
from openai import OpenAI

import static_util
from data_processing import (
    OHLCWindowConfig,
    normalize_ohlc_dataframe,
    prepare_market_data,
)
from context_enrichment import build_case_context, build_macro_context
from trading_graph import TradingGraph

app = Flask(__name__)


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer."""
        from default_config import DEFAULT_CONFIG
        # Start with the benchmark-aligned default config (Mimo)
        self.config = DEFAULT_CONFIG.copy()
        self.trading_graph = TradingGraph(config=self.config)
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Available assets and their display names
        self.asset_mapping = {
            "SPX": "S&P 500",
            "BTC": "Bitcoin",
            "GC": "Gold Futures",
            "NQ": "Nasdaq Futures",
            "CL": "Crude Oil",
            "ES": "E-mini S&P 500",
            "DJI": "Dow Jones",
            "QQQ": "Invesco QQQ Trust",
            "VIX": "Volatility Index",
            "DXY": "US Dollar Index",
            "AAPL": "Apple Inc.",  # New asset
            "TSLA": "Tesla Inc.",  # New asset
        }

        # Yahoo Finance symbol mapping
        self.yfinance_symbols = {
            "SPX": "^GSPC",  # S&P 500
            "BTC": "BTC-USD",  # Bitcoin
            "GC": "GC=F",  # Gold Futures
            "NQ": "NQ=F",  # Nasdaq Futures
            "CL": "CL=F",  # Crude Oil
            "ES": "ES=F",  # E-mini S&P 500
            "DJI": "^DJI",  # Dow Jones
            "QQQ": "QQQ",  # Invesco QQQ Trust
            "VIX": "^VIX",  # Volatility Index
            "DXY": "DX-Y.NYB",  # US Dollar Index
        }

        # Yahoo Finance interval mapping
        self.yfinance_intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # yfinance supports 4h natively!
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo",
        }

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()
        self.use_multimodal_images = bool(self.config.get("use_multimodal_images", False))

    def build_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def current_provider(self) -> str:
        return str(self.config.get("agent_llm_provider", "mimo")).strip().lower()

    def current_model(self) -> str:
        return str(self.config.get("agent_llm_model", "")).strip()

    def current_provider_label(self) -> str:
        provider = self.current_provider()
        mapping = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "qwen": "Qwen / SiliconFlow",
            "mimo": "Mimo",
        }
        return mapping.get(provider, provider)

    def persist_run_log(self, payload: Dict[str, Any]) -> None:
        run_id = payload.get("run_metadata", {}).get("run_id") or self.build_run_id()
        target = self.logs_dir / f"{run_id}.json"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def fetch_yfinance_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            df = yf.download(
                tickers=yf_symbol, start=start_date, end=end_date, interval=yf_interval
            )

            if df is None or df.empty:
                return pd.DataFrame()

            return normalize_ohlc_dataframe(df, keep_volume=False)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_yfinance_data_with_datetime(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance using datetime objects for exact time precision."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            print(
                f"Fetching {yf_symbol} from {start_datetime} to {end_datetime} with interval {yf_interval}"
            )

            # Use datetime objects directly for yfinance
            df = yf.download(
                tickers=yf_symbol,
                start=start_datetime,
                end=end_datetime,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )

            if df is None or df.empty:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = normalize_ohlc_dataframe(df, keep_volume=False)

            print(f"Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_assets(self) -> list:
        """Get list of available assets from the asset mapping dictionary."""
        return sorted(list(self.asset_mapping.keys()))

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def load_uploaded_market_data(self, uploaded_file: Any) -> pd.DataFrame:
        """Load user-uploaded CSV/XLSX market data into a DataFrame."""
        filename = (getattr(uploaded_file, "filename", "") or "").strip()
        if not filename:
            raise ValueError("No uploaded file was provided.")

        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(uploaded_file.stream)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(uploaded_file.stream)
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    def run_analysis(
        self,
        df: pd.DataFrame,
        asset_name: str,
        timeframe: str,
        source_label: str = "Yahoo Finance",
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        run_id = self.build_run_id()
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            # Debug: Check DataFrame structure
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame index: {type(df.index)}")
            print(f"DataFrame shape: {df.shape}")

            prepared_market_data = prepare_market_data(
                raw_df=df,
                config=OHLCWindowConfig(window_size=45, keep_volume=False),
            )
            normalized_df = prepared_market_data["normalized_df"]
            df_slice = prepared_market_data["window_df"]
            df_slice_dict = prepared_market_data["ohlc_dict"]
            macro_context = build_macro_context(df_slice, timeframe=timeframe) or {}
            case_context = build_case_context(
                asset=asset_name,
                timeframe=timeframe,
                pattern_features={},
                trend_features={},
                risk_features={},
                decision_features={},
            )

            # Debug: Check the resulting dictionary
            print(f"Dictionary keys: {list(df_slice_dict.keys())}")
            print(f"Dictionary key types: {[type(k) for k in df_slice_dict.keys()]}")

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            pattern_image = ""
            trend_image = ""
            if self.use_multimodal_images:
                p_image = static_util.generate_kline_image(df_slice_dict)
                t_image = static_util.generate_trend_image(df_slice_dict)
                pattern_image = p_image["pattern_image"]
                trend_image = t_image["trend_image"]

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "forecast_horizon_bars": 3,
                "macro_timeframe": str(macro_context.get("macro_timeframe", "")),
                "macro_kline_data": macro_context.get("macro_kline_data", {}),
                "case_context": case_context,
                "pattern_image": pattern_image,
                "trend_image": trend_image,
            }

            # Run the trading graph
            final_state = self.trading_graph.graph.invoke(initial_state)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice),
                "pipeline_summary": {
                    "data_source": source_label,
                    "raw_rows": int(len(df)),
                    "normalized_rows": int(len(normalized_df)),
                    "window_rows": int(len(df_slice)),
                    "window_size": 45,
                    "future_horizon": 3,
                    "first_timestamp": str(df_slice["Datetime"].iloc[0]) if not df_slice.empty else "",
                    "last_timestamp": str(df_slice["Datetime"].iloc[-1]) if not df_slice.empty else "",
                },
                "run_metadata": {
                    "run_id": run_id,
                    "started_at": started_at,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "provider": self.current_provider(),
                    "provider_label": self.current_provider_label(),
                    "model": self.current_model(),
                    "analysis_mode": "web_interface",
                    "graph_invoked": True,
                    "data_source": source_label,
                },
            }

        except Exception as e:
            error_msg = str(e)
            print(f"!!! ANALYSIS ERROR DETAILS !!!: {error_msg}")
            # print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Get current provider from config
            provider = self.config.get("agent_llm_provider", "mimo")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "mimo":
                provider_name = "Mimo"
            else:
                provider_name = "SiliconFlow"

            run_metadata = {
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "provider": self.current_provider(),
                "provider_label": provider_name,
                "model": self.current_model(),
                "analysis_mode": "web_interface",
                "graph_invoked": False,
                "data_source": source_label,
            }

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "run_metadata": run_metadata,
                    "error": f"Invalid API Key: The {provider_name} API key you provided is invalid or has expired. Please check your API key and try again.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "run_metadata": run_metadata,
                    "error": f"Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "run_metadata": run_metadata,
                    "error": f"Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "run_metadata": run_metadata,
                    "error": f"Network Error: Unable to connect to {provider_name} servers. Please check your internet connection and try again.",
                }
            else:
                return {
                    "success": False,
                    "run_metadata": run_metadata,
                    "error": f"Analysis Error: {error_msg}",
                }

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {
                "success": False,
                "error": results["error"],
                "run_metadata": results.get("run_metadata", {}),
            }

        final_state = results["final_state"]

        indicator_features = final_state.get("indicator_features", {}) or {}
        pattern_features = final_state.get("pattern_features", {}) or {}
        trend_features = final_state.get("trend_features", {}) or {}
        risk_features = final_state.get("risk_features", {}) or {}
        decision_features = final_state.get("decision_features", {}) or {}
        case_context = final_state.get("case_context", {}) or {}

        # Extract analysis results from state fields
        technical_indicators = final_state.get("indicator_report", "")
        pattern_analysis = final_state.get("pattern_report", "")
        trend_analysis = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find("{")
                end = final_decision_raw.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": decision_data.get("decision", "N/A"),
                        "confidence": decision_data.get("confidence", "N/A"),
                        "risk_reward_ratio": decision_data.get(
                            "risk_reward_ratio", "N/A"
                        ),
                        "forecast_horizon": decision_data.get(
                            "forecast_horizon", "N/A"
                        ),
                        "forecast_horizon_bars": decision_data.get(
                            "forecast_horizon_bars", "N/A"
                        ),
                        "justification": decision_data.get("justification", "N/A"),
                        "decision_route": decision_data.get("decision_route", "N/A"),
                        "route_reason": decision_data.get("route_reason", "N/A"),
                        "ai_review_skipped": decision_data.get("ai_review_skipped", "N/A"),
                        "model_role": decision_data.get("model_role", "N/A"),
                        "calibration_mode": decision_data.get("calibration_mode", "N/A"),
                        "calibration_applied": decision_data.get("calibration_applied", "N/A"),
                        "hard_case_score": decision_data.get("hard_case_score", "N/A"),
                        "execution_grade": decision_data.get("execution_grade", "N/A"),
                        "execution_advice": decision_data.get("execution_advice", "N/A"),
                        "signal_quality": decision_data.get("signal_quality", "N/A"),
                        "base_decision": decision_data.get("base_decision", "N/A"),
                        "base_confidence": decision_data.get("base_confidence", "N/A"),
                        "ai_action": decision_data.get("ai_action", "N/A"),
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": final_decision_raw}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": final_decision_raw}

        def _json_block(payload: Dict[str, Any]) -> str:
            if not payload:
                return ""
            try:
                return json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:
                return str(payload)

        def _with_cn_labels(payload: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(payload, dict):
                return payload
            decision_map = {"LONG": "多头", "SHORT": "空头", "NEUTRAL": "中性", "NONE": "无"}
            advice_map = {"execute": "执行", "cautious": "谨慎执行", "skip": "暂不执行", "N/A": "N/A"}
            action_map = {"follow": "跟随基线", "reduce_confidence": "降低置信度", "override": "方向覆写", "N/A": "N/A"}
            quality_map = {"high": "高", "medium": "中", "low": "低", "N/A": "N/A"}
            payload = payload.copy()
            payload["decision_label"] = decision_map.get(str(payload.get("decision", "N/A")).upper(), str(payload.get("decision", "N/A")))
            payload["base_decision_label"] = decision_map.get(str(payload.get("base_decision", "N/A")).upper(), str(payload.get("base_decision", "N/A")))
            payload["execution_advice_label"] = advice_map.get(str(payload.get("execution_advice", "N/A")).lower(), str(payload.get("execution_advice", "N/A")))
            payload["ai_action_label"] = action_map.get(str(payload.get("ai_action", "N/A")).lower(), str(payload.get("ai_action", "N/A")))
            payload["signal_quality_label"] = quality_map.get(str(payload.get("signal_quality", "N/A")).lower(), str(payload.get("signal_quality", "N/A")))
            return payload

        def _build_human_advice(decision: Dict[str, Any], asset_name: str, timeframe: str) -> Dict[str, str]:
            side = str(decision.get("decision", "N/A")).upper()
            execution_advice = str(decision.get("execution_advice", "N/A")).lower()
            confidence = decision.get("confidence", "N/A")
            risk_reward = decision.get("risk_reward_ratio", "N/A")

            if side == "LONG":
                stance = "当前结论偏向多头，可优先关注顺势做多机会。"
            elif side == "SHORT":
                stance = "当前结论偏向空头，应优先防范回落或下破风险。"
            else:
                stance = "当前样本未形成足够稳定的方向优势，应以观察为主。"

            if execution_advice == "execute":
                action = "系统认为信号质量相对完整，若配合个人交易纪律，可考虑执行。"
            elif execution_advice == "cautious":
                action = "系统建议谨慎执行，更适合作为辅助判断信号，需结合位置与风险控制二次确认。"
            elif execution_advice == "skip":
                action = "系统建议暂不执行，当前更适合等待结构进一步明确。"
            else:
                action = "系统未给出明确执行级别，建议以保守观察和小仓位试探为主。"

            summary = (
                f"本次 {asset_name} {timeframe} 分析的综合结论为 {side}，"
                f"置信度 {confidence}，风险收益比参考 {risk_reward}。"
            )
            return {
                "summary": summary,
                "stance": stance,
                "action": action,
            }

        if isinstance(final_decision, dict):
            final_decision = _with_cn_labels(final_decision)

        human_advice = _build_human_advice(
            final_decision if isinstance(final_decision, dict) else {},
            results["asset_name"],
            results["timeframe"],
        )

        return {
            "success": True,
            "run_metadata": results.get("run_metadata", {}),
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "pipeline_summary": results.get("pipeline_summary", {}),
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
            "indicator_features_json": _json_block(indicator_features),
            "pattern_features_json": _json_block(pattern_features),
            "trend_features_json": _json_block(trend_features),
            "risk_features_json": _json_block(risk_features),
            "decision_features_json": _json_block(decision_features),
            "case_context_json": _json_block(case_context),
            "human_advice": human_advice,
        }

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits for a given timeframe."""
        limits = {
            "1m": {"max_days": 7, "description": "1 minute data: max 7 days"},
            "2m": {"max_days": 60, "description": "2 minute data: max 60 days"},
            "5m": {"max_days": 60, "description": "5 minute data: max 60 days"},
            "15m": {"max_days": 60, "description": "15 minute data: max 60 days"},
            "30m": {"max_days": 60, "description": "30 minute data: max 60 days"},
            "60m": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "90m": {"max_days": 60, "description": "90 minute data: max 60 days"},
            "1h": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "4h": {"max_days": 730, "description": "4 hour data: max 730 days"},
            "1d": {"max_days": 730, "description": "1 day data: max 730 days"},
            "5d": {"max_days": 60, "description": "5 day data: max 60 days"},
            "1w": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1wk": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1mo": {"max_days": 730, "description": "1 month data: max 730 days"},
            "3mo": {"max_days": 730, "description": "3 month data: max 730 days"},
        }

        return limits.get(
            timeframe, {"max_days": 730, "description": "Default: max 730 days"}
        )

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "Start date/time must be before end date/time",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"Time range too large. {limits['description']}. Please select a smaller range.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Invalid date/time format: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key by making a simple test call."""
        try:
            # Get provider from config if not provided
            if provider is None:
                provider = self.config.get("agent_llm_provider", "mimo")
            
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                
                # Make a simple test call
                _ = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
                
                provider_name = "OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Anthropic API key is not set. Please update it in the Settings section.",
                    }
                
                client = Anthropic(api_key=api_key)
                
                # Make a simple test call
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                
                provider_name = "Anthropic"
            elif provider == "mimo":
                api_key = os.environ.get("MIMO_API_KEY", "") or self.config.get("mimo_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "鉂?Invalid API Key: The Mimo API key is not set. Please update it in the Settings section.",
                    }

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://token-plan-cn.xiaomimimo.com/v1",
                )
                _ = client.chat.completions.create(
                    model="mimo-v2.5-pro",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )

                provider_name = "Mimo"
            else:  # qwen
                api_key = (
                    os.environ.get("SILICONFLOW_API_KEY", "")
                    or self.config.get("qwen_api_key", "")
                    or self.config.get("siliconflow_api_key", "")
                )
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Qwen API key is not set. Please update it in the Settings section.",
                    }
                
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.siliconflow.cn/v1",
                )
                _ = client.chat.completions.create(
                    model="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )

                provider_name = "SiliconFlow"
            return {"valid": True, "message": f"{provider_name} API key is valid"}

        except Exception as e:
            error_msg = str(e)
            
            # Determine provider name for error messages
            if provider is None:
                provider = self.config.get("agent_llm_provider", "mimo")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "mimo":
                provider_name = "Mimo"
            else:
                provider_name = "SiliconFlow"

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key is invalid or has expired. Please update it in the Settings section.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection.",
                }
            else:
                return {"valid": False, "error": f"❌ API Key Error: {error_msg}"}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            print(f"Error loading custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom asset '{symbol}': {e}")
            return False


# Initialize the analyzer
analyzer = WebTradingAnalyzer()


@app.route("/")
def index():
    """Main landing page - redirect to demo."""
    return render_template("demo_new.html")


@app.route("/demo")
def demo():
    """Demo page with new interface."""
    return render_template("demo_new.html")


@app.route("/output")
def output():
    """Output page with analysis results."""
    # Get results from session or query parameters
    results = request.args.get("results")
    if results:
        try:
            # Handle URL-encoded results
            results = urllib.parse.unquote(results)
            results_data = json.loads(results)
            return render_template("output.html", results=results_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing results: {e}")
            # Fall back to default results
    return render_template("output.html", results={})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        is_multipart = request.content_type and "multipart/form-data" in request.content_type
        if is_multipart:
            data_source = (request.form.get("data_source") or "upload").strip()
            asset = (request.form.get("asset") or "CUSTOM").strip()
            timeframe = (request.form.get("timeframe") or "1h").strip()
            redirect_to_output = (request.form.get("redirect_to_output") or "true").lower() == "true"
        else:
            data = request.get_json() or {}
            data_source = data.get("data_source")
            asset = data.get("asset")
            timeframe = data.get("timeframe")
            redirect_to_output = data.get("redirect_to_output", False)

        display_name = analyzer.asset_mapping.get(asset, asset) or asset

        if data_source == "live":
            if is_multipart:
                start_date = request.form.get("start_date")
                start_time = request.form.get("start_time", "00:00")
                end_date = request.form.get("end_date")
                end_time = request.form.get("end_time", "23:59")
                use_current_time = (request.form.get("use_current_time") or "false").lower() == "true"
            else:
                start_date = data.get("start_date")
                start_time = data.get("start_time", "00:00")
                end_date = data.get("end_date")
                end_time = data.get("end_time", "23:59")
                use_current_time = data.get("use_current_time", False)

            if start_date:
                start_datetime_str = f"{start_date} {start_time}"
                try:
                    start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return jsonify({"error": "Invalid start date/time format."})

                if start_dt > datetime.now():
                    return jsonify({"error": "Start date/time cannot be in the future."})
            else:
                return jsonify({"error": "Start date is required for live analysis."})

            if end_date:
                if use_current_time:
                    end_dt = datetime.now()
                else:
                    end_datetime_str = f"{end_date} {end_time}"
                    try:
                        end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
                    except ValueError:
                        return jsonify({"error": "Invalid end date/time format."})

                    if end_dt > datetime.now():
                        return jsonify({"error": "End date/time cannot be in the future."})
                if end_dt < start_dt:
                    return jsonify({"error": "End date/time cannot be earlier than start date/time."})
            else:
                return jsonify({"error": "End date is required for live analysis."})

            df = analyzer.fetch_yfinance_data_with_datetime(asset, timeframe, start_dt, end_dt)
            if df.empty:
                return jsonify({"error": "No data available for the specified parameters"})
            results = analyzer.run_analysis(df, display_name, timeframe, source_label="Yahoo Finance live feed")

        elif data_source == "upload":
            uploaded_file = request.files.get("market_file")
            if uploaded_file is None:
                return jsonify({"error": "Please upload a CSV or Excel market data file."})
            try:
                df = analyzer.load_uploaded_market_data(uploaded_file)
            except Exception as exc:
                return jsonify({"error": str(exc)})
            if df.empty:
                return jsonify({"error": "The uploaded file did not contain any rows."})
            source_label = f"Uploaded file: {uploaded_file.filename}"
            results = analyzer.run_analysis(df, display_name, timeframe, source_label=source_label)
        else:
            return jsonify({"error": "Unsupported data source. Use live or upload."})

        formatted_results = analyzer.extract_analysis_results(results)
        analyzer.persist_run_log(formatted_results)

        # If redirect is requested, return redirect URL with results
        if redirect_to_output:
            if formatted_results.get("success", False):
                # Create a version without base64 images for URL encoding
                # Base64 images are too large for URL parameters
                url_safe_results = formatted_results.copy()
                url_safe_results["pattern_chart"] = ""  # Remove base64 data
                url_safe_results["trend_chart"] = ""  # Remove base64 data

                # Encode results for URL
                results_json = json.dumps(url_safe_results)
                encoded_results = urllib.parse.quote(results_json)
                run_id = formatted_results.get("run_metadata", {}).get("run_id", "")
                redirect_url = f"/output?run_id={urllib.parse.quote(run_id)}&results={encoded_results}"

                # Store full results (with images) in session or temporary storage
                # For now, we'll pass them back in the response for the frontend to handle
                return jsonify(
                    {
                        "redirect": redirect_url,
                        "full_results": formatted_results,  # Include images in response body
                    }
                )
            else:
                return jsonify(
                    {"error": formatted_results.get("error", "Analysis failed")}
                )

        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/files/<asset>/<timeframe>")
def get_files(asset, timeframe):
    """API endpoint to get available files for an asset/timeframe."""
    try:
        files = analyzer.get_available_files(asset, timeframe)
        file_list = []

        for i, file_path in enumerate(files):
            match = re.search(r"_(\d+)\.csv$", file_path.name)
            file_number = match.group(1) if match else "N/A"
            file_list.append(
                {"index": i, "number": file_number, "name": file_path.name}
            )

        return jsonify({"files": file_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol server-side for persistence."""
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "Symbol required"}), 400

        ok = analyzer.save_custom_asset(symbol)
        if not ok:
            return jsonify({"success": False, "error": "Failed to save symbol"}), 500

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-assets", methods=["GET"])
def custom_assets():
    """Return server-persisted custom assets."""
    try:
        return jsonify({"custom_assets": analyzer.custom_assets or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def get_assets():
    """API endpoint to get available assets."""
    try:
        assets = analyzer.get_available_assets()
        asset_list = []

        for asset in assets:
            asset_list.append(
                {"code": asset, "name": analyzer.asset_mapping.get(asset, asset)}
            )

        # Include server-persisted custom assets at the end
        for custom in analyzer.custom_assets:
            asset_list.append({"code": custom, "name": custom})

        return jsonify({"assets": asset_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/timeframe-limits/<timeframe>")
def get_timeframe_limits(timeframe):
    """API endpoint to get date range limits for a timeframe."""
    try:
        limits = analyzer.get_timeframe_date_limits(timeframe)
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-date-range", methods=["POST"])
def validate_date_range():
    """API endpoint to validate date and time range for a timeframe."""
    try:
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        timeframe = data.get("timeframe")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")

        if not all([start_date, end_date, timeframe]):
            return jsonify({"error": "Missing required parameters"})

        validation = analyzer.validate_date_range(
            start_date, end_date, timeframe, start_time, end_time
        )
        return jsonify(validation)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """API endpoint to update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "mimo")

        if provider not in ["openai", "anthropic", "qwen", "mimo"]:
            return jsonify({"error": "Provider must be 'openai', 'anthropic', 'qwen', or 'mimo'"})

        print(f"Updating provider to: {provider}")

        # Update config in both analyzer and trading_graph
        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
        analyzer.config["vision_llm_provider"] = provider
        analyzer.trading_graph.config["agent_llm_provider"] = provider
        analyzer.trading_graph.config["graph_llm_provider"] = provider
        analyzer.trading_graph.config["vision_llm_provider"] = provider
        
        # Update model names if switching providers
        if provider == "anthropic":
            # Set default Claude models if not already set to Anthropic models
            if not analyzer.config["agent_llm_model"].startswith("claude"):
                analyzer.config["agent_llm_model"] = "claude-haiku-4-5-20251001"
            if not analyzer.config["graph_llm_model"].startswith("claude"):
                analyzer.config["graph_llm_model"] = "claude-haiku-4-5-20251001"
        elif provider == "qwen":
            analyzer.config["agent_llm_model"] = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
            analyzer.config["graph_llm_model"] = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
            analyzer.config["vision_llm_model"] = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
            analyzer.config["qwen_base_url"] = "https://api.siliconflow.cn/v1"
            analyzer.config["qwen_api_env_name"] = "SILICONFLOW_API_KEY"
        elif provider == "mimo":
            analyzer.config["agent_llm_model"] = "mimo-v2.5-pro"
            analyzer.config["graph_llm_model"] = "mimo-v2.5-pro"
            analyzer.config["vision_llm_model"] = "mimo-v2.5-pro"
            analyzer.config["qwen_base_url"] = "https://token-plan-cn.xiaomimimo.com/v1"
            analyzer.config["qwen_api_env_name"] = "MIMO_API_KEY"
        else:
            # Set default OpenAI models if not already set to OpenAI models
            if analyzer.config["agent_llm_model"].startswith(("claude", "qwen", "mimo-")):
                analyzer.config["agent_llm_model"] = "gpt-4o-mini"
            if analyzer.config["graph_llm_model"].startswith(("claude", "qwen", "mimo-")):
                analyzer.config["graph_llm_model"] = "gpt-4o"
            if analyzer.config["vision_llm_model"].startswith(("claude", "qwen", "mimo-")):
                analyzer.config["vision_llm_model"] = "gpt-4o"
        
        analyzer.trading_graph.config.update(analyzer.config)

        # Refresh the trading graph with new provider
        analyzer.trading_graph.refresh_llms()

        print(f"Provider updated to {provider} successfully")
        print(f"graph_llm_model updated to {analyzer.config['graph_llm_model']} successfully")
        print(f"agent_llm updated to {analyzer.config['agent_llm_model']} successfully")
        return jsonify({"success": True, "message": f"Provider updated to {provider}"})

    except Exception as e:
        print(f"Error in update_provider: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """API endpoint to update API key for OpenAI or Anthropic."""
    try:
        data = request.get_json()
        new_api_key = data.get("api_key")
        provider = data.get("provider", "mimo")

        if not new_api_key:
            return jsonify({"error": "API key is required"})

        if provider not in ["openai", "anthropic", "qwen", "mimo"]:
            return jsonify({"error": "Provider must be 'openai', 'anthropic', 'qwen', or 'mimo'"})

        print(f"Updating {provider} API key to: {new_api_key[:8]}...{new_api_key[-4:]}")

        # Update the environment variable
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = new_api_key
            analyzer.config["api_key"] = new_api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = new_api_key
            analyzer.config["anthropic_api_key"] = new_api_key
        elif provider == "qwen":
            os.environ["SILICONFLOW_API_KEY"] = new_api_key
            analyzer.config["qwen_api_key"] = new_api_key
            analyzer.config["siliconflow_api_key"] = new_api_key
        elif provider == "mimo":
            os.environ["MIMO_API_KEY"] = new_api_key
            analyzer.config["mimo_api_key"] = new_api_key

        # Update the API key in the trading graph
        analyzer.trading_graph.update_api_key(new_api_key, provider=provider)

        print(f"{provider} API key updated successfully")
        return jsonify({"success": True, "message": f"{provider.capitalize()} API key updated successfully"})

    except Exception as e:
        print(f"Error in update_api_key: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """API endpoint to check if API key is set for a provider."""
    try:
        provider = request.args.get("provider", "mimo")
        
        # First check environment variables
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("api_key", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("anthropic_api_key", "")
        elif provider == "qwen":
            api_key = (
                os.environ.get("SILICONFLOW_API_KEY", "")
                or (analyzer.config.get("qwen_api_key", "") if hasattr(analyzer, "config") else "")
                or (analyzer.config.get("siliconflow_api_key", "") if hasattr(analyzer, "config") else "")
            )
        elif provider == "mimo":
            api_key = (
                os.environ.get("MIMO_API_KEY", "")
                or (analyzer.config.get("mimo_api_key", "") if hasattr(analyzer, "config") else "")
            )
        else:
            api_key = ""
        
        if api_key and api_key != "your-openai-api-key-here" and api_key != "":
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
    except Exception as e:
        print(f"Error in get_api_key_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "has_key": False})


@app.route("/api/images/<image_type>")
def get_image(image_type):
    """API endpoint to serve generated images."""
    try:
        if image_type == "pattern":
            image_path = "kline_chart.png"
        elif image_type == "trend":
            image_path = "trend_graph.png"
        elif image_type == "pattern_chart":
            image_path = "pattern_chart.png"
        elif image_type == "trend_chart":
            image_path = "trend_chart.png"
        else:
            return jsonify({"error": "Invalid image type"})

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"})

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-api-key", methods=["POST"])
def validate_api_key():
    """API endpoint to validate the current API key."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider") or analyzer.config.get("agent_llm_provider", "mimo")
        validation = analyzer.validate_api_key(provider=provider)
        return jsonify(validation)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets from the assets folder."""
    try:
        return send_file(f"assets/{filename}")
    except FileNotFoundError:
        return jsonify({"error": "Asset not found"}), 404


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Create static directory if it doesn't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    app.run(debug=True, host="127.0.0.1", port=5000)
