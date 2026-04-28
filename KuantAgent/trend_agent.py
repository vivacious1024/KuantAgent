"""
Agent for trend analysis in the KuantAgent workflow.

This version treats structured trend and risk features as the main input.
Trend images remain optional supporting evidence.
"""

import json
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


def invoke_with_retry(call_fn, *args, retries=3, wait_sec=6):
    """Retry model calls to absorb rate limits and transient failures."""

    for attempt in range(retries):
        try:
            return call_fn(*args)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError):
            print(
                f"Trend agent transient error, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"Trend agent error: {exc}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                ,
                file=sys.stderr,
                flush=True,
            )
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Trend agent exceeded maximum retries")


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """Create the trend-analysis node used by the graph."""

    def trend_agent_node(state):
        time_frame = state["time_frame"]
        trend_features = state.get("trend_features", {})
        risk_features = state.get("risk_features", {})
        indicator_features = state.get("indicator_features", {})
        decision_features = state.get("decision_features", {})
        trend_image_b64 = state.get("trend_image")

        prompt_text = (
            f"You are a short-horizon trend analyst working on {time_frame} market data.\n\n"
            "Your primary evidence is the structured trend feature object extracted by deterministic algorithms. "
            "Use risk features to judge whether the trend environment is stable or fragile.\n\n"
            "Tasks:\n"
            "1. Describe the current trend direction and strength.\n"
            "2. Explain how close price is to support and resistance.\n"
            "3. Judge whether the environment favors continuation, reversal, compression breakout, or sideways behavior.\n"
            "4. Explain whether price is breaking out, testing support, testing resistance, or staying inside range.\n"
            "5. Mention any risk warnings from volatility, weak trend structure, fragile breakout quality, or false breakout risk.\n"
            "6. Mention whether repeated support or resistance touches make the level more meaningful.\n\n"
            f"Trend features:\n{json.dumps(trend_features, indent=2)}\n\n"
            f"Risk features:\n{json.dumps(risk_features, indent=2)}\n\n"
            f"Indicator context:\n{json.dumps(indicator_features, indent=2)}\n\n"
            f"Decision context:\n{json.dumps(decision_features, indent=2)}\n\n"
            "Return a concise report covering trend direction, structure quality, location in range, breakout state, and short-term directional bias."
        )

        if trend_image_b64:
            image_prompt = [
                {"type": "text", "text": prompt_text + "\n\nAn optional trend chart is attached as secondary context."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]
            human_msg = HumanMessage(content=image_prompt)
            messages = [
                SystemMessage(
                    content="You are a disciplined trend analyst. Prioritize structured algorithmic evidence over image intuition."
                ),
                human_msg,
            ]
            try:
                response = invoke_with_retry(graph_llm.invoke, messages)
            except Exception as exc:
                if "at least one message" in str(exc).lower():
                    response = invoke_with_retry(graph_llm.invoke, [human_msg])
                else:
                    raise
        else:
            messages = [
                SystemMessage(
                    content="You are a disciplined trend analyst. Prioritize structured algorithmic evidence over image intuition."
                ),
                HumanMessage(content=prompt_text),
            ]
            response = invoke_with_retry(graph_llm.invoke, messages)

        return {
            "messages": state.get("messages", []) + messages + [response],
            "trend_report": response.content,
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png" if trend_image_b64 else None,
            "trend_image_description": (
                "Optional trend chart with support and resistance lines"
                if trend_image_b64
                else None
            ),
        }

    return trend_agent_node
