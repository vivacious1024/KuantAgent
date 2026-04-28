"""
Agent for pattern analysis in the KuantAgent workflow.

This version treats structured pattern features as the primary evidence.
Chart images remain optional supporting context rather than the main path.
"""

import json
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


def invoke_with_retry(call_fn, *args, retries=3, wait_sec=6):
    """Retry model calls to reduce transient API failures."""

    for attempt in range(retries):
        try:
            return call_fn(*args)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError):
            print(
                f"Pattern agent transient error, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"Pattern agent error: {exc}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                ,
                file=sys.stderr,
                flush=True,
            )
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Pattern agent exceeded maximum retries")


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """Create the pattern-analysis node used by the graph."""

    def pattern_agent_node(state):
        time_frame = state["time_frame"]
        pattern_features = state.get("pattern_features", {})
        indicator_features = state.get("indicator_features", {})
        trend_features = state.get("trend_features", {})
        pattern_image_b64 = state.get("pattern_image")

        prompt_text = (
            f"You are a short-horizon trading pattern analyst working on {time_frame} market data.\n\n"
            "Your primary evidence is the structured pattern feature object extracted by deterministic algorithms. "
            "Do not invent visual structures that are not supported by the provided features.\n\n"
            "Tasks:\n"
            "1. Determine whether a meaningful price pattern exists.\n"
            "2. Explain whether the pattern implies bullish, bearish, or neutral bias.\n"
            "3. Judge whether the pattern is mature enough to influence trading decisions now.\n"
            "4. Judge whether a breakout or neckline confirmation is already present.\n"
            "5. Use indicator and trend context only as supporting confirmation.\n\n"
            f"Pattern features:\n{json.dumps(pattern_features, indent=2)}\n\n"
            f"Indicator context:\n{json.dumps(indicator_features, indent=2)}\n\n"
            f"Trend context:\n{json.dumps(trend_features, indent=2)}\n\n"
            "When writing the report, explicitly mention:\n"
            "- whether the pattern is completed,\n"
            "- whether breakout confirmation exists,\n"
            "- whether the breakout looks reliable or still vulnerable to failure,\n"
            "- whether the setup is actionable now or still premature.\n\n"
            "Return a concise report covering: detected pattern, bias, confidence, maturity, and actionability."
        )

        if pattern_image_b64:
            image_prompt = [
                {"type": "text", "text": prompt_text + "\n\nAn optional candlestick chart is attached as secondary context."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]
            human_msg = HumanMessage(content=image_prompt)
            messages = [
                SystemMessage(
                    content="You are a disciplined trading pattern analyst. Prioritize structured algorithmic evidence over image intuition."
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
                    content="You are a disciplined trading pattern analyst. Prioritize structured algorithmic evidence over image intuition."
                ),
                HumanMessage(content=prompt_text),
            ]
            response = invoke_with_retry(graph_llm.invoke, messages)

        return {
            "messages": state.get("messages", []) + messages + [response],
            "pattern_report": response.content,
        }

    return pattern_agent_node
