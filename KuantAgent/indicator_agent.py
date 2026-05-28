"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC,
Stochastic, and Williams %R.
"""

import copy
import json
import time

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


def invoke_with_retry(call_fn, *args, retries=4, wait_sec=6):
    """Retry transient model-call failures so one backend hiccup does not kill a long benchmark run."""

    for attempt in range(retries):
        try:
            return call_fn(*args)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
            print(
                f"Indicator agent transient error: {exc}. "
                f"Retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        except Exception as exc:
            print(
                f"Indicator agent error: {exc}. "
                f"Retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Indicator agent exceeded maximum retries")


def _build_fallback_indicator_report(indicator_features, time_frame):
    """Return a deterministic fallback report when the LLM path is unavailable."""

    rsi_state = indicator_features.get("rsi_state", "unknown")
    macd_cross = indicator_features.get("macd_cross", "none")
    macd_hist = indicator_features.get("macd_hist", 0.0)
    roc_value = indicator_features.get("roc", 0.0)
    stoch_state = indicator_features.get("stoch_state", "unknown")
    willr_state = indicator_features.get("willr_state", "unknown")
    momentum_bias = indicator_features.get("momentum_bias", "neutral")
    divergence = indicator_features.get("rsi_divergence", "none")

    return (
        f"Fallback indicator analysis for {time_frame} data. "
        f"Momentum bias={momentum_bias}; RSI state={rsi_state}; "
        f"MACD cross={macd_cross}; MACD histogram={macd_hist}; ROC={roc_value}; "
        f"Stochastic state={stoch_state}; Williams %R state={willr_state}; "
        f"RSI divergence={divergence}. "
        "The report was generated from deterministic indicator features because the indicator agent LLM path exceeded retries."
    )


def create_indicator_agent(llm, toolkit):
    """Create the indicator-analysis node used inside the LangGraph workflow."""

    def indicator_agent_node(state):
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        indicator_features = state.get("indicator_features", {})

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                    "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                    "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                    "Use them by providing appropriate arguments like `kline_data` and the respective periods.\n\n"
                    f"The OHLC data provided is from a {time_frame} interval and reflects recent market behavior. "
                    "You must interpret this data quickly and accurately.\n\n"
                    "A structured feature summary from the algorithm analysis layer is also provided. "
                    "Use it as a quick reference, then validate or enrich it with tool calls when needed.\n\n"
                    "Structured indicator features:\n{indicator_features}\n\n"
                    "Here is the OHLC data:\n{kline_data}.\n\n"
                    "Call necessary tools, and analyze the results.\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(
            kline_data=json.dumps(state["kline_data"], indent=2),
            indicator_features=json.dumps(indicator_features, indent=2),
        )

        chain = prompt | llm.bind_tools(tools)
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]

        try:
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)
        except Exception:
            fallback_report = _build_fallback_indicator_report(indicator_features, time_frame)
            messages.append(HumanMessage(content=fallback_report))
            return {
                "messages": messages,
                "indicator_report": fallback_report,
            }

        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(tool for tool in tools if tool.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=json.dumps(tool_result),
                    )
                )

        max_iterations = 5
        iteration = 0
        final_response = None

        while iteration < max_iterations:
            iteration += 1
            try:
                final_response = invoke_with_retry(chain.invoke, messages)
                messages.append(final_response)
            except Exception:
                fallback_report = _build_fallback_indicator_report(indicator_features, time_frame)
                messages.append(HumanMessage(content=fallback_report))
                return {
                    "messages": messages,
                    "indicator_report": fallback_report,
                }

            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break

            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(tool for tool in tools if tool.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=json.dumps(tool_result),
                    )
                )

        if final_response:
            report_content = final_response.content
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                for msg in reversed(messages):
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and isinstance(msg.content, str)
                        and msg.content.strip()
                        and not hasattr(msg, "tool_calls")
                    ):
                        report_content = msg.content
                        break
        else:
            report_content = "Indicator analysis completed, but no detailed report was generated."

        return {
            "messages": messages,
            "indicator_report": report_content if report_content else "Indicator analysis completed.",
        }

    return indicator_agent_node
