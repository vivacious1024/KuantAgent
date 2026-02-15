# DEFAULT_CONFIG = {
#     "agent_llm_model": "gpt-4o-mini",
#     "graph_llm_model": "gpt-4o",
#     "agent_llm_provider": "openai",  # "openai", "anthropic", or "qwen"
#     "graph_llm_provider": "openai",  # "openai", "anthropic", or "qwen"
#     "agent_llm_temperature": 0.1,
#     "graph_llm_temperature": 0.1,
#     "api_key": "",  # OpenAI API key
#     "anthropic_api_key": "",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
#     "qwen_api_key": "",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
# }
DEFAULT_CONFIG = {
    # ...
    "agent_llm_model": "qwen3-omni-flash-2025-12-01",     # 修改模型名
    "graph_llm_model": "qwen3-omni-flash-2025-12-01",     # 修改模型名
    "agent_llm_provider": "qwen",      # 修改服务商
    "graph_llm_provider": "qwen",      # 修改服务商
    # ...
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "",  # OpenAI API key
    "anthropic_api_key": "",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-77d2f3305b2645038df708057391a63c",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
}