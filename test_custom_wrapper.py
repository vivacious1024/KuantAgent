
from custom_qwen import CustomChatQwen
from langchain_core.messages import HumanMessage
import sys

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

api_key = "sk-77d2f3305b2645038df708057391a63c"
model = "qwen3-omni-flash-2025-12-01"

print(f"Testing CustomChatQwen with model: {model}")

try:
    chat = CustomChatQwen(model_name=model, dashscope_api_key=api_key)
    response = chat.invoke([HumanMessage(content="Hello!")])
    print("Response Content:", response.content)
except Exception as e:
    print("Custom Wrapper Call Failed:", e)
