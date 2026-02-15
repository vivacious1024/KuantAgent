
from langchain_community.chat_models import ChatTongyi
import os
import sys

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

api_key = "sk-77d2f3305b2645038df708057391a63c"
os.environ["DASHSCOPE_API_KEY"] = api_key
model = "qwen3-omni-flash-2025-12-01"

print(f"Testing ChatTongyi with model: {model}")

try:
    chat = ChatTongyi(model=model)
    response = chat.invoke("Hello!")
    print("Response Content:", response.content)
except Exception as e:
    print("ChatTongyi Call Failed:", e)
