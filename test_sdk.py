
import dashscope
from dashscope import MultiModalConversation
import sys

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

api_key = "sk-77d2f3305b2645038df708057391a63c"
dashscope.api_key = api_key
model = "qwen3-omni-flash-2025-12-01"

print(f"Testing DashScope SDK with model: {model}")

messages = [
    {
        "role": "user",
        "content": [
            {"text": "Hello, can you see this?"}
        ]
    }
]

try:
    response = MultiModalConversation.call(model=model, messages=messages)
    print("Response Status:", response.status_code)
    print("Response Output:", response.output)
    print("Full Response:", response)
except Exception as e:
    print("SDK Call Failed:", e)
