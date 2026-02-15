
import requests
import json
import os

api_key = "sk-77d2f3305b2645038df708057391a63c"
model = "qwen3-omni-flash-2025-12-01"

print(f"Testing API Key: {api_key[:5]}...***")
print(f"Target Model: {model}")

url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": model,
    "input": {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
