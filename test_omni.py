
import requests
import json
import sys

# Force UTF-8 for output
sys.stdout.reconfigure(encoding='utf-8')

api_key = "sk-77d2f3305b2645038df708057391a63c"
model = "qwen3-omni-flash-2025-12-01"

print(f"Testing Model: {model}")

# URL 1: Standard Text Generation Endpoint
url_text = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# URL 2: Multimodal Endpoint (often needed for VL/Omni models)
url_multimodal = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "X-DashScope-WorkSpace": "default" # Sometimes needed
}

# Payload
data = {
    "model": model,
    "input": {
        "messages": [
            {"role": "user", "content": "Hello, are you working?"}
        ]
    }
}

print("\n--- Attempt 1: Text Generation Endpoint ---")
try:
    resp = requests.post(url_text, headers=headers, json=data)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:200]}...") # Truncate for readability
except Exception as e:
    print(f"Error: {e}")

print("\n--- Attempt 2: Multimodal Generation Endpoint ---")
try:
    resp = requests.post(url_multimodal, headers=headers, json=data)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:200]}...")
except Exception as e:
    print(f"Error: {e}")
