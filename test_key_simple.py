
import requests
import json
import sys

# Force UTF-8 for output
sys.stdout.reconfigure(encoding='utf-8')

api_key = "sk-77d2f3305b2645038df708057391a63c"
# Test with a standard model first to validate the Key
model = "qwen-turbo" 

print(f"Testing API Key: {api_key[:5]}...***")
print(f"Testing with known good model: {model}")

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
    print(f"Status: {response.status_code}")
    print(f"Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
