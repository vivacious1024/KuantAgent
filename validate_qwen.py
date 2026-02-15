
import os
import sys

# Try to import Qwen class
try:
    from langchain_qwq import ChatQwen
    print("Successfully imported ChatQwen")
except ImportError:
    print("Error: Could not import ChatQwen. 'langchain_qwq' module might be missing.")
    sys.exit(1)

# Configuration from user (as recently set)
API_KEY = "sk-77d2f3305b2645038df708057391a63c"
# MODEL_NAME = "qwen-max"  # Using the standard name first
MODEL_NAME = "qwen3-omni-flash-2025-12-01" # User's custom name

print(f"Testing Qwen API with key: {API_KEY[:5]}...{API_KEY[-4:]}")
print(f"Model: {MODEL_NAME}")

try:
    # Initialize ChatQwen
    llm = ChatQwen(
        model=MODEL_NAME,
        api_key=API_KEY,
        streaming=False
    )
    
    # Test invocation
    print("Sending test message...")
    response = llm.invoke("Hello, say 'API connection successful' if you can hear me.")
    
    print("\n--- Response ---")
    print(response.content)
    print("----------------")
    print("\nSUCCESS: API Key and Model are working!")
    
except Exception as e:
    print(f"\nFAILURE: {str(e)}")
