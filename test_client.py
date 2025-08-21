import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_chat():
    data = {
        "message": "What is the capital of France?",
        "system_prompt": "You are a helpful geography assistant."
    }
    response = requests.post(f"{BASE_URL}/chat", json=data)
    print("Chat Response:", response.json())

def test_summarize():
    data = {
        "message": "Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry. Machine learning, deep learning, and neural networks are key components of AI."
    }
    response = requests.post(f"{BASE_URL}/summarize", json=data)
    print("Summary Response:", response.json())

if __name__ == "__main__":
    print("Testing LangChain + OpenAI + FastAPI")
    test_health()
    print("\n" + "="*50 + "\n")
    test_chat()
    print("\n" + "="*50 + "\n")
    test_summarize()