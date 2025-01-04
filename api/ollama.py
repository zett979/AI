import requests
import ollama
from ollama import chat, ChatResponse

CHAT_URL = "http://localhost:11434/api/"
MODEL = "llama3.2"


def getModelName():
    try:
        test = ollama.list()
        MODEL = test.models[0].model.split(":")[0]
        return MODEL
    except Exception as e:
        print("Error fetching models", e)


def chatToModel(prompt: str):
    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        return response.response
    except Exception as e:
        print("Error fetching response", e)
