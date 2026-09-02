from ollama import Client
from .config import settings

client = Client(host=settings.ollama_host)


def embed(text):
    try:
        return client.embed(model=settings.embedding_model, input=text)["embeddings"][0]
    except Exception:
        return client.embeddings(model=settings.embedding_model, prompt=text)[
            "embedding"
        ]


def stream_chat(messages):
    return client.chat(model=settings.llm_model, messages=messages, stream=True)


def health_check():
    return client.list()
