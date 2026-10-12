from functools import lru_cache
import httpx
from ollama import Client
from .config import settings

# Generous connection timeout; read=None prevents premature abortion during heavy CPU prompt evaluation
_client_timeout = (
    httpx.Timeout(connect=15.0, read=None, write=30.0, pool=15.0)
    if settings.ollama_timeout is None
    else httpx.Timeout(connect=15.0, read=settings.ollama_timeout, write=30.0, pool=15.0)
)

client = Client(host=settings.ollama_host, timeout=_client_timeout)


@lru_cache(maxsize=settings.embed_cache_size)
def _cached_embed_tuple(text: str) -> tuple[float, ...]:
    try:
        res = client.embed(
            model=settings.embedding_model,
            input=text,
            keep_alive=settings.ollama_keep_alive,
        )
        return tuple(res["embeddings"][0])
    except Exception:
        res = client.embeddings(
            model=settings.embedding_model,
            prompt=text,
            keep_alive=settings.ollama_keep_alive,
        )
        return tuple(res["embedding"])


def embed(text: str) -> list[float]:
    """Generates or retrieves a cached embedding vector for a given text."""
    return list(_cached_embed_tuple(text))


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for multiple texts in a single batch pass."""
    if not texts:
        return []
    try:
        res = client.embed(
            model=settings.embedding_model,
            input=texts,
            keep_alive=settings.ollama_keep_alive,
        )
        embeddings = res["embeddings"]
        for t, emb in zip(texts, embeddings):
            _cached_embed_tuple.__wrapped__(t) if hasattr(_cached_embed_tuple, "__wrapped__") else None
        return embeddings
    except Exception:
        return [embed(t) for t in texts]


def stream_chat(messages: list[dict]):
    """Streams chat responses with CPU-tuned threading, context limits, and keep-alive."""
    return client.chat(
        model=settings.llm_model,
        messages=messages,
        stream=True,
        keep_alive=settings.ollama_keep_alive,
        options={
            "num_thread": settings.ollama_num_thread,
            "num_ctx": settings.ollama_num_ctx,
        },
    )


def health_check():
    return client.list()


def warmup():
    """Warms up models in memory to eliminate cold start latency on first user prompt."""
    try:
        embed("warmup")
    except Exception:
        pass
