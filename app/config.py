from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


def _i(k: str, d: int) -> int:
    val = os.getenv(k)
    if val is None or not val.strip():
        return d
    try:
        return int(val.strip().strip("'\""))
    except ValueError:
        return d


def _f(k: str, d: float) -> float:
    val = os.getenv(k)
    if val is None or not val.strip():
        return d
    try:
        return float(val.strip().strip("'\""))
    except ValueError:
        return d


@dataclass(frozen=True)
class Settings:
    # Database settings
    dbname: str = os.getenv("POSTGRES_DB", "memory_agent").strip()
    user: str = os.getenv("POSTGRES_USER", "postgres").strip()
    password: str = os.getenv("POSTGRES_PASSWORD", "root").strip().strip("'\"")
    host: str = os.getenv("POSTGRES_HOST", "localhost").strip()
    port: int = _i("POSTGRES_PORT", 5432)
    pg_pool_min: int = _i("POSTGRES_POOL_MIN", 1)
    pg_pool_max: int = _i("POSTGRES_POOL_MAX", 10)
    pg_pool_timeout: float = _f("POSTGRES_POOL_TIMEOUT", 30.0)

    # Ollama settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    llm_model: str = os.getenv("LLM_MODEL", "llama3:latest").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest").strip()
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
    ollama_num_thread: int = _i("OLLAMA_NUM_THREAD", 4)
    # None (or 0) disables HTTP read timeout so long streaming evaluations never abort
    ollama_timeout: float | None = None if _f("OLLAMA_TIMEOUT", 0.0) <= 0 else _f("OLLAMA_TIMEOUT", 0.0)
    ollama_num_ctx: int = _i("OLLAMA_NUM_CTX", 2048)

    # ChromaDB & Memory settings
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma_memory").strip()
    max_recent_messages: int = _i("MAX_RECENT_MESSAGES", 6)
    top_k: int = _i("TOP_K", 3)
    max_memory_chars: int = _i("MAX_MEMORY_CHARS", 200)
    # Cosine distance threshold: matches docs SIMILARITY_THRESHOLD or MEMORY_DISTANCE_THRESHOLD
    memory_distance_threshold: float = _f(
        "MEMORY_DISTANCE_THRESHOLD",
        _f("SIMILARITY_THRESHOLD", 0.75),
    )
    background_workers: int = _i("BACKGROUND_WORKERS", 2)
    embed_cache_size: int = _i("EMBED_CACHE_SIZE", 1024)


settings = Settings()
