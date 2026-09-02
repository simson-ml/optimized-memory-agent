from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


def _i(k, d):
    return int(os.getenv(k, d))


def _f(k, d):
    return float(os.getenv(k, d))


@dataclass(frozen=True)
class Settings:
    dbname: str = os.getenv("POSTGRES_DB", "memory_agent")
    user: str = os.getenv("POSTGRES_USER", "root")
    password: str = os.getenv("POSTGRES_PASSWORD", "root")
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = _i("POSTGRES_PORT", 5432)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma_memory")
    max_recent_messages: int = _i("MAX_RECENT_MESSAGES", 10)
    top_k: int = _i("TOP_K", 5)
    memory_distance_threshold: float = _f("MEMORY_DISTANCE_THRESHOLD", 0.75)
    background_workers: int = _i("BACKGROUND_WORKERS", 2)


settings = Settings()
