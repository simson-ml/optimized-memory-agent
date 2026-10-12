from concurrent.futures import ThreadPoolExecutor
import re
from .config import settings
from . import database
from .llm import embed, embed_batch
from .vector_store import VectorStore

store = VectorStore()
executor = ThreadPoolExecutor(max_workers=settings.background_workers)

# Regex matching words and phrases that signify memory or user identity/preferences
RETRIEVE_PATTERN = re.compile(
    r"\b(i|my|mine|me|myself|we|our|ours)\b|"
    r"\b(remember|recall|previous|earlier|before|discussed|conversation|history)\b|"
    r"\b(project|goal|preference|prefer|favorite|favourite|hobby|like|dislike)\b|"
    r"\b(name|age|live|location|job|work|working|identity|who)\b",
    re.IGNORECASE,
)

FACT_PATTERN = re.compile(
    r"\b(my name is|i am|i live in|i like|i prefer|my favorite|my project is|my goal is|remember that|working on)\b",
    re.IGNORECASE,
)

QUESTION_PATTERN = re.compile(
    r"(\?|\b(what|who|where|when|why|how|guess|is|are|can|do|did|does|tell me)\b)",
    re.IGNORECASE,
)


def classify_importance(text: str, explicit: bool = False) -> tuple[float, str]:
    if explicit:
        return 0.95, "explicit_memory"
    # If the user is asking a question, it is an inquiry, not a statement of fact
    if QUESTION_PATTERN.search(text) and text.strip().endswith("?"):
        return 0.20, "conversation"
    if FACT_PATTERN.search(text):
        return 0.90, "fact"
    if len(text.split()) < 4:
        return 0.05, "trivial"
    return 0.30, "conversation"


def should_retrieve(q: str, force: bool = False) -> bool:
    if force:
        return True
    return bool(RETRIEVE_PATTERN.search(q))


def _format_memory_text(r: dict) -> str:
    is_explicit = (
        r.get("memory_type") == "explicit_memory"
        or r.get("source") == "explicit_user_memory"
        or r.get("response") == "Explicit memory stored."
    )
    if is_explicit:
        return f"Fact: {r['prompt']}"
    return f"User: {r['prompt']}\nAssistant: {r['response']}"


def _index(r: dict):
    text = _format_memory_text(r)
    try:
        store.upsert(
            r["id"],
            embed(text),
            text,
            {
                "memory_type": r.get("memory_type", "conversation"),
                "importance": float(r.get("importance", 0.3)),
            },
        )
        database.mark_indexed(r["id"], True)
    except Exception as ex:
        database.mark_indexed(r["id"], False)
        print(f"\n[Warning] Background indexing failed for ID {r.get('id')}: {ex}")


def store_conversation(prompt: str, response: str, explicit: bool = False) -> int:
    imp, typ = classify_importance(prompt, explicit)
    r = database.insert_record(
        prompt,
        response,
        typ,
        imp,
        "explicit_user_memory" if explicit else "user_conversation",
    )
    executor.submit(_index, r)
    return r["id"]


def retrieve(q: str, force: bool = False) -> list[dict]:
    if not should_retrieve(q, force):
        return []

    query_emb = embed(q)
    total_vectors = store.count()
    if total_vectors == 0:
        return []

    # Expand candidate pool to avoid question-matching crowding out factual answers
    pool_size = min(max(settings.top_k * 4, 25), total_vectors)
    candidates = store.search(query_emb, pool_size)
    if not candidates:
        return []

    # Filter out extreme outliers by raw distance threshold
    filtered = [
        c for c in candidates
        if c["distance"] <= settings.memory_distance_threshold
    ]

    # Deduplicate near-identical memory prompts
    seen = set()
    unique = []
    for c in filtered:
        first_line = c["document"].split("\n")[0].strip().lower()
        if first_line not in seen:
            seen.add(first_line)
            unique.append(c)

    # Rank by composite score: reward low distance and high importance
    ranked = sorted(
        unique,
        key=lambda r: r["distance"] - (0.25 * float(r["metadata"].get("importance", 0.2))),
    )

    return ranked[: settings.top_k]


def forget(i: int) -> bool:
    d = database.delete_record(i)
    if d:
        store.delete(i)
    return bool(d)


def reindex(batch_size: int = 50) -> int:
    """Batch reindexes all unindexed records."""
    unindexed = database.get_unindexed(100000)
    if not unindexed:
        return 0
    return _process_records_in_batches(unindexed, batch_size)


def rebuild_index(batch_size: int = 50) -> int:
    """Rebuilds the entire ChromaDB index directly from PostgreSQL."""
    all_ids = database.get_all_record_ids()
    if not all_ids:
        return 0
    records = database.get_records_by_ids(all_ids)
    return _process_records_in_batches(records, batch_size)


def resync_all(batch_size: int = 50) -> dict:
    """
    Bidirectional reconciliation between PostgreSQL and ChromaDB.
    Finds and re-indexes:
    1. Missing vectors in ChromaDB that were marked indexed=TRUE.
    2. Unindexed records in PostgreSQL (indexed=FALSE).
    """
    pg_ids = set(database.get_all_record_ids())
    chroma_ids = store.get_all_ids()

    missing_ids = [pid for pid in pg_ids if str(pid) not in chroma_ids]
    unindexed_records = database.get_unindexed(100000)
    unindexed_ids = {r["id"] for r in unindexed_records}

    all_needed_ids = list(set(missing_ids) | unindexed_ids)
    if not all_needed_ids:
        return {"repaired": 0, "total_synced": len(chroma_ids)}

    records = database.get_records_by_ids(all_needed_ids)
    synced = _process_records_in_batches(records, batch_size)
    return {"repaired": len(missing_ids), "total_synced": synced}


def _process_records_in_batches(records: list[dict], batch_size: int = 50) -> int:
    total = len(records)
    for i in range(0, total, batch_size):
        chunk = records[i : i + batch_size]
        ids = [r["id"] for r in chunk]
        texts = [_format_memory_text(r) for r in chunk]
        metadatas = [
            {
                "memory_type": r.get("memory_type", "conversation"),
                "importance": float(r.get("importance", 0.3)),
            }
            for r in chunk
        ]

        embeddings = embed_batch(texts)
        store.upsert_batch(ids, embeddings, texts, metadatas)
        database.mark_batch_indexed(ids, True)
    return total
