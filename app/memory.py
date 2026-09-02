from concurrent.futures import ThreadPoolExecutor
from .config import settings
from . import database
from .llm import embed
from .vector_store import VectorStore

store = VectorStore()
executor = ThreadPoolExecutor(max_workers=settings.background_workers)
KEYS = {
    "my name",
    "i am",
    "i live",
    "i like",
    "i prefer",
    "my favorite",
    "my project",
    "my goal",
    "remember",
    "working on",
}


def classify_importance(text, explicit=False):
    t = text.lower()
    if explicit:
        return 0.95, "explicit_memory"
    if any(k in t for k in KEYS):
        return 0.75, "fact"
    if len(t.split()) < 4:
        return 0.05, "trivial"
    return 0.3, "conversation"


def should_retrieve(q, force=False):
    if force:
        return True
    return any(
        x in q.lower()
        for x in (
            "my ",
            "i ",
            "remember",
            "previous",
            "earlier",
            "before",
            "we discussed",
            "project",
            "goal",
            "preference",
            "favorite",
            "name",
        )
    )


def _index(r):
    text = f"User: {r['prompt']}\nAssistant: {r['response']}"
    try:
        store.upsert(
            r["id"],
            embed(text),
            text,
            {"memory_type": r["memory_type"], "importance": float(r["importance"])},
        )
        database.mark_indexed(r["id"], True)
    except Exception:
        database.mark_indexed(r["id"], False)
        raise


def store_conversation(prompt, response, explicit=False):
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


def retrieve(q, force=False):
    if not should_retrieve(q, force):
        return []
    results = [
        r
        for r in store.search(embed(q), settings.top_k)
        if r["distance"] <= settings.memory_distance_threshold
    ]
    return sorted(
        results,
        key=lambda r: (r["distance"], -float(r["metadata"].get("importance", 0))),
    )


def forget(i):
    d = database.delete_record(i)
    if d:
        store.delete(i)
    return bool(d)


def reindex():
    rs = database.get_unindexed(100000)
    fs = [executor.submit(_index, r) for r in rs]
    for f in fs:
        f.result()
    return len(rs)
