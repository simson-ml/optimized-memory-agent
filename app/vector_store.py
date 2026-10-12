import threading
import chromadb
from .config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            "memories", metadata={"hnsw:space": "cosine"}
        )
        self._lock = threading.Lock()

    def upsert(self, i, e, d, m):
        with self._lock:
            self.collection.upsert(
                ids=[str(i)], embeddings=[e], documents=[d], metadatas=[m]
            )

    def upsert_batch(self, ids, embeddings, documents, metadatas):
        if not ids:
            return
        str_ids = [str(i) for i in ids]
        with self._lock:
            self.collection.upsert(
                ids=str_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def delete(self, i):
        with self._lock:
            try:
                self.collection.delete(ids=[str(i)])
            except Exception:
                pass

    def count(self) -> int:
        return self.collection.count()

    def get_all_ids(self) -> set[str]:
        try:
            return set(self.collection.get(include=[])["ids"])
        except Exception:
            return set()

    def search(self, e, k: int):
        total = self.count()
        if total == 0:
            return []
        
        n_results = min(max(1, k), total)
        r = self.collection.query(
            query_embeddings=[e],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if not r or not r.get("documents") or not r["documents"][0]:
            return []
            
        return [
            {"document": d, "metadata": m, "distance": x}
            for d, m, x in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
