import chromadb
from .config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            "memories", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, i, e, d, m):
        self.collection.upsert(
            ids=[str(i)], embeddings=[e], documents=[d], metadatas=[m]
        )

    def delete(self, i):
        self.collection.delete(ids=[str(i)])

    def count(self):
        return self.collection.count()

    def search(self, e, k):
        if not self.count():
            return []
        r = self.collection.query(
            query_embeddings=[e],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"document": d, "metadata": m, "distance": x}
            for d, m, x in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
