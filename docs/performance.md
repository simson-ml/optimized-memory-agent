# Performance Optimizations

## Overview

The system is designed to minimize unnecessary operations in the critical user-response path.

## 1. Background Memory Indexing

Embedding generation and vector insertion can be computationally expensive.

Instead of blocking the user interaction:

```text
User Message
      │
      ▼
Store PostgreSQL Memory
      │
      ├──────────────► Continue Conversation
      │
      ▼
Background Worker
      │
      ▼
Generate Embedding
      │
      ▼
Store in ChromaDB
```

This improves perceived responsiveness.

## 2. Adaptive Retrieval

The system avoids vector search when historical context is unlikely to be required.

This reduces unnecessary:

- Embedding generation
- Disk operations
- Vector similarity calculations

## 3. Top-K Limiting

The number of retrieved memories is configurable.

```env
TOP_K=5
```

Retrieving only the most relevant memories reduces prompt size and improves inference efficiency.

## 4. Similarity Threshold

Low-quality matches can be excluded using a similarity threshold.

```env
SIMILARITY_THRESHOLD=0.75
```

This reduces irrelevant context reaching the LLM.

## 5. Rebuildable Vector Index

PostgreSQL stores the authoritative records while ChromaDB remains a derived index.

Benefits include:

- Easier recovery
- Reindexing after embedding model changes
- Reduced risk of permanent data loss

## Recommended Future Optimizations

- PostgreSQL connection pooling
- Batch embedding generation
- Duplicate memory detection
- Hybrid keyword + semantic retrieval
- Result reranking
- Redis caching
- Async I/O
- Streaming LLM responses
- Performance benchmarking
