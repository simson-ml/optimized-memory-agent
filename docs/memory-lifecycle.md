# Memory Lifecycle

## Overview

A memory passes through several stages from user input to semantic retrieval.

## Lifecycle

```text
User Message
     │
     ▼
Memory Detection
     │
     ▼
Important Memory?
 ┌────┴────┐
 │         │
YES        NO
 │         │
 ▼         ▼
Store      Normal
Memory     Conversation
 │
 ▼
PostgreSQL
 │
 ▼
indexed = false
 │
 ▼
Background Worker
 │
 ▼
Generate Embedding
 │
 ▼
ChromaDB
 │
 ▼
indexed = true
```

## Stage 1: Memory Detection

The system analyzes incoming messages to determine whether they contain information worth retaining.

Examples include:

- User preferences
- Personal facts
- Long-term goals
- Technical preferences
- Important project information
- Explicit requests to remember information

## Stage 2: Persistent Storage

Important memories are first stored in PostgreSQL.

This ensures that memory data is preserved even if vector indexing fails.

## Stage 3: Background Indexing

A background worker generates an embedding without blocking the main conversation.

This improves perceived response latency.

## Stage 4: Vector Indexing

The generated embedding is stored in ChromaDB together with the associated memory metadata.

## Stage 5: Retrieval

When a future query requires historical context, the system performs semantic search and supplies relevant memories to the LLM.

## Failure Recovery

If indexing fails:

1. The memory remains stored in PostgreSQL.
2. Its indexing status remains incomplete.
3. The memory can later be reindexed.

This design prevents vector database failures from causing permanent memory loss.
