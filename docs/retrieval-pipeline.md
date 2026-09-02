# Retrieval Pipeline

## Overview

The retrieval pipeline determines when long-term memory should be searched and how relevant memories are injected into the LLM context.

## Pipeline

```text
User Question
      │
      ▼
Should Retrieve?
      │
 ┌────┴─────┐
 │          │
NO         YES
 │          │
 ▼          ▼
LLM    Generate Embedding
             │
             ▼
       ChromaDB Search
             │
             ▼
      Similarity Filtering
             │
             ▼
      Relevant Memories
             │
             ▼
          LLM Prompt
             │
             ▼
         Response
```

## Adaptive Retrieval

Traditional RAG pipelines perform vector search for every query.

```text
Query → Embedding → Vector Search → LLM
```

This can create unnecessary latency.

The Optimized Local LLM Memory Agent first determines whether retrieval is likely to be useful.

For example:

### Direct Knowledge Question

```text
What is Python?
```

Memory retrieval may not be necessary.

### Historical Context Question

```text
What programming language do I prefer?
```

Memory retrieval is likely useful.

## Similarity Filtering

Retrieved memories are filtered using a configurable similarity threshold.

Example configuration:

```env
TOP_K=5
SIMILARITY_THRESHOLD=0.75
```

Only sufficiently relevant memories should be passed to the LLM.

## Benefits

Adaptive retrieval reduces:

- Unnecessary embedding generation
- Vector database operations
- Response latency
- Context window usage

while preserving access to relevant long-term memories.
