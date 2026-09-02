# System Architecture

## Overview

The Optimized Local LLM Memory Agent uses a hybrid memory architecture designed for persistent storage, semantic retrieval, and efficient local inference.

## High-Level Architecture

```text
                           ┌──────────────────┐
                           │       User       │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │   Chat Session   │
                           │     main.py      │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
           ┌─────────────────┐             ┌─────────────────┐
           │ Memory Manager  │             │   Ollama LLM    │
           │    memory.py    │             │     llm.py      │
           └────────┬────────┘             └─────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│    PostgreSQL     │   │     ChromaDB     │
│                  │   │                  │
│ Source of Truth  │──▶│ Semantic Search  │
│ Persistent Data  │   │ Vector Index     │
└──────────────────┘   └──────────────────┘
```

## Core Components

### Chat Session
The application entry point responsible for accepting user input, managing commands, and coordinating the memory and LLM components.

### Memory Manager
Determines whether information should be stored as memory and whether previous memories should be retrieved for a new query.

### PostgreSQL
Acts as the authoritative persistence layer. Original memories and metadata are stored here.

### ChromaDB
Acts as a derived vector index used for semantic similarity search.

### Ollama
Provides local LLM inference and embedding generation.

## Design Principle

The key architectural principle is:

> PostgreSQL is the source of truth; ChromaDB is a rebuildable search index.

This separation improves reliability because the vector database can be rebuilt from persistent records.
