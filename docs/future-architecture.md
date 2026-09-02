# Future Architecture

## Proposed Production Architecture

```text
                         Client
                            │
                            ▼
                      REST API
                      FastAPI
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
         Authentication   Memory        LLM Service
                         Manager
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
           PostgreSQL                ChromaDB
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                       Redis Cache
```

## Proposed Improvements

### API Layer
Introduce FastAPI to expose the memory system through REST endpoints.

### Multi-user Support
Associate memories with user IDs and enforce memory isolation.

### Authentication
Add secure authentication before exposing the application as a service.

### Connection Pooling
Use PostgreSQL connection pools to reduce connection creation overhead.

### Caching
Use Redis for frequently accessed metadata and repeated retrieval operations.

### Hybrid Retrieval
Combine:

- Keyword search
- Vector similarity
- Metadata filtering

### Reranking
Use a reranker after vector retrieval to improve the relevance of memories supplied to the LLM.

### Observability
Add:

- Structured logging
- Metrics
- Latency tracking
- Error monitoring

### Deployment
Support:

- Docker
- Docker Compose
- CI/CD
- Automated testing
