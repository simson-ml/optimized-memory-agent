# Development Guide

## Recommended Environment

- Python 3.11 or Python 3.12
- PostgreSQL
- Ollama
- Git

## Recommended Models

### LLM

```bash
ollama pull llama3
```

### Embedding Model

```bash
ollama pull nomic-embed-text
```

## Running Locally

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using `.env.example` as a template.

Run the application:

```bash
python -m app.main
```

## Code Quality Recommendations

Before committing changes:

```bash
git status
```

Review tracked files:

```bash
git ls-files
```

Do not commit:

- `.env`
- Database passwords
- ChromaDB runtime data
- Python virtual environments
- Compiled Python files
- Personal conversation data

## Testing

Recommended test structure:

```text
tests/
├── test_config.py
├── test_database.py
├── test_memory.py
├── test_vector_store.py
└── test_llm.py
```

Important areas to test:

- Memory classification
- Database operations
- Vector retrieval
- Similarity filtering
- Reindexing
- Failure recovery
