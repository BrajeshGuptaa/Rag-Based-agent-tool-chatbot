# RAG + Agent Platform

Full-stack starter for a production-style RAG chat and tool-using agent service. It ingests documents, builds a hybrid BM25 + vector index, exposes chat/agent APIs, and logs usage for latency, cost, and A/B routing experiments.

## Features
- Ingestion workers: fetch text/PDF/URL/API sources, normalize + chunk, embed with OpenAI, and upsert into a local SQLite-backed store.
- Retrieval: hybrid BM25 + dense similarity with adjustable weights and top-k; citations returned with scores.
- RAG pipeline: retrieve → blend scores → build context → chat completion with citations + hallucination flagging.
- Agent: planning + OpenAI tool-calling with web search, calculator, SQLite query, and internal API tools.
- Observability: JSONL logs per event (`logs/`), capturing latency, tokens, cost, and tool usage.
- A/B testing: profile router to spread traffic across model/retrieval configs (`control`, `fast`, `quality`).

## Quickstart
1. Install deps (Python 3.10+ recommended):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure env (GitHub Models by default):
   - Copy `.env.example` to `.env`.
   - Set `USE_GITHUB_MODELS=true`.
   - Fill `GITHUB_TOKEN` with a token that has `models:read` access.
   - Optional: adjust weights or directories. (If you prefer OpenAI/Azure, toggle the respective flags and fill keys/deployments.)
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Test health:
   ```bash
   curl http://localhost:8000/health
   ```
5. Stop / restart:
   - Stop (if running in foreground): `Ctrl+C`
   - Stop (if running in background): `kill <pid>` (check with `lsof -i :8000 -sTCP:LISTEN -t`)
   - Restart: same as step 3

## API Overview
- `POST /ingest` — add content. Body:
  ```json
  {
    "source_type": "text",        // or "url", "pdf", "api"
    "content": "Your text here",  // base64-encoded for PDFs if no URL
    "url": null,
    "metadata": {"category": "docs"},
    "document_id": null
  }
  ```
  Returns `{ "document_id": "...", "chunks": 12 }`.

- `POST /chat` — RAG chat with citations.
  ```json
  {"query": "What did the policy say?", "ab_profile": "control", "top_k": 5}
  ```
  Returns `answer`, `citations` (with `chunk_id`), latency, cost, and a `hallucination_flag`.

- `POST /agent` — tool-using agent with RAG context.
  ```json
  {"query": "Compare ARR growth to industry averages", "allowed_tools": ["web_search","calculator"]}
  ```
  Returns `answer`, executed `tool_calls`, citations, latency, and cost.

- `GET /documents` — list ingested documents.
- `GET /health` — service + index status.

### Tools exposed to the agent
- `web_search` (DuckDuckGo), `calculator` (numexpr), `database_query` (SQLite read-only), `internal_api` (HTTP request). Pass `allowed_tools` to restrict.

## GitHub Models
- Set in `.env`:
  - `USE_GITHUB_MODELS=true`
  - `GITHUB_MODELS_BASE_URL=https://models.github.ai/inference`
  - `GITHUB_TOKEN=...` (token with `models:read`)
  - `GITHUB_CHAT_MODEL=openai/gpt-4.1` (default) and `GITHUB_EMBED_MODEL=text-embedding-3-large`

## Azure OpenAI (optional fallback)
- Set `USE_AZURE_OPENAI=true` and provide endpoint/key/deployments if you want to route through Azure instead.

## Design Notes
- Storage: SQLite (`data/index.db`) persists chunks with embeddings (pickled NumPy), BM25 tokens, metadata, and timestamps.
- Retrieval: `rank-bm25` for sparse scores + cosine similarity for dense scores; blended via weights from A/B profile or request override.
- Costing: simple token-based estimation using configurable per-1M token prices.
- Logging: JSON lines per event in `logs/` (`rag_chat.log`, `agent_run.log`). Extendable via `observability.log_event`.

## Next Steps (optional)
- Swap in a production vector DB (e.g., PGVector, Chroma) and a reranker model.
- Add auth/rate limiting and persistent job queue for ingestion.
- Wire real hallucination evaluation (answer-validation prompts) and dashboards.
