import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ab_testing import ABRouter
from .agent import AgentOrchestrator
from .config import get_settings
from .ingestion import ingest
from .rag import RAGPipeline
from .retrieval import HybridRetriever
from .schemas import AgentRequest, AgentResponse, ChatRequest, ChatResponse, IngestRequest
from .storage import Storage

app = FastAPI(title="RAG + Agent Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
storage = Storage()
retriever = HybridRetriever(storage)
rag_pipeline = RAGPipeline(storage, retriever)
agent_orchestrator = AgentOrchestrator(storage, retriever)
ab_router = ABRouter()
static_dir = Path(__file__).resolve().parent.parent / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(storage.list_documents()),
        "data_dir": settings.data_dir,
    }


@app.post("/ingest")
def ingest_endpoint(payload: IngestRequest):
    try:
        doc_id, chunk_count = ingest(payload, storage)
        retriever.refresh()
        return {"document_id": doc_id, "chunks": chunk_count}
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    profile = ab_router.choose(payload.ab_profile)
    try:
        return rag_pipeline.run(payload, profile_name=profile)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent", response_model=AgentResponse)
def agent(payload: AgentRequest):
    profile = ab_router.choose(payload.ab_profile)
    try:
        return agent_orchestrator.run(payload, profile_name=profile)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents")
def documents():
    return storage.list_documents()


@app.get("/")
def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Ensure static/index.html exists."}
