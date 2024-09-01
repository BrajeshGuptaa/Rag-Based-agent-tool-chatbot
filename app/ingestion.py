import base64
import io
import uuid
from typing import List, Tuple

import httpx
from pypdf import PdfReader

from .config import get_settings
from .llm import embed_texts
from .schemas import IngestRequest
from .storage import Chunk, Storage
from .text_utils import normalize_text, tokenize


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    normalized = normalize_text(text)
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def extract_pdf_text(binary: bytes) -> str:
    reader = PdfReader(io.BytesIO(binary))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest(request: IngestRequest, storage: Storage) -> Tuple[str, int]:
    settings = get_settings()
    document_id = request.document_id or str(uuid.uuid4())

    if request.source_type == "text":
        raw_text = request.content or ""
        source = "text"
    elif request.source_type == "url":
        resp = httpx.get(request.url, timeout=10.0)
        resp.raise_for_status()
        raw_text = resp.text
        source = request.url or "url"
    elif request.source_type == "pdf":
        if request.url:
            resp = httpx.get(request.url, timeout=15.0)
            resp.raise_for_status()
            raw_text = extract_pdf_text(resp.content)
            source = request.url
        else:
            assert request.content, "PDF content missing"
            binary = base64.b64decode(request.content)
            raw_text = extract_pdf_text(binary)
            source = "pdf-upload"
    elif request.source_type == "api":
        assert request.url, "API url missing"
        resp = httpx.get(request.url, timeout=10.0)
        resp.raise_for_status()
        raw_text = resp.text
        source = request.url
    else:
        raise ValueError(f"Unsupported source_type {request.source_type}")

    storage.upsert_document(document_id, source, request.metadata or {})

    chunks_raw = chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embed_texts(chunks_raw)
    chunk_objs: List[Chunk] = []
    for idx, (chunk_text_val, embedding) in enumerate(zip(chunks_raw, embeddings)):
        chunk_objs.append(
            Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                text=chunk_text_val,
                source=source,
                metadata=request.metadata or {},
                tokens=tokenize(chunk_text_val),
                embedding=embedding,
            )
        )
    storage.add_chunks(document_id, chunk_objs)
    return document_id, len(chunk_objs)
