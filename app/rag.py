import time
from typing import List

from .config import ABProfile, get_settings
from .llm import chat_completion, estimate_cost_usd
from .observability import log_event
from .retrieval import HybridRetriever
from .schemas import ChatRequest, ChatResponse, Citation
from .storage import Storage


class RAGPipeline:
    def __init__(self, storage: Storage, retriever: HybridRetriever) -> None:
        self.storage = storage
        self.retriever = retriever

    def _build_context(self, chunks_with_scores) -> str:
        parts: List[str] = []
        for chunk, score, bm25_score, dense_score in chunks_with_scores:
            parts.append(
                f"[{chunk.id}] {chunk.text}\nsource: {chunk.source} | scores -> bm25:{bm25_score:.2f} dense:{dense_score:.2f} blended:{score:.2f}"
            )
        return "\n\n".join(parts)

    def _as_citations(self, chunks_with_scores) -> List[Citation]:
        citations: List[Citation] = []
        for chunk, score, _, _ in chunks_with_scores:
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    score=score,
                    text=chunk.text[:4000],
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )
        return citations

    def run(self, request: ChatRequest, profile_name: str) -> ChatResponse:
        settings = get_settings()
        profile: ABProfile = settings.ab_profiles.get(profile_name, settings.ab_profiles["control"])
        top_k = request.top_k or profile.top_k or settings.top_k

        if request.rerank_weight is not None:
            embedding_weight = request.rerank_weight
            bm25_weight = max(0.0, 1.0 - embedding_weight)
        else:
            bm25_weight = profile.bm25_weight
            embedding_weight = profile.embedding_weight

        start = time.perf_counter()
        retrieved = self.retriever.search(
            request.query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            embed_model=profile.embed_model,
        )
        context = self._build_context(retrieved)
        citations = self._as_citations(retrieved)

        messages = [{"role": h["role"], "content": h["content"]} for h in request.history]
        system_prompt = (
            "You are a concise assistant. Use the provided context to answer the user's question. "
            "Cite sources using [chunk_id] brackets. If the context is insufficient, say you are unsure."
        )
        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}"})

        completion = chat_completion(
            messages=messages,
            model=profile.chat_model,
        )

        content = completion.choices[0].message.content or ""
        usage = completion.usage
        prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
        completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
        cost_usd = estimate_cost_usd(prompt_tokens, completion_tokens, model=profile.chat_model)

        latency_ms = (time.perf_counter() - start) * 1000
        hallucination_flag = bool(not citations or (retrieved and retrieved[0][1] < 0.2))

        log_event(
            "rag_chat",
            {
                "query": request.query,
                "profile": profile_name,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "top_k": top_k,
                "bm25_weight": bm25_weight,
                "embedding_weight": embedding_weight,
                "citations": [c.model_dump() for c in citations],
            },
        )

        return ChatResponse(
            answer=content,
            citations=citations,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            hallucination_flag=hallucination_flag,
            profile=profile_name,
        )
