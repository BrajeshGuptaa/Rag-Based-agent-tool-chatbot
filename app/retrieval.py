from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .llm import embed_texts
from .storage import Chunk, Storage
from .text_utils import tokenize


class HybridRetriever:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.chunks: List[Chunk] = []
        self.emb_matrix: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        self.chunks = self.storage.load_chunks()
        token_lists = [chunk.tokens for chunk in self.chunks]
        self.bm25 = BM25Okapi(token_lists) if token_lists else None
        if self.chunks:
            self.emb_matrix = np.stack([chunk.embedding for chunk in self.chunks])
            norms = np.linalg.norm(self.emb_matrix, axis=1, keepdims=True) + 1e-10
            self.emb_matrix = self.emb_matrix / norms
        else:
            self.emb_matrix = None

    def search(
        self,
        query: str,
        top_k: int,
        bm25_weight: float,
        embedding_weight: float,
        embed_model: str | None = None,
    ) -> List[Tuple[Chunk, float, float, float]]:
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        bm25_scores = (
            self.bm25.get_scores(query_tokens) if self.bm25 else np.zeros(len(self.chunks))
        )

        query_embedding = embed_texts([query], model=embed_model)[0]
        query_embedding /= np.linalg.norm(query_embedding) + 1e-10
        dense_scores = (
            self.emb_matrix @ query_embedding if self.emb_matrix is not None else np.zeros(len(self.chunks))
        )

        bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-10)
        dense_norm = (dense_scores - dense_scores.min()) / (np.ptp(dense_scores) + 1e-10)

        combined = bm25_weight * bm25_norm + embedding_weight * dense_norm
        top_indices = np.argsort(-combined)[:top_k]

        results: List[Tuple[Chunk, float, float, float]] = []
        for idx in top_indices:
            results.append(
                (
                    self.chunks[idx],
                    float(combined[idx]),
                    float(bm25_scores[idx]),
                    float(dense_scores[idx]),
                )
            )
        return results
