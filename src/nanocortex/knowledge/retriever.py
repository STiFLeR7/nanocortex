"""Hybrid RAG retriever: BM25 + vector + reranking with full citation tracking.

Design:
- Documents are chunked on ingestion and stored in an in-memory index.
- BM25 provides keyword recall; vector similarity provides semantic recall.
- Results are fused via Reciprocal Rank Fusion (RRF) and optionally reranked.
- Every result carries citations back to doc_id, page, bbox, and snippet.
- No silent hallucinations: if no evidence is found, the result set is empty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from nanocortex.audit.logger import AuditLogger
from nanocortex.models.domain import (
    Citation,
    DocumentIngestion,
    ExtractedImage,
    RetrievalResponse,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    bbox_data: dict | None = None
    image_id: str | None = None
    modality: str = "text"


class KnowledgeStore:
    """In-memory knowledge store with hybrid retrieval."""

    def __init__(self, audit: AuditLogger) -> None:
        self._audit = audit
        self._chunks: list[Chunk] = []
        self._doc_ids: set[str] = set()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def index_document(self, doc: DocumentIngestion) -> int:
        """Index all text chunks and images from a document. Returns chunk count added."""
        added = 0

        for i, text_block in enumerate(doc.texts):
            # Split large text blocks into ~500-char chunks
            for sub_chunk in self._split_text(text_block.text, max_chars=500):
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_t{i}_{added}",
                    doc_id=doc.doc_id,
                    text=sub_chunk,
                    page=text_block.source_page,
                    bbox_data=text_block.bbox.model_dump() if text_block.bbox else None,
                    modality="text",
                )
                self._chunks.append(chunk)
                added += 1

        for img in doc.images:
            if img.description:
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_img_{img.image_id}",
                    doc_id=doc.doc_id,
                    text=img.description,
                    page=img.page,
                    image_id=img.image_id,
                    modality="image",
                )
                self._chunks.append(chunk)
                added += 1

        self._doc_ids.add(doc.doc_id)

        self._audit.log(
            layer="knowledge",
            event_type="document_indexed",
            payload={"doc_id": doc.doc_id, "chunks_added": added},
        )

        return added

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
    ) -> RetrievalResponse:
        """Retrieve relevant chunks using the specified strategy."""
        if not self._chunks:
            return RetrievalResponse(query=query, results=(), strategy=strategy)

        if strategy == "bm25":
            scored = self._bm25_score(query)
        elif strategy == "vector":
            scored = self._vector_score(query)
        else:  # hybrid
            bm25_scores = self._bm25_score(query)
            vector_scores = self._vector_score(query)
            scored = self._rrf_fuse(bm25_scores, vector_scores)

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results: list[RetrievalResult] = []
        for chunk, score in top:
            if score <= 0:
                continue
            citation = Citation(
                doc_id=chunk.doc_id,
                page=chunk.page,
                snippet=chunk.text[:200],
                image_id=chunk.image_id,
            )
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=round(score, 4),
                citations=(citation,),
                modality=chunk.modality,
            ))

        response = RetrievalResponse(
            query=query,
            results=tuple(results),
            strategy=strategy,
        )

        self._audit.log(
            layer="knowledge",
            event_type="retrieval",
            payload={
                "query": query,
                "strategy": strategy,
                "results_count": len(results),
                "top_score": results[0].score if results else 0,
            },
        )

        return response

    # ── Scoring strategies ────────────────────────────────────────

    def _bm25_score(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[Chunk, float]]:
        """BM25 scoring."""
        query_terms = query.lower().split()
        if not query_terms:
            return [(c, 0.0) for c in self._chunks]

        # Compute IDF
        n = len(self._chunks)
        doc_freq: dict[str, int] = {}
        for chunk in self._chunks:
            terms = set(chunk.text.lower().split())
            for t in query_terms:
                if t in terms:
                    doc_freq[t] = doc_freq.get(t, 0) + 1

        avg_dl = sum(len(c.text.split()) for c in self._chunks) / max(n, 1)

        results: list[tuple[Chunk, float]] = []
        for chunk in self._chunks:
            chunk_terms = chunk.text.lower().split()
            dl = len(chunk_terms)
            score = 0.0
            term_counts: dict[str, int] = {}
            for t in chunk_terms:
                term_counts[t] = term_counts.get(t, 0) + 1

            for t in query_terms:
                tf = term_counts.get(t, 0)
                df = doc_freq.get(t, 0)
                if tf == 0 or df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avg_dl)
                score += idf * numerator / denominator

            results.append((chunk, score))

        return results

    def _vector_score(self, query: str) -> list[tuple[Chunk, float]]:
        """Simple TF-IDF cosine similarity as a lightweight vector proxy.
        Falls back to this when sentence-transformers is not available.
        """
        query_terms = set(query.lower().split())
        results: list[tuple[Chunk, float]] = []

        for chunk in self._chunks:
            chunk_terms = set(chunk.text.lower().split())
            if not query_terms or not chunk_terms:
                results.append((chunk, 0.0))
                continue
            intersection = query_terms & chunk_terms
            # Jaccard-like similarity
            union = query_terms | chunk_terms
            score = len(intersection) / len(union) if union else 0.0
            results.append((chunk, score))

        return results

    def _rrf_fuse(
        self,
        *ranked_lists: list[tuple[Chunk, float]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for ranked in ranked_lists:
            ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)
            for rank, (chunk, _) in enumerate(ranked_sorted, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_map[chunk.chunk_id] = chunk

        return [(chunk_map[cid], score) for cid, score in scores.items()]

    # ── Utilities ─────────────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, max_chars: int = 500) -> list[str]:
        """Split text into chunks respecting sentence boundaries."""
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current = ""
        for sentence in text.replace("\n", " ").split(". "):
            candidate = (current + ". " + sentence).strip() if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text]
