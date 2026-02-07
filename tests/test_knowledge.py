"""Unit tests for the Knowledge layer (retriever)."""

from __future__ import annotations

import pytest

from nanocortex.audit.logger import AuditLogger
from nanocortex.knowledge.retriever import KnowledgeStore
from nanocortex.models.domain import DocumentIngestion, ExtractedText


class TestKnowledgeStore:
    """Tests for KnowledgeStore."""

    def test_index_document(self, audit_logger: AuditLogger):
        """Test document indexing."""
        store = KnowledgeStore(audit_logger)

        doc = DocumentIngestion(
            filename="test.pdf",
            pages=1,
            texts=(
                ExtractedText(text="Solar energy is renewable.", source_page=0),
                ExtractedText(text="Wind power is sustainable.", source_page=0),
            ),
        )

        chunks_added = store.index_document(doc)
        assert chunks_added >= 2
        assert store.chunk_count >= 2

    def test_retrieve_bm25(self, audit_logger: AuditLogger):
        """Test BM25 retrieval."""
        store = KnowledgeStore(audit_logger)

        doc = DocumentIngestion(
            filename="test.pdf",
            pages=1,
            texts=(
                ExtractedText(text="Solar panels convert sunlight to electricity.", source_page=0),
                ExtractedText(text="Wind turbines generate power from wind.", source_page=0),
                ExtractedText(text="Batteries store excess energy.", source_page=0),
            ),
        )
        store.index_document(doc)

        result = store.retrieve("solar electricity", top_k=2, strategy="bm25")

        assert result.query == "solar electricity"
        assert result.strategy == "bm25"
        assert len(result.results) <= 2
        # Solar should be top result
        assert "solar" in result.results[0].text.lower() or "sunlight" in result.results[0].text.lower()

    def test_retrieve_hybrid(self, audit_logger: AuditLogger):
        """Test hybrid retrieval."""
        store = KnowledgeStore(audit_logger)

        doc = DocumentIngestion(
            filename="test.pdf",
            pages=1,
            texts=(
                ExtractedText(text="Renewable energy reduces carbon emissions.", source_page=0),
                ExtractedText(text="Fossil fuels contribute to climate change.", source_page=0),
            ),
        )
        store.index_document(doc)

        result = store.retrieve("carbon emissions", top_k=2, strategy="hybrid")

        assert result.strategy == "hybrid"
        assert len(result.results) > 0

    def test_empty_store_retrieve(self, audit_logger: AuditLogger):
        """Test retrieval from empty store."""
        store = KnowledgeStore(audit_logger)
        result = store.retrieve("any query", top_k=5)

        assert len(result.results) == 0

    def test_citations_included(self, audit_logger: AuditLogger):
        """Test that citations are included in results."""
        store = KnowledgeStore(audit_logger)

        doc = DocumentIngestion(
            doc_id="doc123",
            filename="test.pdf",
            pages=1,
            texts=(
                ExtractedText(text="Important finding about energy.", source_page=2),
            ),
        )
        store.index_document(doc)

        result = store.retrieve("energy finding", top_k=1)

        if result.results:
            r = result.results[0]
            assert len(r.citations) > 0
            assert r.citations[0].doc_id == "doc123"
            assert r.citations[0].page == 2
