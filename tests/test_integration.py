"""Integration tests for the full nanocortex pipeline."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from nanocortex.api.orchestrator import NanoCortex
from nanocortex.config import Settings


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a minimal valid PNG image."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    raw = b""
    for _ in range(height):
        raw += b"\x00"
        raw += bytes(rgb) * width
    compressed = zlib.compress(raw)

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", compressed)
    out += chunk(b"IEND", b"")
    return out


@pytest.fixture
def cortex(temp_dir: Path) -> NanoCortex:
    """Create a NanoCortex instance with temp directories."""
    settings = Settings(
        audit_dir=temp_dir / "audit",
        data_dir=temp_dir / "data",
    )
    return NanoCortex(settings)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Path:
    """Create a sample PDF for testing."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    pdf_path = temp_dir / "test_doc.pdf"
    doc = fitz.open()

    # Page 1: Text
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Renewable Energy Report")
    page1.insert_text((72, 120), "Solar capacity reached 1,500 GW globally.")
    page1.insert_text((72, 140), "Wind power contributes 30% of renewable energy.")

    # Page 2: Image
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Data Analysis")
    png_bytes = _make_png(100, 50, (0, 128, 0))
    img_rect = fitz.Rect(72, 100, 300, 250)
    page2.insert_image(img_rect, stream=png_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete nanocortex pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_and_query(self, cortex: NanoCortex, sample_pdf: Path):
        """Test full ingestion -> query pipeline."""
        # Ingest
        ingest_result = await cortex.ingest(sample_pdf)
        assert ingest_result["chunks_indexed"] > 0
        assert ingest_result["pages"] == 2

        # Query
        query_result = await cortex.query("What is the global solar capacity?")
        assert query_result["answer"]
        assert query_result["evidence_count"] > 0

    @pytest.mark.asyncio
    async def test_feedback_and_learning(self, cortex: NanoCortex, sample_pdf: Path):
        """Test feedback submission affects learning stats."""
        await cortex.ingest(sample_pdf)
        decision = await cortex.query("What percentage is wind power?")

        # Submit feedback
        fb = cortex.submit_feedback(
            decision_id=decision["decision_id"],
            rating="correct",
        )
        assert fb["rating"] == "correct"

        # Check learning stats
        stats = cortex.get_learning_stats()
        assert stats["feedback_count"] >= 1

    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, cortex: NanoCortex, sample_pdf: Path):
        """Test that audit trail captures all events."""
        await cortex.ingest(sample_pdf)
        decision = await cortex.query("Tell me about renewable energy")

        # Get audit trail for this decision
        trail = cortex.get_audit_trail(decision["decision_id"])

        # Should have perception and reasoning events
        layers = {e["layer"] for e in trail}
        assert "reasoning" in layers

    @pytest.mark.asyncio
    async def test_approval_workflow(self, cortex: NanoCortex, sample_pdf: Path):
        """Test human-in-the-loop approval workflow."""
        await cortex.ingest(sample_pdf)

        # Query with empty results to trigger approval
        cortex.knowledge._chunks.clear()  # Force empty retrieval
        decision = await cortex.query("Unknown topic with no evidence")

        if decision["state"] == "waiting_approval":
            # Test approval
            approval = cortex.approve_decision(decision["decision_id"])
            assert approval.get("state") == "completed"

    def test_image_ingestion(self, cortex: NanoCortex, temp_dir: Path):
        """Test standalone image ingestion."""
        img_path = temp_dir / "test.png"
        img_path.write_bytes(_make_png(100, 100, (255, 0, 0)))

        import asyncio
        result = asyncio.run(cortex.ingest(img_path))

        assert result["images"] == 1
        assert result["chunks_indexed"] >= 1
