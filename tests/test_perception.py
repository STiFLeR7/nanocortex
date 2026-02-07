"""Unit tests for the Perception layer (ingestion pipeline)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from nanocortex.audit.logger import AuditLogger
from nanocortex.perception.ingestion import IngestionPipeline


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


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    def test_ingest_image_basic(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test basic image ingestion."""
        # Create a test PNG
        img_path = temp_dir / "test.png"
        img_path.write_bytes(_make_png(50, 50, (255, 0, 0)))

        pipeline = IngestionPipeline(audit_logger)
        result = pipeline.ingest_image(img_path)

        assert result.filename == "test.png"
        assert result.pages == 1
        assert len(result.images) == 1
        assert result.images[0].image_bytes_b64  # Non-empty base64

    def test_ingest_image_file_not_found(self, audit_logger: AuditLogger):
        """Test error handling for missing file."""
        pipeline = IngestionPipeline(audit_logger)

        with pytest.raises(FileNotFoundError):
            pipeline.ingest_image(Path("/nonexistent/image.png"))

    @pytest.mark.integration
    def test_ingest_pdf_with_text(self, audit_logger: AuditLogger, temp_dir: Path):
        """Test PDF ingestion (requires PyMuPDF)."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        # Create a simple PDF
        pdf_path = temp_dir / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content for nanocortex")
        doc.save(str(pdf_path))
        doc.close()

        pipeline = IngestionPipeline(audit_logger)
        result = pipeline.ingest_pdf(pdf_path)

        assert result.filename == "test.pdf"
        assert result.pages == 1
        assert len(result.texts) > 0
        assert "Test content" in result.texts[0].text
