"""PDF & image ingestion pipeline.

Uses PyMuPDF (fitz) for PDF parsing and Pillow for image handling.
OCR is available via pytesseract when text extraction yields nothing.
Every extraction is grounded with bounding-box coordinates and page numbers.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from nanocortex.audit.logger import AuditLogger
from nanocortex.models.domain import (
    BoundingBox,
    DocumentIngestion,
    ExtractedImage,
    ExtractedText,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Extract structured text and images from documents."""

    def __init__(self, audit: AuditLogger) -> None:
        self._audit = audit

    def ingest_pdf(self, file_path: Path | str) -> DocumentIngestion:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        texts: list[ExtractedText] = []
        images: list[ExtractedImage] = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks with bounding boxes
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:  # text block
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                        block_text += "\n"
                    block_text = block_text.strip()
                    if block_text:
                        bbox = BoundingBox(
                            x=block["bbox"][0],
                            y=block["bbox"][1],
                            width=block["bbox"][2] - block["bbox"][0],
                            height=block["bbox"][3] - block["bbox"][1],
                            page=page_num,
                        )
                        texts.append(ExtractedText(
                            text=block_text,
                            bbox=bbox,
                            source_page=page_num,
                        ))

            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    img_bytes = base_image["image"]
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")
                    images.append(ExtractedImage(
                        page=page_num,
                        image_bytes_b64=img_b64,
                        description=f"Image {img_index} from page {page_num}",
                    ))

        doc.close()

        # If no text was extracted, fall back to OCR
        if not texts:
            texts = self._ocr_pdf(file_path)

        result = DocumentIngestion(
            filename=file_path.name,
            pages=len(doc) if hasattr(doc, '__len__') else 0,
            texts=tuple(texts),
            images=tuple(images),
        )

        self._audit.log(
            layer="perception",
            event_type="document_ingested",
            payload={
                "filename": result.filename,
                "pages": result.pages,
                "text_blocks": len(texts),
                "images_found": len(images),
            },
        )

        return result

    def ingest_image(self, file_path: Path | str) -> DocumentIngestion:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        with open(file_path, "rb") as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        # Attempt OCR on the image
        texts = self._ocr_image(img_bytes)

        image = ExtractedImage(
            page=0,
            image_bytes_b64=img_b64,
            description=f"Standalone image: {file_path.name}",
        )

        result = DocumentIngestion(
            filename=file_path.name,
            mime_type=f"image/{file_path.suffix.lstrip('.')}",
            pages=1,
            texts=tuple(texts),
            images=(image,),
        )

        self._audit.log(
            layer="perception",
            event_type="image_ingested",
            payload={"filename": result.filename, "text_blocks": len(texts)},
        )

        return result

    def _ocr_pdf(self, file_path: Path) -> list[ExtractedText]:
        """Fallback OCR for scanned PDFs."""
        try:
            import fitz
            from PIL import Image
            import pytesseract

            doc = fitz.open(str(file_path))
            texts: list[ExtractedText] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    texts.append(ExtractedText(
                        text=ocr_text,
                        confidence=0.8,
                        source_page=page_num,
                    ))
            doc.close()
            return texts
        except Exception:
            logger.warning("OCR fallback failed for %s", file_path)
            return []

    def _ocr_image(self, img_bytes: bytes) -> list[ExtractedText]:
        """OCR a raw image."""
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                return [ExtractedText(text=ocr_text, confidence=0.8, source_page=0)]
            return []
        except Exception:
            logger.warning("OCR failed on image bytes")
            return []
