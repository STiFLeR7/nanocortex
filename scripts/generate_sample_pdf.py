"""Generate a sample PDF with text and images for the vertical slice demo.

This creates a simple PDF about renewable energy with an embedded chart image,
suitable for testing the full ingestion -> retrieval -> reasoning pipeline.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a minimal valid PNG image filled with a single color."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    raw = b""
    for _ in range(height):
        raw += b"\x00"  # filter byte
        raw += bytes(rgb) * width
    compressed = zlib.compress(raw)

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", compressed)
    out += chunk(b"IEND", b"")
    return out


def generate_sample_pdf(output_path: Path) -> Path:
    """Generate a sample PDF with text content and an embedded image."""
    import fitz  # PyMuPDF

    doc = fitz.open()

    # Page 1: Title and intro
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text(
        (72, 80),
        "Renewable Energy Systems: A Technical Overview",
        fontsize=18,
        fontname="helv",
    )
    page1.insert_text(
        (72, 120),
        "Document ID: SAMPLE-001\nDate: 2026-01-15\nClassification: Internal",
        fontsize=10,
        fontname="helv",
    )
    body_text = (
        "Solar photovoltaic (PV) systems convert sunlight directly into electricity "
        "using semiconductor materials. The global installed capacity reached 1,500 GW "
        "in 2025, with an average module efficiency of 22.5%. Key factors affecting "
        "output include solar irradiance, panel orientation, temperature coefficients, "
        "and shading losses.\n\n"
        "Wind energy systems harness kinetic energy from atmospheric motion. Modern "
        "onshore turbines have a rated capacity of 3-6 MW, while offshore installations "
        "can exceed 15 MW per unit. The capacity factor for well-sited wind farms "
        "ranges from 30-50%.\n\n"
        "Battery energy storage systems (BESS) are critical for grid integration of "
        "variable renewable sources. Lithium-ion technology dominates with round-trip "
        "efficiency of 85-95%. The levelized cost of storage has decreased by 90% "
        "since 2010."
    )
    text_rect = fitz.Rect(72, 160, 540, 700)
    page1.insert_textbox(text_rect, body_text, fontsize=11, fontname="helv")

    # Page 2: Data and image
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text(
        (72, 80),
        "Performance Metrics and Analysis",
        fontsize=16,
        fontname="helv",
    )
    metrics_text = (
        "Table 1: Renewable Energy Cost Comparison (2025)\n\n"
        "Technology       | LCOE ($/MWh) | Capacity Factor\n"
        "Solar PV         | 25-35        | 15-25%\n"
        "Onshore Wind     | 30-45        | 30-45%\n"
        "Offshore Wind    | 55-80        | 40-55%\n"
        "Battery Storage  | 120-180      | N/A\n\n"
        "Figure 1 below shows the capacity distribution across regions. "
        "The Asia-Pacific region leads with 45% of global installed capacity, "
        "followed by Europe at 28% and North America at 18%."
    )
    text_rect2 = fitz.Rect(72, 100, 540, 350)
    page2.insert_textbox(text_rect2, metrics_text, fontsize=11, fontname="helv")

    # Embed a simple chart image (green rectangle representing a bar chart)
    png_bytes = _make_png(200, 100, (34, 139, 34))  # forest green
    img_rect = fitz.Rect(72, 370, 400, 550)
    page2.insert_image(img_rect, stream=png_bytes)

    page2.insert_text(
        (72, 570),
        "Figure 1: Regional Capacity Distribution (illustrative)",
        fontsize=9,
        fontname="helv",
    )

    conclusion = (
        "Conclusion: The transition to renewable energy is accelerating globally. "
        "Cost reductions, policy support, and technological improvements continue "
        "to drive adoption. However, grid integration challenges require significant "
        "investment in storage and transmission infrastructure."
    )
    text_rect3 = fitz.Rect(72, 600, 540, 750)
    page2.insert_textbox(text_rect3, conclusion, fontsize=11, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    out = generate_sample_pdf(Path("data/sample/renewable_energy_report.pdf"))
    print(f"Generated: {out}")
