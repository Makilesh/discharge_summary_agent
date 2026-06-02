"""
pdf_processor.py — PDF to Page Image Extraction
=================================================

Handles the conversion of scanned PDF pages to base64-encoded images
suitable for Gemini Vision API calls.

Clinical Safety:
    - Every page must be processed — skipping pages risks losing critical data.
    - Unreadable pages (image extraction failure) are logged, never silently dropped.
    - Page numbering is 1-indexed to match clinical document references.
"""

from __future__ import annotations
import base64
import io
from typing import Optional

import fitz  # PyMuPDF


# ─── FULL PDF EXTRACTION ────────────────────────────────────────────────────────

def extract_all_page_images(pdf_path: str, dpi: int = 200) -> dict[int, bytes]:
    """
    Extract all pages from a scanned PDF as PNG image bytes.

    Purpose:
        Converts each page of a scanned PDF into a PNG image for OCR
        via Gemini Vision. Uses 200 DPI for balance of quality vs size.

    Clinical Safety:
        Every page must be attempted. A failure on one page must not
        abort processing of remaining pages — each failure is logged
        individually.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 200 — good for handwriting).

    Returns:
        Dict mapping 1-indexed page numbers to PNG image bytes.
        Pages that fail to render are omitted and must be tracked separately.
    """
    doc = fitz.open(pdf_path)
    page_images: dict[int, bytes] = {}
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):
        page_num = page_idx + 1  # 1-indexed for clinical references
        try:
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            page_images[page_num] = img_bytes
        except Exception as e:
            # Log but don't crash — this page will be marked unreadable
            print(f"[PDF_PROCESSOR] WARNING: Failed to render page {page_num}: {e}")
            continue

    doc.close()
    return page_images


def get_page_image_base64(page_images: dict[int, bytes], page_num: int) -> Optional[str]:
    """
    Get a single page image as a base64-encoded string.

    Purpose:
        Prepares a page image for inclusion in a Gemini Vision API call.

    Clinical Safety:
        Returns None if the page is not available — caller must handle
        this as an unreadable page, not silently skip it.

    Args:
        page_images: Dict of page_num -> PNG bytes from extract_all_page_images.
        page_num: 1-indexed page number.

    Returns:
        Base64-encoded PNG string, or None if page not available.
    """
    img_bytes = page_images.get(page_num)
    if img_bytes is None:
        return None
    return base64.b64encode(img_bytes).decode("utf-8")


def get_batch_page_images(
    page_images: dict[int, bytes],
    page_nums: list[int],
) -> list[tuple[int, str]]:
    """
    Get multiple page images as base64 strings for batch processing.

    Purpose:
        Enables batch tool calls (e.g., all drug chart pages in one LLM call)
        to maximize efficiency within the 20-step agent budget.

    Args:
        page_images: Dict of page_num -> PNG bytes.
        page_nums: List of 1-indexed page numbers to retrieve.

    Returns:
        List of (page_num, base64_string) tuples for available pages.
        Missing pages are omitted from the list.
    """
    results: list[tuple[int, str]] = []
    for pn in page_nums:
        b64 = get_page_image_base64(page_images, pn)
        if b64 is not None:
            results.append((pn, b64))
    return results


def get_total_pages(pdf_path: str) -> int:
    """Get the total number of pages in a PDF file."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
