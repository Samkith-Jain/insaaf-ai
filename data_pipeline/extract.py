"""
Stage 1: Extraction.
Pulls raw text out of PDF (digital or scanned) and HTML judgment sources.
"""
import html
import re
import subprocess
import tempfile
import os
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path


def extract_pdf_text(path: str, ocr_threshold_chars: int = 200) -> tuple[str, str]:
    """
    Try native text extraction (pdfplumber) first. If a page yields almost
    no text (i.e. it's a scanned/image-only page), fall back to OCR via
    pytesseract for that page.
    Returns (full_text, method) where method is 'pdfplumber', 'ocr', or 'mixed'.
    """
    text_parts = []
    methods_used = set()

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) >= ocr_threshold_chars:
                text_parts.append(page_text)
                methods_used.add("pdfplumber")
            else:
                ocr_text = _ocr_page(path, i)
                text_parts.append(ocr_text)
                methods_used.add("ocr")

    method = "mixed" if len(methods_used) > 1 else (methods_used.pop() if methods_used else "none")
    return "\n\n".join(text_parts), method


def _ocr_page(pdf_path: str, page_index: int) -> str:
    """Rasterize a single PDF page and run tesseract OCR on it."""
    images = convert_from_path(pdf_path, first_page=page_index + 1, last_page=page_index + 1, dpi=300)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def extract_html_text(path: str) -> str:
    """Strip HTML tags from an NCDRC judgment page saved as .html and return clean text."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    # drop script/style blocks
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # convert common block-ish tags into line breaks before stripping
    raw = re.sub(r"</(tr|table|p|div|br)>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract(path: str) -> dict:
    """Dispatch by extension. Returns {text, method, source_path}."""
    ext = Path(path).suffix.lower()
    if ext in (".pdf",):
        text, method = extract_pdf_text(path)
    elif ext in (".html", ".htm"):
        text, method = extract_html_text(path), "html"
    elif ext in (".txt",):
        text, method = Path(path).read_text(encoding="utf-8", errors="ignore"), "plaintext"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return {"text": text, "method": method, "source_path": path}


if __name__ == "__main__":
    import sys
    result = extract(sys.argv[1])
    print(f"[{result['method']}] {len(result['text'])} chars extracted from {result['source_path']}")
    print(result["text"][:500])
