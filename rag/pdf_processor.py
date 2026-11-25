"""PDF extraction and overlapping text chunking via PyMuPDF."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - handled cleanly at runtime
    fitz = None


@dataclass
class PDFProcessResult:
    """Outcome of extracting and chunking a single PDF document."""

    success: bool
    doc_id: str
    filename: str
    page_count: int = 0
    chunks: List[dict] = field(default_factory=list)
    error: str = ""


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> List[str]:
    """Split text into overlapping character chunks respecting sentence/word boundaries."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    safe_overlap = min(max(chunk_overlap, 0), chunk_size // 2)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer breaking near paragraph, sentence, or space boundary
            window_start = start + int(chunk_size * 0.6)
            split_pos = -1
            for sep in ("\n\n", ". ", "? ", "! ", "\n", " "):
                candidate = cleaned.rfind(sep, window_start, end)
                if candidate != -1:
                    split_pos = candidate + len(sep)
                    break
            if split_pos > start:
                end = split_pos

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        next_start = end - safe_overlap
        start = next_start if next_start > start else end

    return chunks


def extract_and_chunk_pdf(
    filename: str,
    pdf_bytes: bytes,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> PDFProcessResult:
    """Extract text page-by-page using PyMuPDF and return structured chunk dicts."""
    doc_id = uuid.uuid4().hex[:10]
    if fitz is None:
        return PDFProcessResult(
            success=False,
            doc_id=doc_id,
            filename=filename,
            error="PyMuPDF (fitz) is not installed. Run `pip install PyMuPDF`.",
        )

    if not pdf_bytes:
        return PDFProcessResult(
            success=False,
            doc_id=doc_id,
            filename=filename,
            error="Uploaded file is empty.",
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return PDFProcessResult(
            success=False,
            doc_id=doc_id,
            filename=filename,
            error=f"Could not open '{filename}' as a valid PDF ({exc}).",
        )

    try:
        if doc.is_encrypted:
            return PDFProcessResult(
                success=False,
                doc_id=doc_id,
                filename=filename,
                page_count=len(doc),
                error=f"'{filename}' is password-protected/encrypted and cannot be read.",
            )

        page_count = len(doc)
        all_chunks: List[dict] = []
        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text") or ""
            page_chunks = split_text_into_chunks(
                raw_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for local_idx, chunk_text in enumerate(page_chunks):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_id}_p{page_idx + 1}_{local_idx}",
                        "doc_id": doc_id,
                        "source": filename,
                        "page": page_idx + 1,
                        "text": chunk_text,
                    }
                )

        if not all_chunks:
            return PDFProcessResult(
                success=False,
                doc_id=doc_id,
                filename=filename,
                page_count=page_count,
                error=(
                    f"No extractable text found in '{filename}'. "
                    "It may be a scanned image PDF without an OCR text layer."
                ),
            )

        return PDFProcessResult(
            success=True,
            doc_id=doc_id,
            filename=filename,
            page_count=page_count,
            chunks=all_chunks,
        )
    except Exception as exc:
        return PDFProcessResult(
            success=False,
            doc_id=doc_id,
            filename=filename,
            error=f"PDF text extraction failed for '{filename}': {exc}",
        )
    finally:
        doc.close()
