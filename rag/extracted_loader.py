"""Read-only loader for pre-extracted GATE study materials in the local data folder."""

from __future__ import annotations

import collections
import gzip
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from rag.pdf_processor import extract_and_chunk_pdf, split_text_into_chunks

SUPPORTED_EXTENSIONS = {".json", ".gz", ".txt", ".md", ".pdf"}


@dataclass
class LoadedCorpus:
    """Result of scanning and chunking local extracted study materials."""

    documents: List[dict] = field(default_factory=list)
    chunks: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def infer_document_subject(filename: str) -> str:
    """Infer a human-readable GATE subject/category label from the source filename."""
    low = filename.lower()
    if "syllabus" in low and "da" in low:
        return "GATE DA Syllabus"
    if low.startswith("da") and "key" in low:
        return "GATE DA Answer Key"
    if low.startswith("da"):
        return "GATE DA Paper"
    if low.startswith("cs") and "key" in low:
        return "GATE CS Answer Key"
    if low.startswith("cs"):
        return "GATE CS Paper"
    if low.startswith("ae"):
        return "GATE AE Paper"
    if "python" in low:
        return "Programming (Python)"
    if "discrete" in low:
        return "Discrete Structures"
    return "Study Material"


def _load_extracted_json_records(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[dict], List[dict], List[str]]:
    """Parse pre-extracted page records (.json or .json.gz) without re-extracting PDFs."""
    docs: List[dict] = []
    chunks: List[dict] = []
    warnings: List[str] = []

    try:
        if file_path.suffix == ".gz":
            raw_text = gzip.open(file_path, "rt", encoding="utf-8").read()
        else:
            raw_text = file_path.read_text(encoding="utf-8")
        records = json.loads(raw_text)
    except Exception as exc:
        warnings.append(f"Failed to read '{file_path.name}': {exc}")
        return docs, chunks, warnings

    if not isinstance(records, list):
        warnings.append(f"'{file_path.name}' is not a list of extracted page records; skipped.")
        return docs, chunks, warnings

    by_source: Dict[str, List[dict]] = collections.defaultdict(list)
    for rec in records:
        if isinstance(rec, dict):
            src = str(rec.get("file_name") or rec.get("relative_path") or file_path.name)
            by_source[src].append(rec)

    now_label = datetime.now().strftime("%Y-%m-%d %H:%M")

    for source_name, page_records in sorted(by_source.items()):
        doc_id = f"local_{uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path.name}:{source_name}').hex[:10]}"
        page_records_sorted = sorted(
            page_records,
            key=lambda r: int(r.get("page_number") or 0),
        )
        doc_chunks: List[dict] = []
        skipped_pages = 0

        for rec in page_records_sorted:
            extractor = str(rec.get("extractor") or "pre_extracted")
            text = str(rec.get("text") or "").strip()
            page_num = int(rec.get("page_number") or 1)
            rel_path = str(rec.get("relative_path") or source_name)

            if extractor == "skipped_low_text" or len(text) < 25:
                skipped_pages += 1
                continue

            pieces = split_text_into_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for local_idx, piece in enumerate(pieces):
                doc_chunks.append(
                    {
                        "chunk_id": f"{doc_id}_p{page_num}_{local_idx}",
                        "doc_id": doc_id,
                        "source": source_name,
                        "relative_path": rel_path,
                        "page": page_num,
                        "extractor": extractor,
                        "text": piece,
                    }
                )

        total_pages = len(page_records_sorted)
        status = (
            "Indexed"
            if doc_chunks
            else "Skipped (low/no extractable text)"
        )
        docs.append(
            {
                "doc_id": doc_id,
                "filename": source_name,
                "relative_path": str(page_records_sorted[0].get("relative_path") or source_name),
                "subject_tag": infer_document_subject(source_name),
                "pages": total_pages,
                "skipped_pages": skipped_pages,
                "chunk_count": len(doc_chunks),
                "origin": f"data/{file_path.name}",
                "status": status,
                "added_at": now_label,
            }
        )
        chunks.extend(doc_chunks)

    return docs, chunks, warnings


def load_local_data_folder(
    data_dir: Path,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> LoadedCorpus:
    """Load all supported extracted study materials from data_dir without modifying originals."""
    corpus = LoadedCorpus()
    if not data_dir.exists():
        return corpus

    ignored_dirs = {"knowledge_base", "__pycache__"}
    ignored_files = {"settings.json"}

    files_to_read: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if any(part in ignored_dirs for part in path.parts):
            continue
        if not path.is_file() or path.name in ignored_files:
            continue
        if path.name.endswith(".json.gz") or path.suffix.lower() in {".json", ".txt", ".md", ".pdf"}:
            files_to_read.append(path)

    now_label = datetime.now().strftime("%Y-%m-%d %H:%M")

    for path in files_to_read:
        name_lower = path.name.lower()
        if name_lower.endswith(".json.gz") or path.suffix.lower() == ".json":
            docs, chunks, warnings = _load_extracted_json_records(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            corpus.documents.extend(docs)
            corpus.chunks.extend(chunks)
            corpus.warnings.extend(warnings)
        elif path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
                pieces = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                doc_id = f"local_{uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex[:10]}"
                for idx, piece in enumerate(pieces):
                    corpus.chunks.append(
                        {
                            "chunk_id": f"{doc_id}_p1_{idx}",
                            "doc_id": doc_id,
                            "source": path.name,
                            "relative_path": str(path.relative_to(data_dir)),
                            "page": 1,
                            "text": piece,
                        }
                    )
                corpus.documents.append(
                    {
                        "doc_id": doc_id,
                        "filename": path.name,
                        "relative_path": str(path.relative_to(data_dir)),
                        "subject_tag": infer_document_subject(path.name),
                        "pages": 1,
                        "skipped_pages": 0,
                        "chunk_count": len(pieces),
                        "origin": f"data/{path.relative_to(data_dir)}",
                        "status": "Indexed" if pieces else "Skipped (empty)",
                        "added_at": now_label,
                    }
                )
            except Exception as exc:
                corpus.warnings.append(f"Failed to read '{path.name}': {exc}")
        elif path.suffix.lower() == ".pdf":
            try:
                res = extract_and_chunk_pdf(
                    filename=path.name,
                    pdf_bytes=path.read_bytes(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if res.success:
                    for ch in res.chunks:
                        ch["relative_path"] = str(path.relative_to(data_dir))
                    corpus.documents.append(
                        {
                            "doc_id": res.doc_id,
                            "filename": res.filename,
                            "relative_path": str(path.relative_to(data_dir)),
                            "subject_tag": infer_document_subject(res.filename),
                            "pages": res.page_count,
                            "skipped_pages": 0,
                            "chunk_count": len(res.chunks),
                            "origin": f"data/{path.relative_to(data_dir)}",
                            "status": "Indexed",
                            "added_at": now_label,
                        }
                    )
                    corpus.chunks.extend(res.chunks)
                else:
                    corpus.warnings.append(res.error)
            except Exception as exc:
                corpus.warnings.append(f"Failed to read '{path.name}': {exc}")

    return corpus
