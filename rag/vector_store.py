"""Local persistent FAISS / NumPy vector store for GATE study PDFs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import DATA_DIR, KB_DIR
from rag.embeddings import EmbeddingEngine
from rag.extracted_loader import infer_document_subject, load_local_data_folder
from rag.pdf_processor import PDFProcessResult, extract_and_chunk_pdf

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback handled cleanly
    faiss = None


@dataclass
class KBStatus:
    """Summary of local knowledge base health and contents."""

    document_count: int
    chunk_count: int
    embedder_backend: str
    index_backend: str
    ready: bool
    status_message: str


class LocalKnowledgeBase:
    """Stores PDF chunks locally and retrieves relevant excerpts via FAISS or NumPy."""

    def __init__(
        self,
        storage_dir: Path | None = None,
        data_dir: Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 750,
        chunk_overlap: int = 120,
    ) -> None:
        self.storage_dir = storage_dir or KB_DIR
        self.data_dir = data_dir if data_dir is not None else (DATA_DIR if storage_dir is None else None)
        self.docs_file = self.storage_dir / "documents.json"
        self.chunks_file = self.storage_dir / "chunks.json"
        self.embeddings_file = self.storage_dir / "embeddings.npy"
        self.faiss_file = self.storage_dir / "index.faiss"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = EmbeddingEngine(model_name=embedding_model)

        self.documents: List[dict] = []
        self.chunks: List[dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.faiss_index = None
        self._last_error: str = ""
        self._last_warnings: List[str] = []

        self._load_from_disk()

    @property
    def index_backend_name(self) -> str:
        return "FAISS (IndexFlatIP)" if faiss is not None else "NumPy Cosine Fallback"

    def _ensure_dir(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _build_index(self, vectors: np.ndarray) -> None:
        self.embeddings = vectors
        if vectors.size == 0 or len(vectors.shape) != 2 or vectors.shape[0] == 0:
            self.faiss_index = None
            return
        if faiss is not None:
            dim = int(vectors.shape[1])
            idx = faiss.IndexFlatIP(dim)
            idx.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self.faiss_index = idx
        else:
            self.faiss_index = None

    def _save_to_disk(self) -> None:
        self._ensure_dir()
        self.docs_file.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
        self.chunks_file.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        if self.embeddings.size > 0:
            np.save(self.embeddings_file, self.embeddings)
            if faiss is not None and self.faiss_index is not None:
                try:
                    serialized = faiss.serialize_index(self.faiss_index)
                    self.faiss_file.write_bytes(serialized.tobytes())
                except Exception:
                    faiss.write_index(self.faiss_index, str(self.faiss_file))
        else:
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            if self.faiss_file.exists():
                self.faiss_file.unlink()

    def _load_from_disk(self) -> None:
        try:
            if self.docs_file.exists():
                self.documents = json.loads(self.docs_file.read_text(encoding="utf-8"))
            if self.chunks_file.exists():
                self.chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))

            if self.chunks and faiss is not None and self.faiss_file.exists():
                try:
                    raw_bytes = self.faiss_file.read_bytes()
                    idx = faiss.deserialize_index(np.frombuffer(raw_bytes, dtype=np.uint8))
                except Exception:
                    idx = faiss.read_index(str(self.faiss_file))
                if idx.ntotal == len(self.chunks):
                    self.faiss_index = idx
                    if self.embeddings_file.exists():
                        self.embeddings = np.asarray(
                            np.load(self.embeddings_file), dtype=np.float32
                        )
                    return

            if self.chunks and self.embeddings_file.exists():
                loaded = np.load(self.embeddings_file)
                if loaded.shape[0] == len(self.chunks):
                    self._build_index(np.asarray(loaded, dtype=np.float32))
                    if faiss is not None and self.faiss_index is not None and not self.faiss_file.exists():
                        self._save_to_disk()
                else:
                    self.rebuild_index()
            elif self.chunks:
                self.rebuild_index()
            elif not self.documents and self.data_dir is not None and self.data_dir.exists():
                self.ingest_local_data(preserve_uploads=False)
        except Exception as exc:
            self._last_error = f"Knowledge base load warning: {exc}"
            self.documents = []
            self.chunks = []
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self.faiss_index = None

    def ingest_local_data(self, preserve_uploads: bool = True) -> dict:
        """Read extracted study files from data_dir (read-only) and build local FAISS index."""
        if self.data_dir is None or not self.data_dir.exists():
            return {"docs_total": len(self.documents), "chunks_total": len(self.chunks), "warnings": []}

        uploaded_docs = (
            [d for d in self.documents if d.get("origin") == "uploaded_pdf"]
            if preserve_uploads
            else []
        )
        uploaded_ids = {d["doc_id"] for d in uploaded_docs}
        uploaded_chunks = (
            [c for c in self.chunks if c.get("doc_id") in uploaded_ids]
            if preserve_uploads
            else []
        )

        corpus = load_local_data_folder(
            data_dir=self.data_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._last_warnings = corpus.warnings
        self.documents = corpus.documents + uploaded_docs
        self.chunks = corpus.chunks + uploaded_chunks
        self.rebuild_index()
        self._last_error = ""
        return {
            "docs_total": len(self.documents),
            "docs_indexed": sum(1 for d in self.documents if d.get("chunk_count", 0) > 0),
            "chunks_total": len(self.chunks),
            "warnings": corpus.warnings,
        }

    def rebuild_index(self) -> None:
        """Recompute embeddings and vector index for all stored chunks."""
        if not self.chunks:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self.faiss_index = None
            self._save_to_disk()
            return
        texts = [c["text"] for c in self.chunks]
        vectors = self.embedder.fit_and_embed_corpus(texts)
        self._build_index(vectors)
        self._save_to_disk()

    def add_pdf(self, filename: str, pdf_bytes: bytes) -> PDFProcessResult:
        """Extract, chunk, embed, and store an uploaded PDF file."""
        # Avoid duplicate filename re-upload silently stacking identical chunks
        if any(doc["filename"] == filename for doc in self.documents):
            existing_ids = [d["doc_id"] for d in self.documents if d["filename"] == filename]
            for old_id in existing_ids:
                self.documents = [d for d in self.documents if d["doc_id"] != old_id]
                self.chunks = [c for c in self.chunks if c["doc_id"] != old_id]

        result = extract_and_chunk_pdf(
            filename=filename,
            pdf_bytes=pdf_bytes,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not result.success:
            return result

        try:
            self.documents.append(
                {
                    "doc_id": result.doc_id,
                    "filename": result.filename,
                    "relative_path": result.filename,
                    "subject_tag": infer_document_subject(result.filename),
                    "pages": result.page_count,
                    "skipped_pages": 0,
                    "chunk_count": len(result.chunks),
                    "origin": "uploaded_pdf",
                    "status": "Indexed",
                    "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )
            self.chunks.extend(result.chunks)
            self.rebuild_index()
            self._last_error = ""
            return result
        except Exception as exc:
            self._last_error = f"Embedding/indexing failed: {exc}"
            return PDFProcessResult(
                success=False,
                doc_id=result.doc_id,
                filename=filename,
                page_count=result.page_count,
                error=self._last_error,
            )

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and rebuild the vector index."""
        self.documents = [d for d in self.documents if d["doc_id"] != doc_id]
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self.rebuild_index()

    def clear(self) -> None:
        """Remove all documents and chunks from the local knowledge base."""
        self.documents = []
        self.chunks = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.faiss_index = None
        self._last_error = ""
        self._save_to_disk()

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.22,
    ) -> List[Dict]:
        """Retrieve top-k chunks whose similarity exceeds min_score."""
        if not self.chunks or not query.strip():
            return []

        try:
            corpus_texts = [c["text"] for c in self.chunks]
            q_vec = self.embedder.embed_query(query, corpus_texts=corpus_texts)
            if q_vec.size == 0 or (self.faiss_index is None and self.embeddings.size == 0):
                return []

            # Realign vector space if index dimension differs from active embedder
            index_dim = (
                int(self.faiss_index.d)
                if (faiss is not None and self.faiss_index is not None)
                else int(self.embeddings.shape[1])
            )
            if q_vec.shape[1] != index_dim:
                self.rebuild_index()
                q_vec = self.embedder.embed_query(query, corpus_texts=corpus_texts)
                new_dim = (
                    int(self.faiss_index.d)
                    if (faiss is not None and self.faiss_index is not None)
                    else int(self.embeddings.shape[1])
                )
                if q_vec.shape[1] != new_dim:
                    return []

            effective_min_score = (
                min_score if self.embedder.is_neural else min(min_score, 0.08)
            )
            k = min(top_k, len(self.chunks))
            results: List[Dict] = []

            if faiss is not None and self.faiss_index is not None:
                scores, indices = self.faiss_index.search(
                    np.ascontiguousarray(q_vec, dtype=np.float32), k
                )
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    if float(score) >= effective_min_score:
                        item = dict(self.chunks[idx])
                        item["score"] = float(score)
                        results.append(item)
            else:
                sims = (self.embeddings @ q_vec.T).flatten()
                top_indices = np.argsort(sims)[::-1][:k]
                for idx in top_indices:
                    score = float(sims[idx])
                    if score >= effective_min_score:
                        item = dict(self.chunks[int(idx)])
                        item["score"] = score
                        results.append(item)

            return results
        except Exception as exc:
            self._last_error = f"Retrieval error: {exc}"
            return []

    def get_status(self) -> KBStatus:
        """Return structured status for the Settings & Status panel."""
        doc_count = len(self.documents)
        chunk_count = len(self.chunks)
        if self._last_error:
            msg = self._last_error
        elif doc_count == 0:
            msg = "Empty (upload GATE PDFs to enable document-grounded answers)"
        else:
            msg = f"Ready ({doc_count} document(s), {chunk_count} indexed chunk(s))"

        return KBStatus(
            document_count=doc_count,
            chunk_count=chunk_count,
            embedder_backend=self.embedder.backend_label if doc_count > 0 else "Ready (lazy-loaded)",
            index_backend=self.index_backend_name,
            ready=doc_count > 0 and not bool(self._last_error),
            status_message=msg,
        )
